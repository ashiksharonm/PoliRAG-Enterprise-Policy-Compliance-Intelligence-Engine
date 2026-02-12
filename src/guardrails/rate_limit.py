"""Rate limiting for API requests."""

import time
from collections import defaultdict
from typing import Dict, Optional

from loguru import logger

from src.config import get_settings
from src.observability.metrics import rate_limit_hits_total


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self):
        """Initialize rate limiter."""
        self.settings = get_settings()
        self.enabled = self.settings.enable_rate_limiting
        self.requests_per_minute = self.settings.rate_limit_requests_per_minute
        
        # Token buckets per tenant
        self.buckets: Dict[str, Dict[str, any]] = defaultdict(
            lambda: {
                "tokens": self.requests_per_minute,
                "last_update": time.time()
            }
        )
        
        logger.info(
            f"RateLimiter initialized: {self.requests_per_minute} requests/minute"
        )

    def _refill_tokens(self, tenant_id: str) -> None:
        """Refill tokens based on time elapsed.

        Args:
            tenant_id: Tenant ID
        """
        bucket = self.buckets[tenant_id]
        now = time.time()
        time_elapsed = now - bucket["last_update"]
        
        # Refill rate: requests_per_minute / 60 = tokens per second
        tokens_to_add = time_elapsed * (self.requests_per_minute / 60)
        
        bucket["tokens"] = min(
            self.requests_per_minute,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = now

    def check_limit(self, tenant_id: str, cost: float = 1.0) -> tuple[bool, Optional[float]]:
        """Check if request is within rate limit.

        Args:
            tenant_id: Tenant ID
            cost: Cost of request in tokens (default 1.0)

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if not self.enabled:
            return True, None

        self._refill_tokens(tenant_id)
        
        bucket = self.buckets[tenant_id]
        
        if bucket["tokens"] >= cost:
            # Allow request
            bucket["tokens"] -= cost
            return True, None
        else:
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for tenant {tenant_id}")
            
            rate_limit_hits_total.labels(tenant_id=tenant_id).inc()
            
            # Calculate retry after
            tokens_needed = cost - bucket["tokens"]
            retry_after = tokens_needed / (self.requests_per_minute / 60)
            
            return False, retry_after

    def consume(self, tenant_id: str, cost: float = 1.0) -> bool:
        """Consume tokens for a request.

        Args:
            tenant_id: Tenant ID
            cost: Cost in tokens

        Returns:
            True if request allowed
        """
        allowed, _ = self.check_limit(tenant_id, cost)
        return allowed

    def get_remaining(self, tenant_id: str) -> float:
        """Get remaining tokens for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Number of remaining tokens
        """
        if not self.enabled:
            return float('inf')

        self._refill_tokens(tenant_id)
        return self.buckets[tenant_id]["tokens"]

    def reset(self, tenant_id: str) -> None:
        """Reset rate limit for tenant.

        Args:
            tenant_id: Tenant ID
        """
        if tenant_id in self.buckets:
            self.buckets[tenant_id] = {
                "tokens": self.requests_per_minute,
                "last_update": time.time()
            }
            logger.info(f"Rate limit reset for tenant {tenant_id}")