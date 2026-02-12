"""Embedding cache for content-hash based caching."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.config import get_settings
from src.observability.metrics import cache_hits_total, cache_misses_total


class EmbeddingCache:
    """SQLite-based embedding cache with TTL support."""

    def __init__(self, cache_path: Optional[Path] = None):
        """Initialize embedding cache.
        
        Args:
            cache_path: Path to cache database
        """
        self.settings = get_settings()
        
        if cache_path is None:
            cache_path = Path("./indexes/metadata/embedding_cache.db")
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.cache_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP NOT NULL,
                access_count INTEGER DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Embedding cache initialized at {self.cache_path}")
    
    def get(self, content_hash: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache.
        
        Args:
            content_hash: Content hash
            model: Model name
            
        Returns:
            Embedding if found and not expired
        """
        if not self.settings.enable_embedding_cache:
            return None
        
        conn = sqlite3.connect(str(self.cache_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT embedding, created_at FROM embeddings WHERE content_hash = ? AND model = ?",
            (content_hash, model)
        )
        
        row = cursor.fetchone()
        
        if row is None:
            conn.close()
            cache_misses_total.labels(cache_type="embedding").inc()
            return None
        
        embedding_json, created_at_str = row
        
        # Check TTL
        created_at = datetime.fromisoformat(created_at_str)
        ttl = timedelta(seconds=self.settings.cache_ttl_seconds)
        
        if datetime.utcnow() - created_at > ttl:
            # Expired
            logger.debug(f"Cache entry expired for hash {content_hash[:8]}")
            conn.close()
            cache_misses_total.labels(cache_type="embedding").inc()
            return None
        
        # Update access count and timestamp
        cursor.execute(
            "UPDATE embeddings SET last_accessed = ?, access_count = access_count + 1 WHERE content_hash = ? AND model = ?",
            (datetime.utcnow().isoformat(), content_hash, model)
        )
        conn.commit()
        conn.close()
        
        # Parse embedding
        embedding = json.loads(embedding_json)
        
        cache_hits_total.labels(cache_type="embedding").inc()
        logger.debug(f"Cache hit for hash {content_hash[:8]}")
        
        return embedding
    
    def set(
        self,
        content_hash: str,
        embedding: List[float],
        model: str,
        dimensions: int,
        token_count: int
    ) -> None:
        """Store embedding in cache.
        
        Args:
            content_hash: Content hash
            embedding: Embedding vector
            model: Model name
            dimensions: Embedding dimensions
            token_count: Token count
        """
        if not self.settings.enable_embedding_cache:
            return
        
        conn = sqlite3.connect(str(self.cache_path))
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        embedding_json = json.dumps(embedding)
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO embeddings 
            (content_hash, embedding, model, dimensions, token_count, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (content_hash, embedding_json, model, dimensions, token_count, now, now)
        )
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Cached embedding for hash {content_hash[:8]}")
    
    def get_batch(
        self,
        content_hashes: List[str],
        model: str
    ) -> dict[str, List[float]]:
        """Get multiple embeddings from cache.
        
        Args:
            content_hashes: List of content hashes
            model: Model name
            
        Returns:
            Dictionary mapping hash to embedding (only cached entries)
        """
        if not self.settings.enable_embedding_cache or not content_hashes:
            return {}
        
        conn = sqlite3.connect(str(self.cache_path))
        cursor = conn.cursor()
        
        placeholders = ",".join(["?"] * len(content_hashes))
        query = f"""
            SELECT content_hash, embedding, created_at 
            FROM embeddings 
            WHERE content_hash IN ({placeholders}) AND model = ?
        """
        
        cursor.execute(query, (*content_hashes, model))
        rows = cursor.fetchall()
        
        result = {}
        ttl = timedelta(seconds=self.settings.cache_ttl_seconds)
        hashes_to_update = []
        
        for content_hash, embedding_json, created_at_str in rows:
            # Check TTL
            created_at = datetime.fromisoformat(created_at_str)
            if datetime.utcnow() - created_at <= ttl:
                result[content_hash] = json.loads(embedding_json)
                hashes_to_update.append(content_hash)
        
        # Update access stats for found items
        if hashes_to_update:
            placeholders = ",".join(["?"] * len(hashes_to_update))
            cursor.execute(
                f"""
                UPDATE embeddings 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE content_hash IN ({placeholders}) AND model = ?
                """,
                (datetime.utcnow().isoformat(), *hashes_to_update, model)
            )
            conn.commit()
        
        conn.close()
        
        # Update metrics
        cache_hits_total.labels(cache_type="embedding").inc(len(result))
        cache_misses_total.labels(cache_type="embedding").inc(
            len(content_hashes) - len(result)
        )
        
        logger.debug(f"Batch cache: {len(result)}/{len(content_hashes)} hits")
        
        return result
    
    def clear_expired(self) -> int:
        """Clear expired cache entries.
        
        Returns:
            Number of entries deleted
        """
        conn = sqlite3.connect(str(self.cache_path))
        cursor = conn.cursor()
        
        ttl = timedelta(seconds=self.settings.cache_ttl_seconds)
        cutoff = (datetime.utcnow() - ttl).isoformat()
        
        cursor.execute(
            "DELETE FROM embeddings WHERE created_at < ?",
            (cutoff,)
        )
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted > 0:
            logger.info(f"Cleared {deleted} expired cache entries")
        
        return deleted
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        conn = sqlite3.connect(str(self.cache_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), SUM(access_count) FROM embeddings")
        total_entries, total_accesses = cursor.fetchone()
        
        cursor.execute("SELECT DISTINCT model FROM embeddings")
        models = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "total_entries": total_entries or 0,
            "total_accesses": total_accesses or 0,
            "models": models,
            "cache_path": str(self.cache_path)
        }