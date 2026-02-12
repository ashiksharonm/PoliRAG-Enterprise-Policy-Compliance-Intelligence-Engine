"""Tests for guardrails: PII detection, RBAC, rate limiting."""

import hashlib
import time
from unittest.mock import patch

import pytest

from src.guardrails.pii import PIIDetector
from src.guardrails.rbac import RBACEnforcer
from src.guardrails.rate_limit import RateLimiter
from src.models import Chunk, Document, DocumentFormat, Role


# ---------------------------------------------------------------------------
# PII tests (regex-only; spaCy model may not be installed in CI)
# ---------------------------------------------------------------------------

class TestPIIDetector:
    def test_detects_email(self, mock_settings):
        with patch.object(PIIDetector, "__init__", lambda self: None):
            detector = PIIDetector()
            detector.settings = mock_settings
            detector.nlp = None  # skip spacy

        detections = detector.detect("Contact us at admin@example.com for help.")
        types = {d["type"] for d in detections}
        assert "email" in types

    def test_detects_ssn(self, mock_settings):
        with patch.object(PIIDetector, "__init__", lambda self: None):
            detector = PIIDetector()
            detector.settings = mock_settings
            detector.nlp = None

        detections = detector.detect("SSN: 123-45-6789")
        types = {d["type"] for d in detections}
        assert "ssn" in types

    def test_redacts_correctly(self, mock_settings):
        with patch.object(PIIDetector, "__init__", lambda self: None):
            detector = PIIDetector()
            detector.settings = mock_settings
            detector.nlp = None

        text = "Email admin@example.com for info."
        redacted, detections = detector.redact(text)
        assert "[REDACTED_EMAIL]" in redacted
        assert "admin@example.com" not in redacted

    def test_no_false_positives_on_clean_text(self, mock_settings):
        with patch.object(PIIDetector, "__init__", lambda self: None):
            detector = PIIDetector()
            detector.settings = mock_settings
            detector.nlp = None

        detections = detector.detect("The quarterly report is due on Friday.")
        assert len(detections) == 0


# ---------------------------------------------------------------------------
# RBAC tests
# ---------------------------------------------------------------------------

class TestRBACEnforcer:
    def _make_chunk(self, roles):
        from uuid import uuid4
        return Chunk(
            id=uuid4(),
            document_id=uuid4(),
            content="test",
            content_hash=hashlib.sha256(b"test").hexdigest(),
            chunk_index=0,
            token_count=1,
            start_char=0,
            end_char=4,
            metadata={"role_scope": [r.value for r in roles], "tenant_id": "t1"},
        )

    def test_admin_accesses_all(self, mock_settings):
        enforcer = RBACEnforcer()
        chunk = self._make_chunk([Role.LEGAL])
        assert enforcer.can_access_chunk(Role.ADMIN, chunk) is True

    def test_read_only_denied_legal(self, mock_settings):
        enforcer = RBACEnforcer()
        chunk = self._make_chunk([Role.LEGAL])
        assert enforcer.can_access_chunk(Role.READ_ONLY, chunk) is False

    def test_compliance_accesses_read_only(self, mock_settings):
        enforcer = RBACEnforcer()
        chunk = self._make_chunk([Role.READ_ONLY])
        assert enforcer.can_access_chunk(Role.COMPLIANCE, chunk) is True

    def test_filter_chunks(self, mock_settings):
        enforcer = RBACEnforcer()
        chunks = [
            self._make_chunk([Role.LEGAL]),
            self._make_chunk([Role.READ_ONLY]),
            self._make_chunk([Role.ADMIN]),
        ]
        filtered = enforcer.filter_chunks(Role.READ_ONLY, chunks)
        assert len(filtered) == 1  # only READ_ONLY chunk


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_within_limit(self, mock_settings):
        limiter = RateLimiter()
        # 10 requests/minute configured in mock_settings
        for _ in range(5):
            allowed, _ = limiter.check_limit("t1")
            assert allowed is True

    def test_blocks_over_limit(self, mock_settings):
        limiter = RateLimiter()
        # Exhaust all tokens
        for _ in range(10):
            limiter.check_limit("t1")

        allowed, retry_after = limiter.check_limit("t1")
        assert allowed is False
        assert retry_after is not None and retry_after > 0

    def test_reset_restores_tokens(self, mock_settings):
        limiter = RateLimiter()
        for _ in range(10):
            limiter.check_limit("t1")
        limiter.reset("t1")
        allowed, _ = limiter.check_limit("t1")
        assert allowed is True
