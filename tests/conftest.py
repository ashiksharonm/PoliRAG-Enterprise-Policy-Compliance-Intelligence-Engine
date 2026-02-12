"""Shared fixtures for PoliRAG test suite.

All tests use mocked settings — no real OpenAI key or spaCy model required.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from src.config import Settings
from src.models import Chunk, Document, DocumentFormat, Role


# ---------------------------------------------------------------------------
# Settings fixture — overrides env so nothing touches the real .env
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_settings(tmp_path):
    """Provide a Settings instance backed by temp directories."""
    env_overrides = {
        "OPENAI_API_KEY": "test-key-not-real",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
        "OPENAI_EMBEDDING_DIMENSIONS": "1536",
        "OPENAI_LLM_MODEL": "gpt-3.5-turbo",
        "VECTOR_DIMENSION": "1536",
        "DATA_RAW_PATH": str(tmp_path / "raw"),
        "DATA_STAGED_PATH": str(tmp_path / "staged"),
        "DATA_MANIFESTS_PATH": str(tmp_path / "manifests"),
        "FAISS_INDEX_PATH": str(tmp_path / "faiss"),
        "FAISS_METADATA_PATH": str(tmp_path / "metadata"),
        "EVAL_DATASET_PATH": str(tmp_path / "eval" / "golden_qa.json"),
        "ENABLE_PII_REDACTION": "true",
        "ENABLE_RBAC": "true",
        "ENABLE_RATE_LIMITING": "true",
        "RATE_LIMIT_REQUESTS_PER_MINUTE": "10",
        "CHUNK_SIZE": "512",
        "CHUNK_OVERLAP": "50",
        "LOG_LEVEL": "WARNING",
    }

    with patch.dict(os.environ, env_overrides, clear=False):
        # Reset the cached settings singleton
        import src.config as cfg
        cfg._settings = None
        settings = cfg.get_settings()
        # Ensure temp dirs exist
        for p in [
            settings.data_raw_path, settings.data_staged_path,
            settings.data_manifests_path, settings.faiss_index_path,
            settings.faiss_metadata_path,
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)
        yield settings
        cfg._settings = None


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

SAMPLE_POLICY_TEXT = """Data Retention Policy

1. Purpose
This policy establishes requirements for the retention and disposal of
company records and data in compliance with applicable laws and regulations.

2. Scope
This policy applies to all employees, contractors, and third‑party service
providers who create, receive, or manage company records.

3. Retention Periods
- Financial records: 7 years
- Employee records: Duration of employment + 5 years
- Customer data: Duration of relationship + 3 years
- Email communications: 3 years

4. Disposal
Records that have exceeded their retention period must be securely destroyed.
Paper records shall be cross‑cut shredded. Electronic records shall be
permanently deleted using approved tools.

5. Exceptions
Legal hold notices override normal retention schedules. Contact the Legal
department before destroying any records subject to litigation holds.
"""


@pytest.fixture
def sample_text_file(tmp_path) -> Path:
    """Create a sample text file on disk."""
    path = tmp_path / "sample_policy.txt"
    path.write_text(SAMPLE_POLICY_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def sample_markdown_file(tmp_path) -> Path:
    """Create a sample markdown file on disk."""
    content = "# Security Policy\n\nAll passwords must be **12+ characters**.\n"
    path = tmp_path / "security.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_document() -> Document:
    """Return a pre-built Document with inline content."""
    return Document(
        id=uuid4(),
        filename="retention_policy.txt",
        format=DocumentFormat.TEXT,
        content_hash="abc123deadbeef",
        size_bytes=len(SAMPLE_POLICY_TEXT.encode()),
        tenant_id="test-tenant",
        role_scope=[Role.COMPLIANCE, Role.READ_ONLY],
        version=1,
        metadata={"content": SAMPLE_POLICY_TEXT},
    )


@pytest.fixture
def sample_chunks(sample_document) -> List[Chunk]:
    """Return a list of Chunk objects with fake embeddings."""
    chunks = []
    paragraphs = [p.strip() for p in SAMPLE_POLICY_TEXT.split("\n\n") if p.strip()]
    running_pos = 0
    for i, text in enumerate(paragraphs):
        chunk = Chunk(
            id=uuid4(),
            document_id=sample_document.id,
            content=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            chunk_index=i,
            token_count=len(text.split()),
            start_char=running_pos,
            end_char=running_pos + len(text),
            embedding=np.random.randn(1536).tolist(),
            metadata={
                "document_filename": sample_document.filename,
                "tenant_id": sample_document.tenant_id,
                "role_scope": [r.value for r in sample_document.role_scope],
            },
        )
        chunks.append(chunk)
        running_pos += len(text) + 2  # account for paragraph separator
    return chunks
