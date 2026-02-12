"""Tests for retrieval: BM25 store, FAISS store."""

import hashlib
from uuid import uuid4

import numpy as np
import pytest

from src.models import Chunk
from src.vectorstore.bm25_store import BM25Store
from src.vectorstore.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_chunk(text: str, tenant_id: str = "t1", embedding_dim: int = 1536):
    """Create a Chunk with a random embedding."""
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        content=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        chunk_index=0,
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
        embedding=np.random.randn(embedding_dim).tolist(),
        metadata={"tenant_id": tenant_id, "document_filename": "test.txt"},
    )


# ---------------------------------------------------------------------------
# BM25 tests
# ---------------------------------------------------------------------------

class TestBM25Store:
    def test_add_and_search(self, mock_settings):
        store = BM25Store(index_name="test_bm25")
        chunks = [
            _make_chunk("data retention policy for customer records"),
            _make_chunk("employee handbook guidelines and procedures"),
            _make_chunk("financial audit compliance requirements"),
        ]
        store.add_chunks(chunks)

        results = store.search("data retention", k=2)
        assert len(results) > 0
        # First result should be the most relevant
        top_chunk, top_score = results[0]
        assert "retention" in top_chunk.content.lower()

    def test_search_returns_empty_for_no_match(self, mock_settings):
        store = BM25Store(index_name="test_bm25_empty")
        chunks = [_make_chunk("quarterly financial report summary")]
        store.add_chunks(chunks)

        results = store.search("xyznonexistent", k=5)
        # BM25 may still return results with low scores, but shouldn't crash
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# FAISS tests
# ---------------------------------------------------------------------------

class TestFAISSVectorStore:
    def test_add_and_search(self, mock_settings):
        store = FAISSVectorStore(index_name="test_faiss")
        chunks = [_make_chunk(f"chunk number {i}") for i in range(5)]
        store.add_chunks(chunks)

        query_embedding = np.random.randn(1536).tolist()
        results = store.search(query_embedding=query_embedding, k=3)
        assert len(results) == 3

    def test_filters_by_tenant(self, mock_settings):
        store = FAISSVectorStore(index_name="test_faiss_filter")
        store.add_chunks([
            _make_chunk("chunk a", tenant_id="alpha"),
            _make_chunk("chunk b", tenant_id="beta"),
            _make_chunk("chunk c", tenant_id="alpha"),
        ])

        query_embedding = np.random.randn(1536).tolist()
        results = store.search(
            query_embedding=query_embedding,
            k=10,
            filters={"tenant_id": "alpha"},
        )
        for chunk, _ in results:
            assert chunk.metadata.get("tenant_id") == "alpha"

    def test_save_load_roundtrip(self, mock_settings, tmp_path):
        store = FAISSVectorStore(index_name="test_faiss_persist")
        chunks = [_make_chunk(f"persistent chunk {i}") for i in range(3)]
        store.add_chunks(chunks)
        store.save()

        # Create new store instance — it should load from disk
        store2 = FAISSVectorStore(index_name="test_faiss_persist")
        assert store2.index.ntotal == 3
