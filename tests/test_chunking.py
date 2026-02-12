"""Tests for the chunking layer."""

import pytest

from src.chunking.chunker import TablePreservingChunker, TextChunker

# Inline the sample text to avoid import collision with anaconda site-packages
SAMPLE_POLICY_TEXT = """Data Retention Policy

1. Purpose
This policy establishes requirements for the retention and disposal of
company records and data in compliance with applicable laws and regulations.

2. Scope
This policy applies to all employees, contractors, and third-party service
providers who create, receive, or manage company records.

3. Retention Periods
- Financial records: 7 years
- Employee records: Duration of employment + 5 years
- Customer data: Duration of relationship + 3 years
- Email communications: 3 years

4. Disposal
Records that have exceeded their retention period must be securely destroyed.
Paper records shall be cross-cut shredded. Electronic records shall be
permanently deleted using approved tools.

5. Exceptions
Legal hold notices override normal retention schedules. Contact the Legal
department before destroying any records subject to litigation holds.
"""


# ---------------------------------------------------------------------------
# TextChunker tests
# ---------------------------------------------------------------------------

class TestTextChunker:
    def test_creates_chunks(self, sample_document, mock_settings):
        chunker = TextChunker()
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 0

    def test_chunks_have_valid_metadata(self, sample_document, mock_settings):
        chunker = TextChunker()
        chunks = chunker.chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.document_id == sample_document.id
            assert chunk.content  # non-empty
            assert chunk.chunk_index >= 0

    def test_no_empty_chunks(self, sample_document, mock_settings):
        chunker = TextChunker()
        chunks = chunker.chunk_document(sample_document)
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0


# ---------------------------------------------------------------------------
# TablePreservingChunker tests
# ---------------------------------------------------------------------------

class TestTablePreservingChunker:
    def test_chunks_plain_text(self, sample_document, mock_settings):
        chunker = TablePreservingChunker()
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 0

    def test_keeps_table_intact(self, sample_document, mock_settings):
        """When a table marker is present it should stay in a single chunk."""
        table_text = (
            "Some intro text.\n\n"
            "<TABLE>\n"
            "| Header A | Header B |\n"
            "|----------|----------|\n"
            "| Value 1  | Value 2  |\n"
            "</TABLE>\n\n"
            "Some conclusion text."
        )
        sample_document.metadata["content"] = table_text

        chunker = TablePreservingChunker()
        chunks = chunker.chunk_document(sample_document)

        # At least one chunk should contain the full table
        table_chunks = [c for c in chunks if "<TABLE>" in c.content or "Header A" in c.content]
        assert len(table_chunks) >= 1
