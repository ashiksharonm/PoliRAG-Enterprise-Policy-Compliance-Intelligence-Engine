"""Text chunking strategies."""

import hashlib
import re
from typing import List, Optional
from uuid import UUID, uuid4

import tiktoken
from loguru import logger

from src.config import get_settings
from src.models import Chunk, Document


class TextChunker:
    """Recursive text chunking with structure preservation."""

    def __init__(self):
        """Initialize chunker."""
        self.settings = get_settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
        self.chunk_min_size = self.settings.chunk_min_size
        self.chunk_max_size = self.settings.chunk_max_size
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, using approximation")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximation: 1 token ≈ 4 characters
            return len(text) // 4

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk document content.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        content = document.metadata.get("content", "")
        
        if not content:
            logger.warning(f"No content found in document {document.id}")
            return []

        logger.info(f"Chunking document {document.id} ({len(content)} chars)")

        # Use recursive chunking strategy
        chunks = self._recursive_chunk(content)

        # Create Chunk objects
        chunk_objects = []
        for idx, (chunk_text, start_char, end_char) in enumerate(chunks):
            chunk = Chunk(
                id=uuid4(),
                document_id=document.id,
                content=chunk_text,
                content_hash=self._compute_hash(chunk_text),
                chunk_index=idx,
                token_count=self.count_tokens(chunk_text),
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "document_filename": document.filename,
                    "document_format": document.format,
                    "tenant_id": document.tenant_id,
                    "role_scope": document.role_scope,
                }
            )
            chunk_objects.append(chunk)

        logger.info(f"Created {len(chunk_objects)} chunks for document {document.id}")

        return chunk_objects

    def _recursive_chunk(self, text: str) -> List[tuple[str, int, int]]:
        """Recursively chunk text with structure preservation.

        Strategy:
        1. Split by major headers (##, ###)
        2. Split by paragraphs (double newline)
        3. Split by sentences (period + space)
        4. Split by words if still too large

        Args:
            text: Input text

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []
        
        # Separators in order of preference
        separators = [
            "\n## ",      # Markdown H2
            "\n### ",     # Markdown H3
            "\n\n",       # Paragraph break
            "\n",         # Line break
            ". ",         # Sentence end
            " ",          # Word boundary
            ""            # Character-level (last resort)
        ]

        chunks = self._split_recursive(text, separators, 0, 0)
        
        return chunks

    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        current_offset: int,
        depth: int
    ) -> List[tuple[str, int, int]]:
        """Recursive splitting helper.

        Args:
            text: Text to split
            separators: List of separators to try
            current_offset: Current character offset in original text
            depth: Recursion depth

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        token_count = self.count_tokens(text)

        # Base case: text is small enough
        if token_count <= self.chunk_max_size:
            if token_count >= self.chunk_min_size or depth == 0:
                return [(text, current_offset, current_offset + len(text))]
            else:
                # Too small, return empty (will be merged)
                return []

        # Try each separator
        if not separators:
            # Last resort: force split at token boundary
            return self._force_split(text, current_offset)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator:
            parts = text.split(separator)
        else:
            # Character-level split
            parts = list(text)

        # Process parts
        chunks = []
        current_chunk = ""
        chunk_start = current_offset

        for i, part in enumerate(parts):
            # Re-add separator (except for last part)
            if i > 0 and separator:
                part = separator + part

            test_chunk = current_chunk + part
            test_token_count = self.count_tokens(test_chunk)

            if test_token_count <= self.chunk_max_size:
                # Can add to current chunk
                current_chunk = test_chunk
            else:
                # Current chunk is full
                if current_chunk:
                    # Save current chunk
                    chunk_end = chunk_start + len(current_chunk)
                    chunks.append((current_chunk, chunk_start, chunk_end))
                    
                    # Add overlap
                    overlap_tokens = int(self.chunk_overlap * self.chunk_size / 100)
                    overlap_text = self._get_last_n_tokens(current_chunk, overlap_tokens)
                    
                    # Start new chunk with overlap
                    current_chunk = overlap_text + part
                    chunk_start = chunk_end - len(overlap_text)
                else:
                    # Part itself is too large, recurse
                    sub_chunks = self._split_recursive(
                        part,
                        remaining_separators,
                        chunk_start,
                        depth + 1
                    )
                    chunks.extend(sub_chunks)
                    chunk_start += len(part)
                    current_chunk = ""

        # Add final chunk
        if current_chunk and self.count_tokens(current_chunk) >= self.chunk_min_size:
            chunk_end = chunk_start + len(current_chunk)
            chunks.append((current_chunk, chunk_start, chunk_end))

        return chunks

    def _force_split(self, text: str, current_offset: int) -> List[tuple[str, int, int]]:
        """Force split text at token boundaries.

        Args:
            text: Text to split
            current_offset: Current offset

        Returns:
            List of chunks
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_tokens = 0
        char_offset = current_offset

        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.chunk_max_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((
                        chunk_text,
                        char_offset,
                        char_offset + len(chunk_text)
                    ))
                    char_offset += len(chunk_text) + 1
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(word)
            current_tokens += word_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((
                chunk_text,
                char_offset,
                char_offset + len(chunk_text)
            ))

        return chunks

    def _get_last_n_tokens(self, text: str, n: int) -> str:
        """Get last n tokens from text.

        Args:
            text: Input text
            n: Number of tokens

        Returns:
            Text with approximately n tokens
        """
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= n:
                return text
            overlap_tokens = tokens[-n:]
            return self.tokenizer.decode(overlap_tokens)
        else:
            # Approximation using words
            words = text.split()
            return " ".join(words[-n:])

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute SHA-256 hash of text.

        Args:
            text: Input text

        Returns:
            Hash string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TablePreservingChunker(TextChunker):
    """Chunker that preserves tables as complete units."""

    def _recursive_chunk(self, text: str) -> List[tuple[str, int, int]]:
        """Override to handle tables specially.

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        # Extract tables
        table_pattern = r'\[TABLE\](.*?)\[/TABLE\]'
        tables = list(re.finditer(table_pattern, text, re.DOTALL))

        if not tables:
            # No tables, use standard chunking
            return super()._recursive_chunk(text)

        chunks = []
        last_end = 0

        for table_match in tables:
            table_start = table_match.start()
            table_end = table_match.end()
            table_text = table_match.group(0)

            # Chunk text before table
            if table_start > last_end:
                pre_table_text = text[last_end:table_start]
                pre_chunks = super()._recursive_chunk(pre_table_text)
                # Adjust offsets
                adjusted_chunks = [
                    (chunk_text, start + last_end, end + last_end)
                    for chunk_text, start, end in pre_chunks
                ]
                chunks.extend(adjusted_chunks)

            # Add table as single chunk (if not too large)
            table_tokens = self.count_tokens(table_text)
            if table_tokens <= self.chunk_max_size:
                chunks.append((table_text, table_start, table_end))
            else:
                # Table is too large, split it
                logger.warning(f"Table is too large ({table_tokens} tokens), splitting")
                table_chunks = super()._recursive_chunk(table_text)
                adjusted_chunks = [
                    (chunk_text, start + table_start, end + table_start)
                    for chunk_text, start, end in table_chunks
                ]
                chunks.extend(adjusted_chunks)

            last_end = table_end

        # Chunk remaining text
        if last_end < len(text):
            remaining_text = text[last_end:]
            remaining_chunks = super()._recursive_chunk(remaining_text)
            adjusted_chunks = [
                (chunk_text, start + last_end, end + last_end)
                for chunk_text, start, end in remaining_chunks
            ]
            chunks.extend(adjusted_chunks)

        return chunks
