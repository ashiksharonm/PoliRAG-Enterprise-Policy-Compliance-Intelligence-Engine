"""BM25 search implementation for hybrid retrieval."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.models import Chunk


class BM25Store:
    """BM25 text search store."""

    def __init__(self, index_name: str = "main"):
        """Initialize BM25 store.

        Args:
            index_name: Name of the index
        """
        self.settings = get_settings()
        self.index_name = index_name
        
        # Paths
        self.metadata_dir = Path(self.settings.faiss_metadata_path)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.bm25_file = self.metadata_dir / f"{index_name}_bm25.pkl"
        self.chunks_file = self.metadata_dir / f"{index_name}_bm25_chunks.pkl"
        
        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        
        self._load_or_create()
        
        logger.info(f"BM25Store '{index_name}' initialized")

    def _load_or_create(self) -> None:
        """Load existing BM25 index or create new one."""
        if self.bm25_file.exists() and self.chunks_file.exists():
            try:
                with open(self.bm25_file, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.tokenized_corpus = data["tokenized_corpus"]
                
                with open(self.chunks_file, "rb") as f:
                    self.chunks = pickle.load(f)
                
                logger.info(f"Loaded BM25 index with {len(self.chunks)} documents")
            except Exception as e:
                logger.error(f"Error loading BM25 index: {e}. Creating new.")
                self.bm25 = None
                self.chunks = []
                self.tokenized_corpus = []
        else:
            logger.info("No existing BM25 index found, will create on first add")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to BM25 index.

        Args:
            chunks: List of chunks
        """
        if not chunks:
            return

        logger.info(f"Adding {len(chunks)} chunks to BM25 index '{self.index_name}'")
        
        # Tokenize new chunks
        new_tokenized = [self._tokenize(chunk.content) for chunk in chunks]
        
        # Update corpus
        self.chunks.extend(chunks)
        self.tokenized_corpus.extend(new_tokenized)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"BM25 index now contains {len(self.chunks)} documents")

    def search(
        self,
        query: str,
        k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Chunk, float]]:
        """Search using BM25.

        Args:
            query: Query string
            k: Number of results
            filters: Metadata filters

        Returns:
            List of (Chunk, score) tuples
        """
        if self.bm25 is None or not self.chunks:
            logger.warning("BM25 index is empty")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        # Convert to results with filtering
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])
            
            # Apply filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            
            results.append((chunk, score))
            
            if len(results) >= k:
                break
        
        return results

    def _matches_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk matches filters.

        Args:
            chunk: Chunk to check
            filters: Filter dictionary

        Returns:
            True if matches
        """
        # Tenant filter
        if "tenant_id" in filters:
            chunk_tenant = chunk.metadata.get("tenant_id")
            if chunk_tenant != filters["tenant_id"]:
                return False
        
        # Role filter
        if "role" in filters:
            chunk_roles = chunk.metadata.get("role_scope", [])
            if filters["role"] not in chunk_roles:
                return False
        
        # Document ID filter
        if "document_id" in filters:
            if str(chunk.document_id) != str(filters["document_id"]):
                return False
        
        return True

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple whitespace tokenization with lowercasing
        return text.lower().split()

    def save(self) -> None:
        """Save BM25 index to disk."""
        logger.info(f"Saving BM25 index '{self.index_name}' with {len(self.chunks)} documents")
        
        # Save BM25 index
        with open(self.bm25_file, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "tokenized_corpus": self.tokenized_corpus
            }, f)
        
        # Save chunks
        with open(self.chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"BM25 index saved to {self.bm25_file}")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "index_name": self.index_name,
            "total_documents": len(self.chunks),
            "bm25_file": str(self.bm25_file),
            "index_size_mb": self.bm25_file.stat().st_size / (1024 * 1024) if self.bm25_file.exists() else 0
        }