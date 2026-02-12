"""FAISS vector store with metadata filtering."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID

import faiss
import numpy as np
from loguru import logger

from src.config import get_settings
from src.models import Chunk
from src.observability.metrics import vector_index_size


class FAISSVectorStore:
    """FAISS-based vector store with persistence."""

    def __init__(self, index_name: str = "main"):
        """Initialize FAISS vector store.

        Args:
            index_name: Name of the index
        """
        self.settings = get_settings()
        self.index_name = index_name
        self.dimension = self.settings.vector_dimension
        
        # Paths
        self.index_dir = Path(self.settings.faiss_index_path)
        self.metadata_dir = Path(self.settings.faiss_metadata_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_dir / f"{index_name}.index"
        self.metadata_file = self.metadata_dir / f"{index_name}_metadata.pkl"
        self.mapping_file = self.metadata_dir / f"{index_name}_mapping.json"
        
        # Initialize or load index
        self.index: Optional[faiss.Index] = None
        self.id_to_idx: Dict[str, int] = {}  # chunk_id -> faiss_idx
        self.idx_to_chunk: Dict[int, Chunk] = {}  # faiss_idx -> Chunk
        
        self._load_or_create_index()
        
        logger.info(f"FAISSVectorStore '{index_name}' initialized")

    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        if self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded existing index: {self.index.ntotal} vectors")
                
                # Load metadata
                if self.metadata_file.exists():
                    with open(self.metadata_file, "rb") as f:
                        data = pickle.load(f)
                        self.idx_to_chunk = data.get("idx_to_chunk", {})
                
                # Load mapping
                if self.mapping_file.exists():
                    with open(self.mapping_file, "r") as f:
                        self.id_to_idx = json.load(f)
                
                vector_index_size.labels(index_name=self.index_name).set(self.index.ntotal)
                
            except Exception as e:
                logger.error(f"Error loading index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create new FAISS index."""
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        if self.settings.faiss_index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.settings.faiss_index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            # Default to IP
            self.index = faiss.IndexFlatIP(self.dimension)
        
        logger.info(f"Created new {self.settings.faiss_index_type} index")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the index.

        Args:
            chunks: List of chunks with embeddings
        """
        if not chunks:
            return

        # Filter chunks with embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        
        if not valid_chunks:
            logger.warning("No chunks with embeddings to add")
            return

        logger.info(f"Adding {len(valid_chunks)} chunks to index '{self.index_name}'")
        
        # Prepare embeddings
        embeddings = np.array([c.embedding for c in valid_chunks], dtype=np.float32)
        
        # Normalize for cosine similarity (if using IndexFlatIP)
        if self.settings.faiss_index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings)
        
        # Get starting index
        start_idx = self.index.ntotal
        
        # Add to FAISS
        self.index.add(embeddings)
        
        # Update mappings
        for i, chunk in enumerate(valid_chunks):
            faiss_idx = start_idx + i
            chunk_id = str(chunk.id)
            
            self.id_to_idx[chunk_id] = faiss_idx
            self.idx_to_chunk[faiss_idx] = chunk
        
        logger.info(f"Index now contains {self.index.ntotal} vectors")
        vector_index_size.labels(index_name=self.index_name).set(self.index.ntotal)

    def search(
        self,
        query_embedding: List[float],
        k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Chunk, float]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Metadata filters (tenant_id, role_scope, etc.)

        Returns:
            List of (Chunk, score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Prepare query
        query = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.settings.faiss_index_type == "IndexFlatIP":
            faiss.normalize_L2(query)
        
        # Search with larger k if using filters
        search_k = k * 5 if filters else k
        search_k = min(search_k, self.index.ntotal)
        
        # Perform search
        distances, indices = self.index.search(query, search_k)
        
        # Convert to results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            chunk = self.idx_to_chunk.get(idx)
            if chunk is None:
                continue
            
            # Apply filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            
            # Convert distance to score (higher is better)
            score = float(distance)
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

    def get_chunk_by_id(self, chunk_id: UUID) -> Optional[Chunk]:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk if found
        """
        faiss_idx = self.id_to_idx.get(str(chunk_id))
        if faiss_idx is not None:
            return self.idx_to_chunk.get(faiss_idx)
        return None

    def save(self) -> None:
        """Save index and metadata to disk."""
        logger.info(f"Saving index '{self.index_name}' with {self.index.ntotal} vectors")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_file))
        
        # Save metadata
        with open(self.metadata_file, "wb") as f:
            pickle.dump({"idx_to_chunk": self.idx_to_chunk}, f)
        
        # Save mapping
        with open(self.mapping_file, "w") as f:
            json.dump(self.id_to_idx, f)
        
        logger.info(f"Index saved to {self.index_file}")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "index_name": self.index_name,
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.settings.faiss_index_type,
            "total_chunks": len(self.idx_to_chunk),
            "index_file": str(self.index_file),
            "index_size_mb": self.index_file.stat().st_size / (1024 * 1024) if self.index_file.exists() else 0
        }