"""Hybrid retrieval with BM25 + Semantic search and reranking."""

import time
from typing import Dict, List, Optional, Any

from loguru import logger
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.embeddings.service import EmbeddingService
from src.models import Chunk, Query, RetrievedChunk
from src.observability.metrics import (
    retrieval_queries_total,
    retrieval_duration,
    retrieval_results_count,
)
from src.vectorstore.bm25_store import BM25Store
from src.vectorstore.faiss_store import FAISSVectorStore


class HybridRetriever:
    """Hybrid retrieval combining BM25 and semantic search."""

    def __init__(self, index_name: str = "main"):
        """Initialize hybrid retriever.

        Args:
            index_name: Index name
        """
        self.settings = get_settings()
        self.index_name = index_name
        
        # Initialize stores
        self.vector_store = FAISSVectorStore(index_name=index_name)
        self.bm25_store = BM25Store(index_name=index_name)
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        
        # Weights for hybrid search
        self.bm25_weight = self.settings.retrieval_bm25_weight
        self.semantic_weight = self.settings.retrieval_semantic_weight
        
        logger.info(
            f"HybridRetriever initialized (BM25: {self.bm25_weight}, "
            f"Semantic: {self.semantic_weight})"
        )

    async def retrieve(
        self,
        query: Query,
        k: int = None,
        method: str = "hybrid"
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for query.

        Args:
            query: Query object
            k: Number of results (default from config)
            method: "hybrid", "semantic", or "bm25"

        Returns:
            List of retrieved chunks with scores
        """
        if k is None:
            k = self.settings.retrieval_top_k

        start_time = time.time()
        
        logger.info(f"Retrieving with method '{method}' for query: {query.text[:100]}")

        # Prepare filters from query
        filters = {
            "tenant_id": query.tenant_id,
            "role": query.user_role,
        }
        if query.filters:
            filters.update(query.filters)

        # Retrieve based on method
        if method == "hybrid":
            results = await self._hybrid_search(query.text, k, filters)
        elif method == "semantic":
            results = await self._semantic_search(query.text, k, filters)
        elif method == "bm25":
            results = self._bm25_search(query.text, k, filters)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        duration = time.time() - start_time

        # Update metrics
        retrieval_queries_total.labels(
            tenant_id=query.tenant_id, method=method
        ).inc()
        retrieval_duration.labels(method=method).observe(duration)
        retrieval_results_count.labels(method=method).observe(len(results))

        logger.info(
            f"Retrieved {len(results)} chunks in {duration:.2f}s using {method}"
        )

        return results

    async def _hybrid_search(
        self,
        query_text: str,
        k: int,
        filters: Dict[str, Any]
    ) -> List[RetrievedChunk]:
        """Hybrid search combining BM25 and semantic.

        Args:
            query_text: Query text
            k: Number of results
            filters: Filters

        Returns:
            List of retrieved chunks
        """
        # Get results from both methods
        semantic_results = await self._semantic_search(query_text, k * 2, filters)
        bm25_results = self._bm25_search(query_text, k * 2, filters)

        # Normalize scores
        semantic_scores = self._normalize_scores(
            [r.score for r in semantic_results]
        )
        bm25_scores = self._normalize_scores(
            [r.score for r in bm25_results]
        )

        # Combine scores
        combined_scores: Dict[str, tuple[RetrievedChunk, float]] = {}

        for result, norm_score in zip(semantic_results, semantic_scores):
            chunk_id = str(result.chunk.id)
            weighted_score = norm_score * self.semantic_weight
            combined_scores[chunk_id] = (result, weighted_score)

        for result, norm_score in zip(bm25_results, bm25_scores):
            chunk_id = str(result.chunk.id)
            weighted_score = norm_score * self.bm25_weight
            
            if chunk_id in combined_scores:
                # Combine scores
                existing_result, existing_score = combined_scores[chunk_id]
                combined_scores[chunk_id] = (
                    existing_result,
                    existing_score + weighted_score
                )
            else:
                combined_scores[chunk_id] = (result, weighted_score)

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Create RetrievedChunk objects with hybrid method
        retrieved = []
        for rank, (result, score) in enumerate(sorted_results):
            retrieved_chunk = RetrievedChunk(
                chunk=result.chunk,
                score=score,
                rank=rank,
                retrieval_method="hybrid"
            )
            retrieved.append(retrieved_chunk)

        return retrieved

    async def _semantic_search(
        self,
        query_text: str,
        k: int,
        filters: Dict[str, Any]
    ) -> List[RetrievedChunk]:
        """Semantic search using embeddings.

        Args:
            query_text: Query text
            k: Number of results
            filters: Filters

        Returns:
            List of retrieved chunks
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query_text)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            filters=filters
        )

        # Convert to RetrievedChunk
        retrieved = []
        for rank, (chunk, score) in enumerate(results):
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=score,
                rank=rank,
                retrieval_method="semantic"
            )
            retrieved.append(retrieved_chunk)

        return retrieved

    def _bm25_search(
        self,
        query_text: str,
        k: int,
        filters: Dict[str, Any]
    ) -> List[RetrievedChunk]:
        """BM25 keyword search.

        Args:
            query_text: Query text
            k: Number of results
            filters: Filters

        Returns:
            List of retrieved chunks
        """
        # Search BM25 store
        results = self.bm25_store.search(
            query=query_text,
            k=k,
            filters=filters
        )

        # Convert to RetrievedChunk
        retrieved = []
        for rank, (chunk, score) in enumerate(results):
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=score,
                rank=rank,
                retrieval_method="bm25"
            )
            retrieved.append(retrieved_chunk)

        return retrieved

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range.

        Args:
            scores: List of scores

        Returns:
            Normalized scores
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]


class Reranker:
    """Cross-encoder based reranking."""

    def __init__(self):
        """Initialize reranker."""
        self.settings = get_settings()
        self.model_name = self.settings.rerank_model
        self.batch_size = self.settings.rerank_batch_size
        
        # Load cross-encoder model
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Reranker initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            self.model = None

    def rerank(
        self,
        query_text: str,
        retrieved_chunks: List[RetrievedChunk],
        top_k: int = None
    ) -> List[RetrievedChunk]:
        """Rerank retrieved chunks.

        Args:
            query_text: Query text
            retrieved_chunks: Retrieved chunks to rerank
            top_k: Number of top results to return (default from config)

        Returns:
            Reranked chunks
        """
        if top_k is None:
            top_k = self.settings.retrieval_rerank_top_k

        if not retrieved_chunks:
            return []

        if self.model is None:
            logger.warning("Reranker model not available, returning original order")
            return retrieved_chunks[:top_k]

        logger.info(f"Reranking {len(retrieved_chunks)} chunks to top-{top_k}")

        start_time = time.time()

        # Prepare pairs for cross-encoder
        pairs = [
            (query_text, chunk.chunk.content)
            for chunk in retrieved_chunks
        ]

        # Get reranking scores
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            
            # Update chunks with new scores
            for chunk, score in zip(retrieved_chunks, scores):
                chunk.score = float(score)
                chunk.retrieval_method = "reranked"
            
            # Sort by new scores
            reranked = sorted(
                retrieved_chunks,
                key=lambda x: x.score,
                reverse=True
            )[:top_k]
            
            # Update ranks
            for rank, chunk in enumerate(reranked):
                chunk.rank = rank
            
            duration = time.time() - start_time
            logger.info(f"Reranking completed in {duration:.2f}s")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return retrieved_chunks[:top_k]
