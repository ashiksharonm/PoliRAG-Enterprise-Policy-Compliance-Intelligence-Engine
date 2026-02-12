"""Evaluation metrics for RAG system quality."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID

from loguru import logger

from src.config import get_settings
from src.models import EvaluationResult, Query, GeneratedResponse, RetrievedChunk
from src.observability.metrics import (
    eval_recall_at_k,
    eval_mrr,
    eval_hallucination_rate,
)


class EvaluationMetrics:
    """Calculate evaluation metrics for RAG system."""

    def __init__(self):
        """Initialize evaluation metrics calculator."""
        self.settings = get_settings()

    def calculate_recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = None
    ) -> float:
        """Calculate Recall@K.

        Recall@K = (# relevant docs in top K) / (# total relevant docs)

        Args:
            retrieved_ids: List of retrieved document/chunk IDs (in rank order)
            relevant_ids: List of relevant document/chunk IDs (ground truth)
            k: Consider top K results (default: retrieval_top_k)

        Returns:
            Recall@K score (0.0-1.0)
        """
        if k is None:
            k = self.settings.retrieval_top_k

        if not relevant_ids:
            logger.warning("No relevant IDs provided for Recall@K calculation")
            return 0.0

        # Consider only top K
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        # Count how many relevant docs are in top K
        relevant_in_top_k = len(top_k_retrieved.intersection(relevant_set))

        recall = relevant_in_top_k / len(relevant_set)

        logger.debug(
            f"Recall@{k}: {relevant_in_top_k}/{len(relevant_set)} = {recall:.3f}"
        )

        return recall

    def calculate_mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / (rank of first relevant document)

        Args:
            retrieved_ids: List of retrieved document/chunk IDs (in rank order)
            relevant_ids: List of relevant document/chunk IDs

        Returns:
            MRR score (0.0-1.0)
        """
        if not relevant_ids:
            logger.warning("No relevant IDs provided for MRR calculation")
            return 0.0

        relevant_set = set(relevant_ids)

        # Find rank of first relevant document (1-indexed)
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                mrr = 1.0 / rank
                logger.debug(f"MRR: First relevant at rank {rank}, MRR = {mrr:.3f}")
                return mrr

        # No relevant document found
        logger.debug("MRR: No relevant document found in results")
        return 0.0

    def calculate_precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = None
    ) -> float:
        """Calculate Precision@K.

        Precision@K = (# relevant docs in top K) / K

        Args:
            retrieved_ids: List of retrieved document/chunk IDs
            relevant_ids: List of relevant document/chunk IDs
            k: Consider top K results

        Returns:
            Precision@K score (0.0-1.0)
        """
        if k is None:
            k = self.settings.retrieval_top_k

        if not relevant_ids or k == 0:
            return 0.0

        top_k_retrieved = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)

        relevant_in_top_k = len(top_k_retrieved.intersection(relevant_set))

        precision = relevant_in_top_k / k

        logger.debug(f"Precision@{k}: {relevant_in_top_k}/{k} = {precision:.3f}")

        return precision

    def calculate_ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids_with_scores: Dict[str, float],
        k: int = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            retrieved_ids: List of retrieved document/chunk IDs
            relevant_ids_with_scores: Dict mapping doc IDs to relevance scores
            k: Consider top K results

        Returns:
            NDCG@K score (0.0-1.0)
        """
        if k is None:
            k = self.settings.retrieval_top_k

        if not relevant_ids_with_scores:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            relevance = relevant_ids_with_scores.get(doc_id, 0.0)
            # DCG formula: sum(rel_i / log2(i+2))
            dcg += relevance / (i + 2)

        # Calculate IDCG (ideal DCG)
        sorted_relevances = sorted(
            relevant_ids_with_scores.values(),
            reverse=True
        )[:k]
        idcg = sum(rel / (i + 2) for i, rel in enumerate(sorted_relevances))

        if idcg == 0.0:
            return 0.0

        ndcg = dcg / idcg

        logger.debug(f"NDCG@{k}: {dcg:.3f}/{idcg:.3f} = {ndcg:.3f}")

        return ndcg

    def evaluate_retrieval(
        self,
        retrieved_chunks: List[RetrievedChunk],
        relevant_chunk_ids: List[str],
        k_values: List[int] = None
    ) -> Dict[str, float]:
        """Evaluate retrieval quality.

        Args:
            retrieved_chunks: Retrieved chunks
            relevant_chunk_ids: Ground truth relevant chunk IDs
            k_values: List of K values to evaluate (default: [5, 10, 20])

        Returns:
            Dictionary of metrics
        """
        if k_values is None:
            k_values = [5, 10, 20]

        retrieved_ids = [str(chunk.chunk.id) for chunk in retrieved_chunks]

        metrics = {}

        # Calculate metrics for each K
        for k in k_values:
            metrics[f"recall@{k}"] = self.calculate_recall_at_k(
                retrieved_ids, relevant_chunk_ids, k
            )
            metrics[f"precision@{k}"] = self.calculate_precision_at_k(
                retrieved_ids, relevant_chunk_ids, k
            )

        # MRR (not K-dependent)
        metrics["mrr"] = self.calculate_mrr(retrieved_ids, relevant_chunk_ids)

        return metrics