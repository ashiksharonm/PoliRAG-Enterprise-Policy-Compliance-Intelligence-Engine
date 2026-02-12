"""Evaluation runner for automated testing."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger

from src.config import get_settings
from src.eval.golden_dataset import GoldenDataset, GoldenDatasetManager, GoldenQA
from src.eval.metrics import EvaluationMetrics
from src.generation.generator import HallucinationDetector, ResponseGenerator
from src.models import EvaluationResult, Query, Role
from src.observability.metrics import (
    eval_recall_at_k,
    eval_mrr,
    eval_hallucination_rate,
)
from src.retrieval.hybrid import HybridRetriever, Reranker


class EvaluationRunner:
    """Run automated evaluation tests."""

    def __init__(self, index_name: str = "main"):
        """Initialize evaluation runner.

        Args:
            index_name: Index name to evaluate
        """
        self.settings = get_settings()
        self.index_name = index_name

        # Initialize components
        self.retriever = HybridRetriever(index_name=index_name)
        self.reranker = Reranker()
        self.generator = ResponseGenerator()
        self.hallucination_detector = HallucinationDetector()
        self.metrics_calculator = EvaluationMetrics()
        self.dataset_manager = GoldenDatasetManager()

        logger.info(f"EvaluationRunner initialized for index '{index_name}'")

    async def evaluate_single_query(
        self,
        qa: GoldenQA,
        tenant_id: str = "default",
        user_role: Role = Role.ADMIN
    ) -> Dict[str, Any]:
        """Evaluate single Q&A pair.

        Args:
            qa: Golden Q&A pair
            tenant_id: Tenant ID
            user_role: User role

        Returns:
            Evaluation result dictionary
        """
        logger.info(f"Evaluating query: {qa.question[:100]}")

        start_time = time.time()

        # Create query
        query = Query(
            text=qa.question,
            tenant_id=tenant_id,
            user_role=user_role
        )

        try:
            # Step 1: Retrieve
            retrieved_chunks = await self.retriever.retrieve(
                query=query,
                method="hybrid"
            )

            # Step 2: Rerank
            reranked_chunks = self.reranker.rerank(
                query_text=query.text,
                retrieved_chunks=retrieved_chunks
            )

            # Step 3: Generate response
            response = await self.generator.generate(
                query=query,
                retrieved_chunks=reranked_chunks
            )

            # Step 4: Detect hallucination
            hallucination_result = self.hallucination_detector.detect(
                response=response,
                query_text=query.text
            )

            # Step 5: Calculate retrieval metrics
            retrieved_ids = [str(chunk.chunk.id) for chunk in retrieved_chunks]
            retrieval_metrics = self.metrics_calculator.evaluate_retrieval(
                retrieved_chunks=retrieved_chunks,
                relevant_chunk_ids=qa.relevant_chunk_ids,
                k_values=[5, 10, 20]
            )

            duration = time.time() - start_time

            result = {
                "qa_id": qa.id,
                "question": qa.question,
                "category": qa.category,
                "difficulty": qa.difficulty,
                "retrieval_metrics": retrieval_metrics,
                "generation_metrics": {
                    "confidence": response.confidence,
                    "tokens_used": response.tokens_used,
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.metadata.get("cost_usd", 0.0)
                },
                "hallucination_metrics": hallucination_result,
                "num_retrieved": len(retrieved_chunks),
                "num_reranked": len(reranked_chunks),
                "num_citations": len(response.citations),
                "total_duration_ms": duration * 1000,
                "success": True,
                "error": None
            }

            logger.info(
                f"Evaluation complete: Recall@5={retrieval_metrics.get('recall@5', 0):.2f}, "
                f"Confidence={response.confidence:.2f}, "
                f"Hallucination={hallucination_result['hallucination_score']:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            return {
                "qa_id": qa.id,
                "question": qa.question,
                "category": qa.category,
                "difficulty": qa.difficulty,
                "success": False,
                "error": str(e)
            }

    async def evaluate_dataset(
        self,
        dataset: GoldenDataset,
        tenant_id: str = "default",
        user_role: Role = Role.ADMIN,
        category_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate entire golden dataset.

        Args:
            dataset: Golden dataset
            tenant_id: Tenant ID
            user_role: User role
            category_filter: Filter by category
            difficulty_filter: Filter by difficulty

        Returns:
            Evaluation result
        """
        logger.info(f"Starting dataset evaluation: {dataset.name}")

        # Filter Q&A pairs
        qa_pairs = dataset.qa_pairs
        if category_filter:
            qa_pairs = [qa for qa in qa_pairs if qa.category == category_filter]
        if difficulty_filter:
            qa_pairs = [qa for qa in qa_pairs if qa.difficulty == difficulty_filter]

        logger.info(f"Evaluating {len(qa_pairs)} Q&A pairs")

        # Evaluate each query
        results = []
        for qa in qa_pairs:
            result = await self.evaluate_single_query(qa, tenant_id, user_role)
            results.append(result)

            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)

        # Aggregate metrics
        aggregated = self._aggregate_results(results)

        # Create evaluation result
        evaluation_result = EvaluationResult(
            test_set_name=dataset.name,
            recall_at_k=aggregated["recall@5"],
            mean_reciprocal_rank=aggregated["mrr"],
            hallucination_rate=aggregated["hallucination_rate"],
            avg_confidence=aggregated["avg_confidence"],
            avg_latency_ms=aggregated["avg_latency_ms"],
            total_queries=len(results),
            passed=self._check_thresholds(aggregated),
            details=aggregated
        )

        # Update Prometheus metrics
        eval_recall_at_k.labels(test_set=dataset.name, k="5").set(aggregated["recall@5"])
        eval_recall_at_k.labels(test_set=dataset.name, k="10").set(aggregated["recall@10"])
        eval_recall_at_k.labels(test_set=dataset.name, k="20").set(aggregated["recall@20"])
        eval_mrr.labels(test_set=dataset.name).set(aggregated["mrr"])
        eval_hallucination_rate.labels(test_set=dataset.name).set(aggregated["hallucination_rate"])

        logger.info(
            f"Dataset evaluation complete: "
            f"Recall@5={aggregated['recall@5']:.2f}, "
            f"MRR={aggregated['mrr']:.2f}, "
            f"Hallucination Rate={aggregated['hallucination_rate']:.2f}, "
            f"Passed={evaluation_result.passed}"
        )

        return evaluation_result

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate evaluation results.

        Args:
            results: List of individual results

        Returns:
            Aggregated metrics
        """
        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            logger.warning("No successful results to aggregate")
            return {
                "recall@5": 0.0,
                "recall@10": 0.0,
                "recall@20": 0.0,
                "precision@5": 0.0,
                "mrr": 0.0,
                "hallucination_rate": 1.0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
                "total_cost_usd": 0.0,
                "success_rate": 0.0
            }

        # Calculate averages
        aggregated = {}

        # Retrieval metrics
        for metric in ["recall@5", "recall@10", "recall@20", "precision@5", "mrr"]:
            values = [
                r["retrieval_metrics"].get(metric, 0.0)
                for r in successful_results
                if "retrieval_metrics" in r
            ]
            aggregated[metric] = sum(values) / len(values) if values else 0.0

        # Hallucination rate
        hallucination_scores = [
            r["hallucination_metrics"].get("hallucination_score", 0.0)
            for r in successful_results
            if "hallucination_metrics" in r
        ]
        aggregated["hallucination_rate"] = (
            sum(hallucination_scores) / len(hallucination_scores)
            if hallucination_scores else 0.0
        )

        # Generation metrics
        confidences = [
            r["generation_metrics"].get("confidence", 0.0)
            for r in successful_results
            if "generation_metrics" in r
        ]
        aggregated["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0

        latencies = [
            r["generation_metrics"].get("latency_ms", 0.0)
            for r in successful_results
            if "generation_metrics" in r
        ]
        aggregated["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0

        costs = [
            r["generation_metrics"].get("cost_usd", 0.0)
            for r in successful_results
            if "generation_metrics" in r
        ]
        aggregated["total_cost_usd"] = sum(costs)

        # Success rate
        aggregated["success_rate"] = len(successful_results) / len(results)

        # Category breakdown
        by_category = {}
        for result in successful_results:
            category = result.get("category", "unknown")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)

        aggregated["by_category"] = {
            category: {
                "count": len(results),
                "avg_confidence": sum(
                    r["generation_metrics"].get("confidence", 0.0)
                    for r in results
                ) / len(results)
            }
            for category, results in by_category.items()
        }

        return aggregated

    def _check_thresholds(self, aggregated: Dict[str, Any]) -> bool:
        """Check if metrics meet minimum thresholds.

        Args:
            aggregated: Aggregated metrics

        Returns:
            True if all thresholds met
        """
        checks = [
            aggregated["recall@5"] >= self.settings.eval_min_recall_at_k,
            aggregated["mrr"] >= self.settings.eval_min_mrr,
            aggregated["hallucination_rate"] <= self.settings.eval_max_hallucination_rate,
        ]

        passed = all(checks)

        if not passed:
            logger.warning(
                f"Evaluation thresholds not met: "
                f"Recall@5={aggregated['recall@5']:.2f} (min: {self.settings.eval_min_recall_at_k}), "
                f"MRR={aggregated['mrr']:.2f} (min: {self.settings.eval_min_mrr}), "
                f"Hallucination={aggregated['hallucination_rate']:.2f} (max: {self.settings.eval_max_hallucination_rate})"
            )

        return passed

    def save_results(
        self,
        evaluation_result: EvaluationResult,
        output_path: Optional[Path] = None
    ) -> None:
        """Save evaluation results to file.

        Args:
            evaluation_result: Evaluation result
            output_path: Output path
        """
        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"./data/eval/results_{timestamp}.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                evaluation_result.model_dump(mode="json"),
                f,
                indent=2,
                default=str
            )

        logger.info(f"Evaluation results saved to {output_path}")
