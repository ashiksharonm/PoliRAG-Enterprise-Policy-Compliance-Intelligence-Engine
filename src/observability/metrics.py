"""Metrics collection for PoliRAG."""

from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, Summary

# Request metrics
request_count = Counter(
    "polirag_requests_total", "Total number of requests", ["endpoint", "method", "status"]
)

request_duration = Histogram(
    "polirag_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Ingestion metrics
ingestion_documents_total = Counter(
    "polirag_ingestion_documents_total",
    "Total documents ingested",
    ["format", "tenant_id", "status"],
)

ingestion_chunks_total = Counter(
    "polirag_ingestion_chunks_total", "Total chunks created", ["tenant_id"]
)

ingestion_duration = Histogram(
    "polirag_ingestion_duration_seconds",
    "Document ingestion duration",
    ["format"],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

# Embedding metrics
embedding_requests_total = Counter(
    "polirag_embedding_requests_total", "Total embedding requests", ["model", "cached"]
)

embedding_tokens_total = Counter(
    "polirag_embedding_tokens_total", "Total tokens embedded", ["model"]
)

embedding_duration = Histogram(
    "polirag_embedding_duration_seconds",
    "Embedding generation duration",
    ["model", "batch_size"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
)

embedding_cost = Summary(
    "polirag_embedding_cost_usd", "Embedding cost in USD", ["model"]
)

# Retrieval metrics
retrieval_queries_total = Counter(
    "polirag_retrieval_queries_total", "Total retrieval queries", ["tenant_id", "method"]
)

retrieval_duration = Histogram(
    "polirag_retrieval_duration_seconds",
    "Retrieval duration",
    ["method"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)

retrieval_results_count = Histogram(
    "polirag_retrieval_results_count",
    "Number of results retrieved",
    ["method"],
    buckets=(1, 5, 10, 20, 50, 100),
)

# Generation metrics
generation_requests_total = Counter(
    "polirag_generation_requests_total", "Total generation requests", ["model", "tenant_id"]
)

generation_tokens_total = Counter(
    "polirag_generation_tokens_total", "Total tokens generated", ["model", "type"]
)

generation_duration = Histogram(
    "polirag_generation_duration_seconds",
    "Generation duration",
    ["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

generation_confidence = Histogram(
    "polirag_generation_confidence",
    "Generation confidence score",
    ["tenant_id"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

generation_cost = Summary(
    "polirag_generation_cost_usd", "Generation cost in USD", ["model"]
)

# Evaluation metrics
eval_recall_at_k = Gauge(
    "polirag_eval_recall_at_k", "Recall@K evaluation metric", ["test_set", "k"]
)

eval_mrr = Gauge(
    "polirag_eval_mrr", "Mean Reciprocal Rank evaluation metric", ["test_set"]
)

eval_hallucination_rate = Gauge(
    "polirag_eval_hallucination_rate", "Hallucination rate", ["test_set"]
)

# Guardrails metrics
pii_detections_total = Counter(
    "polirag_pii_detections_total", "Total PII detections", ["pii_type", "tenant_id"]
)

rbac_denials_total = Counter(
    "polirag_rbac_denials_total", "Total RBAC denials", ["tenant_id", "role", "resource"]
)

rate_limit_hits_total = Counter(
    "polirag_rate_limit_hits_total", "Total rate limit hits", ["tenant_id"]
)

# System metrics
active_connections = Gauge(
    "polirag_active_connections", "Number of active connections"
)

vector_index_size = Gauge(
    "polirag_vector_index_size", "Size of vector index", ["index_name"]
)

cache_hits_total = Counter(
    "polirag_cache_hits_total", "Total cache hits", ["cache_type"]
)

cache_misses_total = Counter(
    "polirag_cache_misses_total", "Total cache misses", ["cache_type"]
)