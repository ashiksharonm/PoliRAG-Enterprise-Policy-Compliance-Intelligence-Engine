"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.models import Role, DocumentFormat


# Request models

class IngestRequest(BaseModel):
    """Document ingestion request."""

    file_path: str = Field(..., description="Path to file to ingest")
    tenant_id: str = Field(default="default", description="Tenant ID")
    role_scope: List[Role] = Field(
        default=[Role.READ_ONLY],
        description="Roles that can access this document"
    )
    skip_if_duplicate: bool = Field(default=True, description="Skip if duplicate found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryRequest(BaseModel):
    """Query request."""

    text: str = Field(..., description="Query text")
    tenant_id: str = Field(default="default", description="Tenant ID")
    user_role: Role = Field(default=Role.READ_ONLY, description="User role")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    retrieval_method: str = Field(
        default="hybrid",
        description="Retrieval method: hybrid, semantic, or bm25"
    )
    top_k: Optional[int] = Field(None, description="Number of results to return")


class EvaluateRequest(BaseModel):
    """Evaluation request."""

    dataset_name: str = Field(..., description="Golden dataset name")
    tenant_id: str = Field(default="default", description="Tenant ID")
    user_role: Role = Field(default=Role.ADMIN, description="User role")
    category_filter: Optional[str] = Field(None, description="Filter by category")
    difficulty_filter: Optional[str] = Field(None, description="Filter by difficulty")


# Response models

class ChunkResponse(BaseModel):
    """Chunk in response."""

    id: str
    content: str
    document_filename: str
    score: float
    rank: int
    retrieval_method: str


class CitationResponse(BaseModel):
    """Citation in response."""

    source_id: int
    document: str
    chunk_text: str
    relevance: str


class QueryResponse(BaseModel):
    """Query response."""

    query_id: str
    answer: str
    citations: List[CitationResponse]
    confidence: float
    retrieved_chunks: List[ChunkResponse]
    tokens_used: int
    latency_ms: float
    cost_usd: float
    created_at: datetime


class IngestResponse(BaseModel):
    """Ingestion response."""

    document_id: str
    filename: str
    status: str
    chunks_created: int
    embeddings_generated: int
    processing_time_seconds: float
    is_duplicate: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime
    components: Dict[str, str]


class StatsResponse(BaseModel):
    """Statistics response."""

    total_documents: int
    total_chunks: int
    vector_index_size: int
    bm25_index_size: int
    cache_stats: Dict[str, Any]
    tenant_counts: Dict[str, int]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)