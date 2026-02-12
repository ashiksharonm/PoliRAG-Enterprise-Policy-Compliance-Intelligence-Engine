"""FastAPI application."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src import __version__
from src.app.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    EvaluateRequest,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
    ChunkResponse,
    CitationResponse,
)
from src.chunking.chunker import TablePreservingChunker
from src.config import get_settings
from src.embeddings.service import EmbeddingService
from src.eval.golden_dataset import GoldenDatasetManager
from src.eval.runner import EvaluationRunner
from src.generation.generator import ResponseGenerator
from src.guardrails.pii import PIIDetector
from src.guardrails.rbac import RBACEnforcer
from src.guardrails.rate_limit import RateLimiter
from src.ingestion.pipeline import IngestionPipeline
from src.models import Query, Role
from src.observability.logging import setup_logging
from src.observability.metrics import active_connections
from src.retrieval.hybrid import HybridRetriever, Reranker
from src.vectorstore.faiss_store import FAISSVectorStore
from src.vectorstore.bm25_store import BM25Store


# Initialize logging
setup_logging()

# Global components (initialized in lifespan)
components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting PoliRAG application...")
    
    settings = get_settings()
    
    # Initialize components
    try:
        components["settings"] = settings
        components["ingestion_pipeline"] = IngestionPipeline()
        components["chunker"] = TablePreservingChunker()
        components["embedding_service"] = EmbeddingService()
        components["vector_store"] = FAISSVectorStore()
        components["bm25_store"] = BM25Store()
        components["retriever"] = HybridRetriever(
            vector_store=components["vector_store"],
            bm25_store=components["bm25_store"],
            embedding_service=components["embedding_service"],
        )
        components["reranker"] = Reranker()
        components["generator"] = ResponseGenerator()
        components["pii_detector"] = PIIDetector()
        components["rbac_enforcer"] = RBACEnforcer()
        components["rate_limiter"] = RateLimiter()
        components["dataset_manager"] = GoldenDatasetManager()
        components["evaluation_runner"] = EvaluationRunner()
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down PoliRAG application...")
    
    # Save indexes
    try:
        if "vector_store" in components:
            components["vector_store"].save()
        if "bm25_store" in components:
            components["bm25_store"].save()
        logger.info("Indexes saved successfully")
    except Exception as e:
        logger.error(f"Error saving indexes: {e}")


# Create FastAPI app
app = FastAPI(
    title="PoliRAG API",
    description="Enterprise Policy & Compliance Intelligence Engine",
    version=__version__,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).model_dump(mode="json")
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump(mode="json")
    )


# Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "PoliRAG API",
        "version": __version__,
        "description": "Enterprise Policy & Compliance Intelligence Engine",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    component_status = {}
    
    for name, component in components.items():
        if component is not None:
            component_status[name] = "healthy"
        else:
            component_status[name] = "unavailable"
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in component_status.values()) else "degraded",
        version=__version__,
        timestamp=datetime.utcnow(),
        components=component_status
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """Ingest a document."""
    logger.info(f"Ingestion request for: {request.file_path}")
    
    # Check rate limit
    rate_limiter = components["rate_limiter"]
    allowed, retry_after = rate_limiter.check_limit(request.tenant_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds"
        )
    
    # Ingest document
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}"
        )
    
    try:
        pipeline = components["ingestion_pipeline"]
        document, manifest = pipeline.ingest_file(
            file_path=file_path,
            tenant_id=request.tenant_id,
            role_scope=request.role_scope,
            skip_if_duplicate=request.skip_if_duplicate,
            **request.metadata
        )
        
        # Chunk document
        chunker = components["chunker"]
        chunks = chunker.chunk_document(document)
        
        # Generate embeddings
        embedding_service = components["embedding_service"]
        chunks = await embedding_service.embed_chunks(chunks)
        
        # Add to indexes
        vector_store = components["vector_store"]
        bm25_store = components["bm25_store"]
        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)
        
        # Update manifest
        manifest.chunks_created = len(chunks)
        manifest.embeddings_generated = len(chunks)
        
        return IngestResponse(
            document_id=str(document.id),
            filename=document.filename,
            status=manifest.status,
            chunks_created=manifest.chunks_created,
            embeddings_generated=manifest.embeddings_generated,
            processing_time_seconds=manifest.processing_time_seconds or 0.0,
            is_duplicate=manifest.status == "duplicate"
        )
    
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the system."""
    logger.info(f"Query request: {request.text[:100]}")
    
    # Check rate limit
    rate_limiter = components["rate_limiter"]
    allowed, retry_after = rate_limiter.check_limit(request.tenant_id, cost=2.0)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds"
        )
    
    # PII detection in query
    pii_detector = components["pii_detector"]
    if pii_detector.contains_pii(request.text):
        logger.warning(f"PII detected in query from tenant {request.tenant_id}")
    
    try:
        # Create query
        query_obj = Query(
            text=request.text,
            tenant_id=request.tenant_id,
            user_role=request.user_role,
            filters=request.filters
        )
        
        # Retrieve
        retriever = components["retriever"]
        retrieved_chunks = await retriever.retrieve(
            query=query_obj,
            k=request.top_k,
            method=request.retrieval_method
        )
        
        # Apply RBAC filtering
        rbac_enforcer = components["rbac_enforcer"]
        filtered_chunks = [
            chunk for chunk in retrieved_chunks
            if rbac_enforcer.can_access_chunk(request.user_role, chunk.chunk)
        ]
        
        if len(filtered_chunks) < len(retrieved_chunks):
            logger.info(f"RBAC filtered {len(retrieved_chunks) - len(filtered_chunks)} chunks")
        
        # Rerank
        reranker = components["reranker"]
        reranked_chunks = reranker.rerank(request.text, filtered_chunks)
        
        # Generate response
        generator = components["generator"]
        response = await generator.generate(query_obj, reranked_chunks)
        
        # Convert to response model
        return QueryResponse(
            query_id=str(response.query_id),
            answer=response.answer,
            citations=[
                CitationResponse(**citation) for citation in response.citations
            ],
            confidence=response.confidence,
            retrieved_chunks=[
                ChunkResponse(
                    id=str(chunk.chunk.id),
                    content=chunk.chunk.content[:500],  # Truncate for response
                    document_filename=chunk.chunk.metadata.get("document_filename", "Unknown"),
                    score=chunk.score,
                    rank=chunk.rank,
                    retrieval_method=chunk.retrieval_method
                )
                for chunk in reranked_chunks
            ],
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
            cost_usd=response.metadata.get("cost_usd", 0.0),
            created_at=response.created_at
        )
    
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        vector_store = components["vector_store"]
        bm25_store = components["bm25_store"]
        embedding_service = components["embedding_service"]
        ingestion_pipeline = components["ingestion_pipeline"]
        
        # Get documents by tenant
        documents = ingestion_pipeline.list_ingested_documents()
        tenant_counts = {}
        for doc in documents:
            tenant_counts[doc.tenant_id] = tenant_counts.get(doc.tenant_id, 0) + 1
        
        return StatsResponse(
            total_documents=len(documents),
            total_chunks=vector_store.index.ntotal,
            vector_index_size=vector_store.index.ntotal,
            bm25_index_size=len(bm25_store.chunks),
            cache_stats=embedding_service.cache.get_stats(),
            tenant_counts=tenant_counts
        )
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower()
    )