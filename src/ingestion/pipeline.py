"""Document ingestion pipeline."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from loguru import logger

from src.config import get_settings
from src.ingestion.loaders import LoaderFactory
from src.ingestion.manager import DocumentManager
from src.models import Document, IngestionManifest, Role
from src.observability.metrics import (
    ingestion_documents_total,
    ingestion_duration,
)


class IngestionPipeline:
    """Orchestrates document ingestion process."""

    def __init__(self):
        """Initialize ingestion pipeline."""
        self.settings = get_settings()
        self.document_manager = DocumentManager()

    def ingest_file(
        self,
        file_path: Path,
        tenant_id: str,
        role_scope: Optional[List[Role]] = None,
        skip_if_duplicate: bool = True,
        **metadata
    ) -> tuple[Document, IngestionManifest]:
        """Ingest a single file.

        Args:
            file_path: Path to file
            tenant_id: Tenant ID
            role_scope: List of roles that can access this document
            skip_if_duplicate: Skip if duplicate found
            **metadata: Additional metadata

        Returns:
            Tuple of (Document, IngestionManifest)
        """
        start_time = datetime.utcnow()
        
        # Create manifest
        manifest = IngestionManifest(
            document_id=UUID(int=0),  # Temporary, will be updated
            filename=file_path.name,
            content_hash="",
            status="pending",
            started_at=start_time
        )

        try:
            logger.info(f"Starting ingestion: {file_path}")
            manifest.status = "processing"

            # Load document
            loader = LoaderFactory.get_loader(file_path)
            document = loader.load(
                file_path,
                tenant_id=tenant_id,
                role_scope=role_scope or [Role.READ_ONLY],
                **metadata
            )

            # Update manifest with document ID and hash
            manifest.document_id = document.id
            manifest.content_hash = document.content_hash

            # Check for duplicates
            if skip_if_duplicate:
                existing_doc = self.document_manager.check_duplicate(
                    document.content_hash, tenant_id
                )
                if existing_doc:
                    logger.info(f"Duplicate document found: {existing_doc.id}")
                    manifest.status = "completed"
                    manifest.completed_at = datetime.utcnow()
                    manifest.processing_time_seconds = (
                        manifest.completed_at - manifest.started_at
                    ).total_seconds()
                    
                    # Metric
                    ingestion_documents_total.labels(
                        format=document.format,
                        tenant_id=tenant_id,
                        status="duplicate"
                    ).inc()
                    
                    return existing_doc, manifest

            # Store document in raw storage
            self.document_manager.store_document(document, copy_file=True)

            # Save staged content
            self.document_manager.save_staged_content(document)

            # Create manifest
            manifest.status = "completed"
            manifest.completed_at = datetime.utcnow()
            manifest.processing_time_seconds = (
                manifest.completed_at - manifest.started_at
            ).total_seconds()
            
            self.document_manager.create_manifest(document, manifest)

            logger.info(
                f"Ingestion completed: {file_path} -> {document.id} "
                f"({manifest.processing_time_seconds:.2f}s)"
            )

            # Metrics
            ingestion_documents_total.labels(
                format=document.format,
                tenant_id=tenant_id,
                status="completed"
            ).inc()
            
            ingestion_duration.labels(format=document.format).observe(
                manifest.processing_time_seconds
            )

            return document, manifest

        except Exception as e:
            logger.error(f"Ingestion failed for {file_path}: {e}")
            manifest.status = "failed"
            manifest.error_message = str(e)
            manifest.completed_at = datetime.utcnow()
            
            # Metric
            ingestion_documents_total.labels(
                format="unknown",
                tenant_id=tenant_id,
                status="failed"
            ).inc()
            
            raise

    def ingest_directory(
        self,
        directory: Path,
        tenant_id: str,
        role_scope: Optional[List[Role]] = None,
        recursive: bool = True,
        skip_if_duplicate: bool = True,
        **metadata
    ) -> List[tuple[Document, IngestionManifest]]:
        """Ingest all supported files in a directory.

        Args:
            directory: Directory path
            tenant_id: Tenant ID
            role_scope: List of roles
            recursive: Recursively scan subdirectories
            skip_if_duplicate: Skip duplicates
            **metadata: Additional metadata

        Returns:
            List of (Document, IngestionManifest) tuples
        """
        logger.info(f"Ingesting directory: {directory}")
        
        results = []
        supported_formats = LoaderFactory.supported_formats()

        # Get all files
        if recursive:
            files = [f for f in directory.rglob("*") if f.is_file()]
        else:
            files = [f for f in directory.iterdir() if f.is_file()]

        logger.info(f"Found {len(files)} files in {directory}")

        # Process each file
        for file_path in files:
            if file_path.suffix.lower() in supported_formats:
                try:
                    doc, manifest = self.ingest_file(
                        file_path=file_path,
                        tenant_id=tenant_id,
                        role_scope=role_scope,
                        skip_if_duplicate=skip_if_duplicate,
                        **metadata
                    )
                    results.append((doc, manifest))
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")
                    continue
            else:
                logger.debug(f"Skipping unsupported file: {file_path}")

        logger.info(f"Directory ingestion completed: {len(results)} documents")
        
        return results

    def get_ingestion_status(self, document_id: UUID) -> Optional[IngestionManifest]:
        """Get ingestion status for a document.

        Args:
            document_id: Document ID

        Returns:
            Manifest if found
        """
        return self.document_manager.get_manifest(document_id)

    def list_ingested_documents(self, tenant_id: Optional[str] = None) -> List[Document]:
        """List all ingested documents.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of documents
        """
        return self.document_manager.list_documents(tenant_id=tenant_id)
