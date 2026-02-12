"""Document management with versioning and deduplication."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from loguru import logger

from src.config import get_settings
from src.models import Document, IngestionManifest


class DocumentManager:
    """Manages document storage, versioning, and deduplication."""

    def __init__(self):
        """Initialize document manager."""
        self.settings = get_settings()
        self.raw_path = Path(self.settings.data_raw_path)
        self.staged_path = Path(self.settings.data_staged_path)
        self.manifests_path = Path(self.settings.data_manifests_path)
        
        # Ensure directories exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.staged_path.mkdir(parents=True, exist_ok=True)
        self.manifests_path.mkdir(parents=True, exist_ok=True)

    def store_document(self, document: Document, copy_file: bool = True) -> Path:
        """Store document in raw storage.
        
        Args:
            document: Document to store
            copy_file: Whether to copy file to raw storage
            
        Returns:
            Path to stored document
        """
        # Create tenant directory
        tenant_dir = self.raw_path / document.tenant_id
        tenant_dir.mkdir(exist_ok=True)
        
        # Store with content hash in filename to ensure uniqueness
        filename = f"{document.id}_{document.content_hash[:8]}_{document.filename}"
        dest_path = tenant_dir / filename
        
        if copy_file and document.source_path:
            source_path = Path(document.source_path)
            if source_path.exists() and source_path != dest_path:
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied document to: {dest_path}")
        
        # Update document path
        document.source_path = str(dest_path)
        
        return dest_path

    def check_duplicate(self, content_hash: str, tenant_id: str) -> Optional[Document]:
        """Check if document with same content hash already exists.
        
        Args:
            content_hash: SHA-256 hash of document content
            tenant_id: Tenant ID
            
        Returns:
            Existing document if found, None otherwise
        """
        # Search through manifests for matching hash
        tenant_manifests_dir = self.manifests_path / tenant_id
        if not tenant_manifests_dir.exists():
            return None
        
        for manifest_file in tenant_manifests_dir.glob("*.json"):
            try:
                with open(manifest_file, "r") as f:
                    manifest_data = json.load(f)
                    if manifest_data.get("content_hash") == content_hash:
                        logger.info(f"Found duplicate document: {manifest_file}")
                        # Load and return document
                        doc_data = manifest_data.get("document", {})
                        if doc_data:
                            return Document(**doc_data)
            except Exception as e:
                logger.error(f"Error reading manifest {manifest_file}: {e}")
                continue
        
        return None

    def get_document_version(self, content_hash: str, tenant_id: str) -> int:
        """Get next version number for document.
        
        Args:
            content_hash: Content hash
            tenant_id: Tenant ID
            
        Returns:
            Version number
        """
        # Check existing versions
        existing_doc = self.check_duplicate(content_hash, tenant_id)
        if existing_doc:
            return existing_doc.version + 1
        return 1

    def save_staged_content(self, document: Document) -> Path:
        """Save processed content to staged storage.
        
        Args:
            document: Document with content
            
        Returns:
            Path to staged file
        """
        # Create tenant directory
        tenant_dir = self.staged_path / document.tenant_id
        tenant_dir.mkdir(exist_ok=True)
        
        # Save as JSON with metadata
        filename = f"{document.id}_{document.content_hash[:8]}.json"
        staged_file = tenant_dir / filename
        
        # Prepare data
        data = {
            "document_id": str(document.id),
            "filename": document.filename,
            "content": document.metadata.get("content", ""),
            "content_hash": document.content_hash,
            "format": document.format,
            "created_at": document.created_at.isoformat(),
            "metadata": {k: v for k, v in document.metadata.items() if k != "content"}
        }
        
        with open(staged_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Staged content saved to: {staged_file}")
        document.staged_path = str(staged_file)
        
        return staged_file

    def create_manifest(self, document: Document, manifest: IngestionManifest) -> Path:
        """Create ingestion manifest.
        
        Args:
            document: Document
            manifest: Ingestion manifest
            
        Returns:
            Path to manifest file
        """
        # Create tenant directory
        tenant_dir = self.manifests_path / document.tenant_id
        tenant_dir.mkdir(exist_ok=True)
        
        # Save manifest
        manifest_file = tenant_dir / f"{document.id}_manifest.json"
        
        data = {
            "document": document.model_dump(mode="json"),
            "manifest": manifest.model_dump(mode="json")
        }
        
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Manifest created: {manifest_file}")
        
        return manifest_file

    def update_manifest(self, document_id: UUID, updates: Dict) -> None:
        """Update existing manifest.
        
        Args:
            document_id: Document ID
            updates: Dictionary of updates
        """
        # Search for manifest file
        for tenant_dir in self.manifests_path.iterdir():
            if not tenant_dir.is_dir():
                continue
            
            manifest_file = tenant_dir / f"{document_id}_manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        data = json.load(f)
                    
                    # Update manifest section
                    if "manifest" in data:
                        data["manifest"].update(updates)
                        data["manifest"]["updated_at"] = datetime.utcnow().isoformat()
                    
                    with open(manifest_file, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                    
                    logger.info(f"Manifest updated: {manifest_file}")
                    return
                except Exception as e:
                    logger.error(f"Error updating manifest {manifest_file}: {e}")
        
        logger.warning(f"Manifest not found for document: {document_id}")

    def get_manifest(self, document_id: UUID) -> Optional[IngestionManifest]:
        """Get manifest for document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Manifest if found
        """
        for tenant_dir in self.manifests_path.iterdir():
            if not tenant_dir.is_dir():
                continue
            
            manifest_file = tenant_dir / f"{document_id}_manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        data = json.load(f)
                    
                    manifest_data = data.get("manifest", {})
                    return IngestionManifest(**manifest_data)
                except Exception as e:
                    logger.error(f"Error reading manifest {manifest_file}: {e}")
        
        return None

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Document]:
        """List all documents, optionally filtered by tenant.
        
        Args:
            tenant_id: Optional tenant ID filter
            
        Returns:
            List of documents
        """
        documents = []
        
        search_dirs = [self.manifests_path / tenant_id] if tenant_id else list(self.manifests_path.iterdir())
        
        for tenant_dir in search_dirs:
            if not tenant_dir.is_dir():
                continue
            
            for manifest_file in tenant_dir.glob("*_manifest.json"):
                try:
                    with open(manifest_file, "r") as f:
                        data = json.load(f)
                    
                    doc_data = data.get("document", {})
                    if doc_data:
                        documents.append(Document(**doc_data))
                except Exception as e:
                    logger.error(f"Error reading manifest {manifest_file}: {e}")
        
        return documents

    def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document if found
        """
        for tenant_dir in self.manifests_path.iterdir():
            if not tenant_dir.is_dir():
                continue
            
            manifest_file = tenant_dir / f"{document_id}_manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        data = json.load(f)
                    
                    doc_data = data.get("document", {})
                    if doc_data:
                        return Document(**doc_data)
                except Exception as e:
                    logger.error(f"Error reading manifest {manifest_file}: {e}")
        
        return None

    def get_staged_content(self, document: Document) -> Optional[str]:
        """Get staged content for document.
        
        Args:
            document: Document
            
        Returns:
            Content if found
        """
        if document.staged_path:
            staged_file = Path(document.staged_path)
            if staged_file.exists():
                try:
                    with open(staged_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data.get("content", "")
                except Exception as e:
                    logger.error(f"Error reading staged content: {e}")
        
        return None