"""Role-Based Access Control (RBAC) enforcement."""

from typing import List, Optional

from loguru import logger

from src.config import get_settings
from src.models import Role, Chunk, Document
from src.observability.metrics import rbac_denials_total


class RBACEnforcer:
    """Enforce role-based access control."""

    # Role hierarchy (higher roles inherit lower role permissions)
    ROLE_HIERARCHY = {
        Role.ADMIN: [Role.ADMIN, Role.LEGAL, Role.AUDIT, Role.COMPLIANCE, Role.READ_ONLY],
        Role.LEGAL: [Role.LEGAL, Role.READ_ONLY],
        Role.AUDIT: [Role.AUDIT, Role.READ_ONLY],
        Role.COMPLIANCE: [Role.COMPLIANCE, Role.READ_ONLY],
        Role.READ_ONLY: [Role.READ_ONLY],
    }

    def __init__(self):
        """Initialize RBAC enforcer."""
        self.settings = get_settings()
        logger.info("RBACEnforcer initialized")

    def can_access_document(self, user_role: Role, document: Document) -> bool:
        """Check if user can access document.

        Args:
            user_role: User's role
            document: Document to access

        Returns:
            True if access allowed
        """
        if not self.settings.enable_rbac:
            return True

        # Get allowed roles for this user
        allowed_roles = self.ROLE_HIERARCHY.get(user_role, [user_role])

        # Check if any of document's roles match
        document_roles = document.role_scope or []
        
        has_access = any(role in allowed_roles for role in document_roles)

        if not has_access:
            logger.warning(
                f"RBAC: User with role {user_role} denied access to document {document.id}"
            )
            rbac_denials_total.labels(
                tenant_id=document.tenant_id,
                role=user_role,
                resource="document"
            ).inc()

        return has_access

    def can_access_chunk(self, user_role: Role, chunk: Chunk) -> bool:
        """Check if user can access chunk.

        Args:
            user_role: User's role
            chunk: Chunk to access

        Returns:
            True if access allowed
        """
        if not self.settings.enable_rbac:
            return True

        # Get allowed roles for this user
        allowed_roles = self.ROLE_HIERARCHY.get(user_role, [user_role])

        # Check if any of chunk's roles match
        chunk_roles = chunk.metadata.get("role_scope", [])
        
        has_access = any(role in allowed_roles for role in chunk_roles)

        if not has_access:
            tenant_id = chunk.metadata.get("tenant_id", "unknown")
            logger.warning(
                f"RBAC: User with role {user_role} denied access to chunk {chunk.id}"
            )
            rbac_denials_total.labels(
                tenant_id=tenant_id,
                role=user_role,
                resource="chunk"
            ).inc()

        return has_access

    def filter_chunks(self, user_role: Role, chunks: List[Chunk]) -> List[Chunk]:
        """Filter chunks based on user role.

        Args:
            user_role: User's role
            chunks: List of chunks

        Returns:
            Filtered list of accessible chunks
        """
        if not self.settings.enable_rbac:
            return chunks

        filtered = [
            chunk for chunk in chunks
            if self.can_access_chunk(user_role, chunk)
        ]

        removed_count = len(chunks) - len(filtered)
        if removed_count > 0:
            logger.info(f"RBAC: Filtered out {removed_count} chunks for role {user_role}")

        return filtered

    def get_allowed_roles(self, user_role: Role) -> List[Role]:
        """Get list of roles accessible to user.

        Args:
            user_role: User's role

        Returns:
            List of accessible roles
        """
        return self.ROLE_HIERARCHY.get(user_role, [user_role])