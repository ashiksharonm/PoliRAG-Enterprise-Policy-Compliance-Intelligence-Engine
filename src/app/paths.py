"""Request-supplied paths, confined to a configured directory."""
from __future__ import annotations

from pathlib import Path


def resolve_ingest_path(raw: str, root: str | Path) -> Path | None:
    """`raw` resolved under `root`, or None if it would leave that directory.

    Relative paths are taken from `root`. Resolution follows `..` and symlinks before the
    check, so neither can escape it.
    """
    base = Path(root).resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    return resolved if resolved.is_relative_to(base) else None
