"""Request-supplied paths, confined to a configured directory."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_ingest_path(raw: str, root: str | Path) -> Path | None:
    """`raw` resolved under `root`, or None if it would leave that directory.

    Relative paths are taken from `root`. realpath follows `..` and symlinks before the
    prefix check, so neither can escape it.
    """
    base = os.path.realpath(root)
    full = os.path.realpath(os.path.join(base, raw))
    if not full.startswith(base + os.sep):
        return None
    return Path(full)
