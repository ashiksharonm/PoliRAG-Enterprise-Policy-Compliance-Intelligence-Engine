from pathlib import Path

from src.app.paths import resolve_ingest_path


def test_relative_path_inside_root(tmp_path: Path):
    (tmp_path / "policy.txt").write_text("x")
    assert resolve_ingest_path("policy.txt", tmp_path) == (tmp_path / "policy.txt").resolve()


def test_absolute_path_inside_root(tmp_path: Path):
    target = tmp_path / "sub" / "a.txt"
    target.parent.mkdir()
    target.write_text("x")
    assert resolve_ingest_path(str(target), tmp_path) == target.resolve()


def test_traversal_and_outside_paths_rejected(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    assert resolve_ingest_path("../secret.txt", root) is None
    assert resolve_ingest_path("/etc/passwd", root) is None
    assert resolve_ingest_path(str(tmp_path), root) is None


def test_symlink_escape_rejected(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("x")
    (root / "link.txt").symlink_to(tmp_path / "outside.txt")
    assert resolve_ingest_path("link.txt", root) is None
