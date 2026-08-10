from pathlib import Path
from app.core.config import get_settings
from .chunking import load_sections, chunk_sections
from .vector_store import get_vector_store

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv"}
IGNORE_NAMES = {"README.md"}


def eligible_file(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.name not in IGNORE_NAMES
        and path.suffix.lower() in SUPPORTED
    )


def ingest_path(path: Path, reset: bool = False) -> dict:
    s = get_settings()
    store = get_vector_store()
    if reset:
        store.reset()
    if path.is_file():
        files = [path] if eligible_file(path) else []
    else:
        files = sorted(p for p in path.rglob("*") if eligible_file(p))
    stats = {"files": 0, "chunks": 0, "errors": []}
    for f in files:
        try:
            sections = load_sections(f)
            chunks = chunk_sections(sections, s.chunk_size, s.chunk_overlap)
            stats["chunks"] += store.upsert_chunks(chunks)
            stats["files"] += 1
        except Exception as e:
            stats["errors"].append({"file": str(f), "error": str(e)})
    return stats
