from dataclasses import dataclass
import csv
from pathlib import Path
from pypdf import PdfReader
from docx import Document

@dataclass
class RawSection:
    text: str
    source: str
    page: int | None = None

@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    index: int


def load_sections(path: Path) -> list[RawSection]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return [RawSection((page.extract_text() or "").strip(), path.name, i + 1)
                for i, page in enumerate(reader.pages) if (page.extract_text() or "").strip()]
    if ext == ".docx":
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [RawSection(text, path.name)] if text.strip() else []
    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [RawSection(text, path.name)] if text.strip() else []
    if ext == ".csv":
        rows = []
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            for row in csv.reader(f):
                rows.append(" | ".join(row))
        text = "\n".join(rows)
        return [RawSection(text, path.name)] if text.strip() else []
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_sections(sections: list[RawSection], size: int = 700, overlap: int = 120) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    step = size - overlap
    idx = 0
    for section in sections:
        text = " ".join(section.text.split())
        for start in range(0, len(text), step):
            part = text[start:start + size].strip()
            if not part:
                continue
            chunks.append(Chunk(part, section.source, section.page, idx))
            idx += 1
            if start + size >= len(text):
                break
    return chunks
