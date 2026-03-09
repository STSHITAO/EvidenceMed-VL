from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

from medical_rag.schemas import DocChunk


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _read_text_file(path)
    if suffix == ".pdf":
        return _read_pdf_file(path)
    raise ValueError(f"Unsupported document type: {suffix}")


def sliding_window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def build_chunks(doc_dir: str, chunk_size: int, chunk_overlap: int) -> List[DocChunk]:
    dir_path = Path(doc_dir)
    paths: Iterable[Path] = sorted(
        p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"}
    )

    all_chunks: List[DocChunk] = []
    for path in paths:
        text = load_document_text(path)
        split_chunks = sliding_window_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(split_chunks):
            all_chunks.append(
                DocChunk(
                    doc_id=f"{path.stem}-{idx}",
                    source=str(path),
                    chunk_index=idx,
                    content=chunk,
                )
            )
    return all_chunks
