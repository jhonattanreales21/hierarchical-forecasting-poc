"""Local document RAG utilities for the Forecast Assistant."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    chunk_id: int
    page_number: int | None = None


def extract_document_text(path: Path) -> list[tuple[str, int | None]]:
    """Extract text as ``(text, page_number)`` records from PDF/DOCX/MD/TXT."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            (page.extract_text() or "", page_idx + 1)
            for page_idx, page in enumerate(reader.pages)
        ]
    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        return [("\n".join(p.text for p in doc.paragraphs), None)]
    if suffix in {".md", ".markdown", ".txt"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    raise ValueError(f"Unsupported RAG document type: {suffix}")


def chunk_text_records(
    text_records: list[tuple[str, int | None]],
    source: str,
    chunk_words: int = 900,
    overlap_words: int = 120,
) -> list[DocumentChunk]:
    """Split extracted text records into overlapping word chunks."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_words.")

    chunks: list[DocumentChunk] = []
    for text, page_number in text_records:
        words = text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunks.append(
                DocumentChunk(
                    text=" ".join(words[start:end]),
                    source=source,
                    page_number=page_number,
                    chunk_id=len(chunks),
                )
            )
            if end == len(words):
                break
            start = end - overlap_words
    return chunks


def build_chunks_from_path(path: Path) -> list[DocumentChunk]:
    """Extract and chunk a supported local document."""
    return chunk_text_records(extract_document_text(path), source=path.name)


def save_uploaded_file(file: BinaryIO, target_dir: Path, filename: str) -> Path:
    """Persist an uploaded document for local indexing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    target = target_dir / safe_name
    data = file.read()
    target.write_bytes(data)
    return target


class FaissVectorStore:
    """Small local FAISS vector store using normalized MiniLM embeddings."""

    def __init__(self, directory: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.directory = directory
        self.model_name = model_name
        self.index_path = directory / "index.faiss"
        self.metadata_path = directory / "chunks.json"
        self.manifest_path = directory / "manifest.json"

    def exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def build(self, chunks: list[DocumentChunk], source_path: Path | None = None) -> int:
        """Build and persist a FAISS inner-product index."""
        if not chunks:
            raise ValueError("Cannot build a RAG index with zero chunks.")
        import faiss

        embeddings = self._embed([chunk.text for chunk in chunks])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps([asdict(chunk) for chunk in chunks], indent=2),
            encoding="utf-8",
        )
        if source_path:
            self.manifest_path.write_text(
                json.dumps(
                    {
                        "source_path": str(source_path),
                        "source_name": source_path.name,
                        "source_modified_at": source_path.stat().st_mtime,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return len(chunks)

    def manifest(self) -> dict:
        """Return metadata for the indexed document source."""
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def is_stale_for(self, source_path: Path | None) -> bool:
        """Return whether a source is newer or different than the current index."""
        if source_path is None:
            return False
        if not self.exists() or not self.manifest_path.exists():
            return True
        manifest = self.manifest()
        return (
            manifest.get("source_path") != str(source_path)
            or float(manifest.get("source_modified_at", 0)) < source_path.stat().st_mtime
        )

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k chunks for a question."""
        if not self.exists():
            return []
        import faiss

        index = faiss.read_index(str(self.index_path))
        query = self._embed([question])
        scores, indices = index.search(query, top_k)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(metadata):
                continue
            item = dict(metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def _embed(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")
