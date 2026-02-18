"""
Build FAISS index for AgriSathi RAG.

Reads files from:
- data/docs/ (.pdf, .txt, .md)

Writes:
- rag/rag_index.faiss
- rag/rag_chunks.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("data/docs")
INDEX_PATH = Path("rag/rag_index.faiss")
CHUNKS_PATH = Path("rag/rag_chunks.jsonl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Requirement: chunk size between 400 and 600 tokens with overlap 50.
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


def read_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(str(path))
    pages_text: List[str] = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def read_text_file(path: Path) -> str:
    """Read plain text/markdown files."""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    return ""


def token_chunk(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    """
    Simple token chunking by whitespace tokens.
    Keeps overlap tokens between consecutive chunks.
    """
    tokens = text.split()
    if not tokens:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        part = " ".join(tokens[start:end]).strip()
        if part:
            chunks.append(part)
        if end == len(tokens):
            break
        start += step

    return chunks


def iter_docs(docs_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in ("*.pdf", "*.txt", "*.md"):
        files.extend(docs_dir.rglob(pattern))
    return sorted(files)


def build() -> None:
    """Builds FAISS index from data/docs/."""
    docs_dir = DOCS_DIR
    index_path = INDEX_PATH
    chunks_path = CHUNKS_PATH

    docs_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    files = iter_docs(docs_dir)
    if not files:
        raise FileNotFoundError(f"No .pdf/.txt/.md files found in {docs_dir}")

    all_chunks: List[Dict[str, str]] = []
    for file_path in files:
        raw_text = read_document(file_path)
        if not raw_text.strip():
            continue

        pieces = token_chunk(raw_text, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS)
        for i, piece in enumerate(pieces, start=1):
            all_chunks.append(
                {
                    "title": f"{file_path.stem} | chunk {i}",
                    "source_path": str(file_path).replace("\\", "/"),
                    "text": piece,
                }
            )

    if not all_chunks:
        raise RuntimeError("No text chunks created from documents.")

    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode(
        [c["text"] for c in all_chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_path))

    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Index built successfully with {len(all_chunks)} chunks")


if __name__ == "__main__":
    build()
