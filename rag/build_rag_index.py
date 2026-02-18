"""
Build a FAISS index from trusted agriculture documents.

Input docs:
- Place .txt or .md files in rag/docs/

Outputs:
- rag_index.faiss
- rag_index_meta.json

Example:
python rag/build_rag_index.py
python rag/build_rag_index.py --docs_dir rag/docs --index_path rag_index.faiss --meta_path rag_index_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for AgriSathi RAG")
    parser.add_argument("--docs_dir", type=str, default="rag/docs")
    parser.add_argument("--index_path", type=str, default="rag_index.faiss")
    parser.add_argument("--meta_path", type=str, default="rag_index_meta.json")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--chunk_chars", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    return parser.parse_args()


def read_docs(docs_dir: Path) -> List[Path]:
    files = []
    for ext in ("*.txt", "*.md"):
        files.extend(docs_dir.rglob(ext))
    return sorted(files)


def split_long_text(text: str, chunk_chars: int, chunk_overlap: int) -> List[str]:
    """Split long text into overlapping fixed-size chunks."""
    chunks: List[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_document(content: str, chunk_chars: int, chunk_overlap: int) -> List[str]:
    """
    Chunk by paragraph first, then split oversized paragraphs.
    This keeps chunks small and semantically cleaner than fixed-only slicing.
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_chars:
            chunks.append(para)
        else:
            chunks.extend(split_long_text(para, chunk_chars, chunk_overlap))
    return chunks


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    doc_files = read_docs(docs_dir)
    if not doc_files:
        raise FileNotFoundError(f"No .txt or .md files found in: {docs_dir}")

    all_chunks = []
    for file_path in doc_files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_document(text, args.chunk_chars, args.chunk_overlap)
        for i, chunk in enumerate(chunks, start=1):
            all_chunks.append(
                {
                    "title": f"{file_path.stem} | chunk {i}",
                    "source_path": str(file_path).replace("\\", "/"),
                    "text": chunk,
                }
            )

    if not all_chunks:
        raise RuntimeError("No chunks generated from input documents.")

    print(f"Loaded {len(doc_files)} files")
    print(f"Generated {len(all_chunks)} chunks")
    print(f"Embedding model: {args.embedding_model}")

    embedder = SentenceTransformer(args.embedding_model)
    vectors = embedder.encode(
        [x["text"] for x in all_chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved index: {index_path}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
