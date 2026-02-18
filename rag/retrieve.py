"""
Simple FAISS retriever for AgriSathi.

Returns top-k relevant chunks with:
- title
- source_path
- score
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_PATH = "rag/rag_index.faiss"
DEFAULT_CHUNKS_PATH = "rag/rag_chunks.jsonl"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks_jsonl(path: Path) -> List[Dict]:
    chunks: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def retrieve_top_k(
    query: str,
    top_k: int = 3,
    index_path: str = DEFAULT_INDEX_PATH,
    meta_path: str = DEFAULT_CHUNKS_PATH,
    embedding_model: str = DEFAULT_EMBED_MODEL,
) -> List[Dict]:
    """
    Retrieve top-k chunks for a query.
    Returns list of dicts including title, source_path, text, score.
    """
    idx_path = Path(index_path)
    chunks_path = Path(meta_path)

    if not idx_path.exists() or not chunks_path.exists():
        return []

    index = faiss.read_index(str(idx_path))
    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        return []

    embedder = SentenceTransformer(embedding_model)
    qvec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    k = min(max(1, int(top_k)), len(chunks))
    scores, indices = index.search(qvec, k)

    results: List[Dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = chunks[idx]
        results.append(
            {
                "title": row.get("title", f"chunk-{idx}"),
                "source_path": row.get("source_path", ""),
                "text": row.get("text", ""),
                "score": float(score),
            }
        )
    return results
