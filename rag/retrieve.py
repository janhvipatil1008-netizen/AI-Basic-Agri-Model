"""
Simple FAISS retriever for AgriSathi.

This module is intentionally small and beginner-friendly so app code can call:
    from rag.retrieve import retrieve_top_k
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def retrieve_top_k(
    query: str,
    top_k: int = 3,
    index_path: str = "rag_index.faiss",
    meta_path: str = "rag_index_meta.json",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict]:
    """
    Retrieve top-k chunks for a query from FAISS index + metadata JSON.
    Returns list of dicts with keys: title, text, source_path, score
    """
    idx_path = Path(index_path)
    md_path = Path(meta_path)
    if not idx_path.exists() or not md_path.exists():
        return []

    index = faiss.read_index(str(idx_path))
    with md_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if not meta:
        return []

    embedder = SentenceTransformer(embedding_model)
    qvec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    k = min(max(1, int(top_k)), len(meta))
    scores, indices = index.search(qvec, k)

    out: List[Dict] = []
    for score, i in zip(scores[0], indices[0]):
        if i < 0:
            continue
        item = meta[i]
        out.append(
            {
                "title": item.get("title", f"chunk-{i}"),
                "text": item.get("text", ""),
                "source_path": item.get("source_path", ""),
                "score": float(score),
            }
        )
    return out
