"""
searcher.py — Given a question, return the top-k most relevant chunks.

This is the retrieval half of RAG. It:
  1. Embeds the user's query with the same model used during ingest.
  2. Searches the FAISS index for the nearest chunk vectors.
  3. Returns the matching chunk dicts, each with a distance score.

Lower score = more similar (FAISS uses L2 distance).
Tune TOP_K in Week 2 based on eval results. 5 is a good starting point.
"""

import json
import numpy as np
import faiss

INDEX_PATH = "data/chunks/faiss_index"
CHUNKS_PATH = "data/chunks/chunks.json"
TOP_K = 5

_index = None
_chunks = None


def _load():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH) as f:
            _chunks = json.load(f)


def _embed_local(query: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([query]).astype("float32")


def _embed_openai(query: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(input=[query], model="text-embedding-3-small")
    return np.array([resp.data[0].embedding], dtype="float32")


def search(query: str, k: int = TOP_K, use_local: bool = True) -> list[dict]:
    """
    Returns up to k chunks most relevant to the query.
    Each chunk dict has: chunk_id, source, text, score.
    """
    _load()
    vec = _embed_local(query) if use_local else _embed_openai(query)
    distances, indices = _index.search(vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(_chunks[idx])
        chunk["score"] = float(dist)
        results.append(chunk)
    return results
