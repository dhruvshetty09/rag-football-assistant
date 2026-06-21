"""
embedder.py — Convert chunks to embeddings and save to the vector store.

Pick ONE option and delete the other before running.

OPTION A — local, completely free:
  pip install sentence-transformers faiss-cpu
  Downloads ~90MB model on first run. No API key needed.

OPTION B — hosted, higher quality, costs ~$0.01 per full ingest:
  pip install openai faiss-cpu
  Set OPENAI_API_KEY in your .env file.

The vector store saves to data/chunks/ so you only re-run this
when your corpus changes.
"""

import json
import os
import numpy as np
import faiss

INDEX_PATH = "data/chunks/faiss_index"
CHUNKS_PATH = "data/chunks/chunks.json"


# --- OPTION A: local embeddings (sentence-transformers) ---
def embed_local(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True).astype("float32")


# --- OPTION B: OpenAI embeddings ---
def embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    vecs = [item.embedding for item in response.data]
    return np.array(vecs, dtype="float32")


def build_index(chunks: list[dict], use_local: bool = True):
    texts = [c["text"] for c in chunks]

    print("[embedder] Embedding chunks...")
    vectors = embed_local(texts) if use_local else embed_openai(texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs("data/chunks", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)

    print(f"[embedder] Saved index ({index.ntotal} vectors, dim={dim})")
