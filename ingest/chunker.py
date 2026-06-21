"""
chunker.py — Split documents into overlapping chunks.

Why chunking matters: embedding a 20-page paper as one vector loses detail.
Smaller chunks = more precise retrieval, but too small = missing context.
Start with CHUNK_SIZE=500 and OVERLAP=50, then tune in Week 2 based on eval.

Returns a list of dicts:
  [{"chunk_id": "paper.pdf_0", "source": "paper.pdf", "text": "..."}, ...]
"""

CHUNK_SIZE = 500   # characters — tune this after seeing your Week 2 eval numbers
OVERLAP = 50       # characters of overlap between consecutive chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    chunks = []
    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        start = 0
        idx = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{source}_{idx}",
                    "source": source,
                    "text": chunk_text,
                })
                idx += 1
            start += CHUNK_SIZE - OVERLAP

    print(f"[chunker] Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks
