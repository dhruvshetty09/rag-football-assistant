"""
loader.py — Load raw documents from a folder.

Supports .txt and .pdf files. Returns a list of dicts:
  [{"source": "filename.pdf", "text": "full text..."}, ...]

What to do here (Week 1, Session 2):
  - Drop your corpus files into data/raw/
  - For PDFs: pip install pymupdf  then uncomment the fitz block below.
  - For plain text, the current code works as-is.
"""

import os

DOC_DIR = "data/raw"


def load_documents(doc_dir: str = DOC_DIR) -> list[dict]:
    docs = []
    for fname in os.listdir(doc_dir):
        fpath = os.path.join(doc_dir, fname)

        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append({"source": fname, "text": f.read()})

        elif fname.endswith(".pdf"):
            import fitz
            with fitz.open(fpath) as pdf:
                text = "\n".join(page.get_text() for page in pdf)
                docs.append({"source": fname, "text": text})

    print(f"[loader] Loaded {len(docs)} documents from {doc_dir}")
    return docs
