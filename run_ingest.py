"""
run_ingest.py — Index your corpus into the vector store.

Run this once after dropping documents into data/raw/, then again
any time you add new documents or change chunking parameters.

Usage (from project root, with venv active):
  python run_ingest.py
"""

from ingest.loader import load_documents
from ingest.chunker import chunk_documents
from ingest.embedder import build_index

if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("No documents found in data/raw/ — add .txt or .pdf files first.")
    else:
        chunks = chunk_documents(docs)
        build_index(chunks, use_local=True)  # set use_local=False for OpenAI
        print("\nIngest complete. Run: streamlit run app/streamlit_app.py")
