# Agentic RAG Research Assistant

A domain-specific question-answering system built with:
- Retrieval-Augmented Generation (RAG) over a real document corpus
- An agentic decision loop with query rewriting and confidence-based fallback
- Structured JSON outputs with graceful error handling
- A quantitative evaluation harness (hit rate, MRR, faithfulness)
- A deployed Streamlit interface with source transparency

## Architecture

```
data/raw/ → chunk + embed → FAISS vector store
                                    ↑  retrieve
User question → Agent loop → LLM → validated JSON answer + citations
                     ↓
              Eval harness: hit_rate · MRR · faithfulness
```

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the local LLM: download Ollama from https://ollama.com, then:
```
ollama pull llama3
```

## Usage

```powershell
# 1. Drop your documents (.txt or .pdf) into data/raw/

# 2. Build the vector index
python run_ingest.py

# 3. Run the app locally
streamlit run app/streamlit_app.py

# 4. Measure quality
python run_eval.py
```

## Design decisions & tradeoffs

*(Fill this in as you build — your reasoning is more valuable than the code)*

- **Chunk size:** Started at 500 chars with 50-char overlap. Eval showed hit_rate of [X]; tuned to [Y] chars which improved to [Z].
- **Vector store:** FAISS chosen for zero infrastructure overhead. Would switch to pgvector for production persistence and filtering.
- **Agent routing:** Hand-rolled loop rather than a framework — every decision is explicit and explainable.
- **LLM-as-judge for faithfulness:** Fast to run but noisy. Would supplement with human spot-checks on a random 10% sample for higher confidence.
- **What I'd do with more time:** Add chunk-level metadata filtering, hybrid BM25+dense retrieval, and a proper regression test suite.

## Eval results

| Version | Chunk size | Top-k | Hit rate | MRR | Notes |
|---------|-----------|-------|----------|-----|-------|
| Baseline | 500 | 5 | 1.0 | 0.923 | Strong on direct questions; comparative questions (e.g. "difference between X and Y") show higher distance scores (~1.0-1.2), indicating split-chunk weakness |

*(Run `python run_eval.py` after each change and update this table)*
