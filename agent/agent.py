"""
agent.py — Makes the system "agentic."


The five steps on every question:
  1. Retrieve top-k chunks for the question.
  2. Check if the context looks relevant (distance threshold heuristic).
  3. If context is weak, rewrite the query and retrieve again.
  4. Build a structured prompt with chunk IDs for citation tracking.
  5. Parse and validate the LLM response into a clean dict.

WHY hand-rolled instead of LangChain/LangGraph?
  Because you need to explain every line in an interview.
  A framework hides the decisions; this file makes them explicit.
  Once you can explain this fully, you can explain any framework too.S
"""

from retrieve.searcher import search
from agent.llm import call_llm
from agent.output_parser import parse_response

USE_LOCAL = True   # set False to use OpenAI for both embeddings and LLM

SYSTEM_PROMPT = """You are a precise research assistant. Answer questions using
ONLY the provided context passages. Return your response as valid JSON with
exactly these keys:
- "answer": your answer as a string
- "citations": a list of chunk_id strings from the passages you used
- "confidence": "high", "medium", or "low"

If the context does not contain enough information to answer the question,
set answer to "I don't have enough information in my knowledge base to answer this."
and confidence to "low". Never invent facts not present in the context."""


def run(question: str) -> dict:
    """
    Main entry point. Takes a question string, returns a validated dict:
      {
        "answer": str,
        "citations": [chunk_id, ...],
        "confidence": "high" | "medium" | "low",
        "retrieved_chunks": [chunk_dict, ...]
      }
    """
    print(f"\n[agent] Question: {question}")

    # Step 1 — first retrieval attempt
    chunks = search(question, use_local=USE_LOCAL)
    print(f"[agent] Retrieved {len(chunks)} chunks (best score: {chunks[0]['score']:.3f if chunks else 'n/a'})")

    # Step 2 — check context quality; retry with a rewritten query if weak
    if _context_too_weak(chunks):
        print("[agent] Context looks weak — rewriting query and retrying...")
        rewritten = _rewrite_query(question)
        print(f"[agent] Rewritten: {rewritten}")
        chunks = search(rewritten, use_local=USE_LOCAL)
        print(f"[agent] Re-retrieved {len(chunks)} chunks")

    # Step 3 — format context with chunk IDs so the model can cite them
    context = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in chunks
    )

    # Step 4 — call the LLM
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    raw = call_llm(prompt, system=SYSTEM_PROMPT, use_local=USE_LOCAL)

    # Step 5 — parse, validate, attach raw chunks for the UI and eval
    result = parse_response(raw)
    result["retrieved_chunks"] = chunks
    return result


def _context_too_weak(chunks: list[dict], threshold: float = 1.5) -> bool:
    """
    Heuristic: if the best match has a high L2 distance, it's probably
    not relevant to the question. Lower distance = more similar in FAISS.

    Tune this threshold in Week 2 — look at the scores of correct vs.
    incorrect retrievals in your eval set to find a good cutoff.
    """
    if not chunks:
        return True
    return chunks[0]["score"] > threshold


def _rewrite_query(question: str) -> str:
    """
    Ask the LLM to rephrase the question for better retrieval.
    Different wording can match different chunk vocabulary.
    """
    prompt = (
        "Rewrite the following question to improve keyword overlap with a technical "
        "document database. Return ONLY the rewritten question, nothing else.\n\n"
        f"Original: {question}"
    )
    return call_llm(prompt, use_local=USE_LOCAL).strip()
