"""
metrics.py — Measure retrieval quality and answer faithfulness.

This is your differentiator. Two layers:

1. RETRIEVAL METRICS (does the right chunk get fetched?)
   - hit_rate:  fraction of questions where the correct source appears in top-k
   - mrr:       mean reciprocal rank — rewards finding the right chunk higher up
                e.g. correct chunk at rank 1 → 1.0, at rank 2 → 0.5, at rank 5 → 0.2

2. ANSWER METRICS (is the generated answer grounded in the context?)
   - faithfulness: a second LLM call judges whether the answer is supported by
                   the retrieved chunks, or whether it contains invented content.

Workflow:
  python run_eval.py            → get your baseline numbers
  [make a change to chunk size / top-k / agent logic]
  python run_eval.py            → see if it improved
  Record both runs in your README's eval table.
"""

import json
from retrieve.searcher import search
from agent.agent import run
from agent.llm import call_llm

EVAL_PATH = "eval/eval_set.json"


def load_eval_set() -> list[dict]:
    with open(EVAL_PATH) as f:
        return json.load(f)


def compute_retrieval_metrics(eval_set: list[dict], k: int = 5) -> dict:
    """
    For each eval question, check whether the expected source document
    appears in the top-k retrieved chunks.
    """
    hits = 0
    reciprocal_ranks = []

    for item in eval_set:
        chunks = search(item["question"], k=k)
        sources = [c["source"] for c in chunks]

        # Hit rate: correct source anywhere in the top-k
        hit = item["expected_source"] in sources
        hits += int(hit)

        # MRR: reciprocal of the rank of the first correct source
        rank = next(
            (i + 1 for i, s in enumerate(sources) if s == item["expected_source"]),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    return {
        "hit_rate": round(hits / len(eval_set), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
        "n": len(eval_set),
    }


def score_faithfulness(answer: str, context_chunks: list[dict]) -> float:
    """
    Use a second LLM call as a judge.
    Returns 0.0 (hallucinated) to 1.0 (fully grounded in context).

    This is an LLM-as-judge pattern — fast but noisy.
    For higher confidence, add human spot-checks on a random sample.
    """
    context = "\n\n".join(c["text"] for c in context_chunks)
    prompt = (
        "Given the context and answer below, rate how faithful the answer is "
        "to the context on a scale of 0.0 to 1.0.\n"
        "0.0 = the answer contains information not present in the context.\n"
        "1.0 = every claim in the answer is directly supported by the context.\n"
        "Return ONLY the number, nothing else.\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\nScore:"
    )
    raw = call_llm(prompt).strip()
    try:
        return float(raw)
    except ValueError:
        print(f"[metrics] Could not parse faithfulness score from: {raw!r}")
        return 0.0


def run_full_eval(eval_set: list[dict] | None = None) -> dict:
    if eval_set is None:
        eval_set = load_eval_set()

    print(f"[eval] Running on {len(eval_set)} questions...")

    print("[eval] Computing retrieval metrics...")
    retrieval = compute_retrieval_metrics(eval_set)
    print(f"       hit_rate={retrieval['hit_rate']}  mrr={retrieval['mrr']}")

    print("[eval] Scoring answer faithfulness...")
    scores = []
    for i, item in enumerate(eval_set):
        result = run(item["question"])
        score = score_faithfulness(result["answer"], result["retrieved_chunks"])
        scores.append(score)
        print(f"       [{i+1}/{len(eval_set)}] faithfulness={score:.2f}")

    avg_faith = round(sum(scores) / len(scores), 3)
    print(f"       avg_faithfulness={avg_faith}")

    return {
        **retrieval,
        "avg_faithfulness": avg_faith,
    }
