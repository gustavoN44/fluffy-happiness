"""Retrieval (IR) evaluation: Precision@K, Recall@K, MRR against the gold set.

Scores the retriever independently of generation. Relevance is derived from the
span-based gold labels (DECISIONS.md D3): a chunk is relevant to a question if it
contains one of that question's gold spans (whitespace-normalized). Chunk identity
is (source, chunk_index) — stable across re-ingests, unlike the DB id.

Unanswerable questions have no relevant chunks and are excluded here; they are
scored on the generation side. Run: python -m eval.retrieval_metrics
"""

import json
import re
from datetime import datetime
from pathlib import Path

from app.chunker import CHUNK_SIZE, OVERLAP
from app.config import settings
from app.db import connect
from app.retriever import retrieve

DATASET_PATH = Path("eval/dataset.json")
RESULTS_DIR = Path("eval/results")
K_VALUES = [1, 3, 5]
MRR_DEPTH = 5  # rank depth searched for the first relevant chunk


def _norm(text: str) -> str:
    """Collapse all whitespace so PDF line-wrapping doesn't break span matching."""
    return re.sub(r"\s+", " ", text).strip()


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


def _load_corpus_chunks() -> list[tuple[str, int, str]]:
    """All current chunks as (source, chunk_index, normalized_content)."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT metadata->>'source', (metadata->>'chunk_index')::int, content "
            "FROM chunks"
        )
        return [(s, i, _norm(c)) for s, i, c in cur.fetchall()]


def _relevant_chunks(gold_spans: list[str], corpus: list[tuple[str, int, str]]) -> set[tuple[str, int]]:
    """Chunks whose content contains any gold span (both whitespace-normalized)."""
    spans = [_norm(s) for s in gold_spans]
    return {
        (source, idx)
        for source, idx, content in corpus
        if any(span in content for span in spans)
    }


def _evaluate_query(question: str, relevant: set[tuple[str, int]]) -> dict:
    retrieved = retrieve(question, k=MRR_DEPTH)
    hits = [((r.source, r.chunk_index) in relevant) for r in retrieved]

    precision = {k: sum(hits[:k]) / k for k in K_VALUES}
    recall = {k: (sum(hits[:k]) / len(relevant) if relevant else 0.0) for k in K_VALUES}
    reciprocal_rank = next((1.0 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)

    return {
        "retrieved": [
            {"source": r.source, "chunk_index": r.chunk_index, "rank": i + 1,
             "similarity": round(r.similarity, 4), "is_relevant": hits[i]}
            for i, r in enumerate(retrieved)
        ],
        "precision_at_k": precision,
        "recall_at_k": recall,
        "reciprocal_rank": reciprocal_rank,
    }


def run() -> dict:
    dataset = _load_dataset()
    corpus = _load_corpus_chunks()
    if not corpus:
        raise SystemExit("No chunks in the DB. Ingest a document first: python -m app.store data/<doc>")

    answerable = [d for d in dataset if d["answerable"]]
    per_query = []
    zero_relevant = []

    for item in answerable:
        relevant = _relevant_chunks(item["gold_spans"], corpus)
        if not relevant:
            # A gold span found in the doc but in no single chunk (e.g. split
            # across a boundary). Flag loudly — it would silently tank recall.
            zero_relevant.append(item["id"])
        result = _evaluate_query(item["question"], relevant)
        per_query.append({
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "num_relevant": len(relevant),
            "relevant_chunks": sorted(relevant),
            **result,
        })

    n = len(per_query)
    aggregate = {
        "num_queries": n,
        "precision_at_k": {str(k): round(sum(q["precision_at_k"][k] for q in per_query) / n, 4) for k in K_VALUES},
        "recall_at_k": {str(k): round(sum(q["recall_at_k"][k] for q in per_query) / n, 4) for k in K_VALUES},
        "mrr": round(sum(q["reciprocal_rank"] for q in per_query) / n, 4),
    }

    return {
        "run": {"timestamp": datetime.now().isoformat(timespec="seconds"), "type": "retrieval"},
        "config": {
            "dataset": str(DATASET_PATH),
            "embedding_model": settings.embedding_model,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "k_values": K_VALUES,
            "mrr_depth": MRR_DEPTH,
            "corpus_chunks": len(corpus),
            "excluded_unanswerable": [d["id"] for d in dataset if not d["answerable"]],
            "zero_relevant_queries": zero_relevant,
        },
        "aggregate": aggregate,
        "per_query": per_query,
    }


def _save(results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = results["run"]["timestamp"].replace(":", "").replace("-", "")
    path = RESULTS_DIR / f"retrieval_{stamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def _print_summary(results: dict) -> None:
    agg = results["aggregate"]
    cfg = results["config"]
    print(f"\nRetrieval metrics — {agg['num_queries']} answerable queries "
          f"(excluded unanswerable: {cfg['excluded_unanswerable']})")
    print(f"corpus: {cfg['corpus_chunks']} chunks | embedding: {cfg['embedding_model']} | "
          f"chunk {cfg['chunk_size']}/{cfg['overlap']}")
    if cfg["zero_relevant_queries"]:
        print(f"  WARNING zero-relevant-chunk queries: {cfg['zero_relevant_queries']}")
    print(f"\n  {'K':>3} | {'Precision@K':>12} | {'Recall@K':>10}")
    print(f"  {'-'*3}-+-{'-'*12}-+-{'-'*10}")
    for k in K_VALUES:
        print(f"  {k:>3} | {agg['precision_at_k'][str(k)]:>12.3f} | {agg['recall_at_k'][str(k)]:>10.3f}")
    print(f"\n  MRR (depth {cfg['mrr_depth']}): {agg['mrr']:.3f}\n")

    print("  per-query first-relevant rank:")
    for q in results["per_query"]:
        rr = q["reciprocal_rank"]
        rank = f"rank {round(1/rr)}" if rr else "not in top-%d" % MRR_DEPTH
        print(f"    {q['id']} [{q['category']:<9}] relevant={q['num_relevant']}  {rank}")


if __name__ == "__main__":
    results = run()
    path = _save(results)
    _print_summary(results)
    print(f"  saved: {path}")
