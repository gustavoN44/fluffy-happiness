"""Baseline run: execute both metric families and record one scored snapshot.

This is Phase 2's deliverable — the reproducible baseline every later change
(Phase 3 swaps, Phase 5 matrix) is measured against. It writes two things:
  - a timestamped combined run under eval/results/ (history), and
  - eval/baseline.json (the committed, canonical reference).

Caveat: the retrieval/IR metrics are deterministic, but the LLM-judged generation
metrics wobble a few points run-to-run even at temperature 0 (see DECISIONS.md D5).
This records a single pilot run; treat generation numbers as approximate.

Run in the eval venv (superset of the app venv + RAGAS):
    .venv-eval/bin/python -m eval.run_all
"""

import json
from datetime import datetime
from pathlib import Path

from eval import generation_metrics, retrieval_metrics

RESULTS_DIR = Path("eval/results")
BASELINE_PATH = Path("eval/baseline.json")

NOTE = (
    "Retrieval/IR metrics are deterministic. Generation metrics are LLM-judged "
    "(gpt-4o-mini) and vary a few points run-to-run even at temperature 0; "
    "answer_relevancy is the noisiest. Single pilot run over 14 items."
)


def run() -> dict:
    retrieval = retrieval_metrics.run()
    generation = generation_metrics.run()

    rc, gc = retrieval["config"], generation["config"]
    return {
        "run": {"timestamp": datetime.now().isoformat(timespec="seconds"), "type": "baseline"},
        "config": {
            "dataset": rc["dataset"],
            "corpus_chunks": rc["corpus_chunks"],
            "chunk_size": rc["chunk_size"],
            "overlap": rc["overlap"],
            "embedding_model": rc["embedding_model"],
            "generation_model": gc["generation_model"],
            "judge_model": gc["judge_model"],
            "retrieval_k": gc["k"],
            "k_values": rc["k_values"],
            "mrr_depth": rc["mrr_depth"],
            "num_answerable": gc["num_answerable"],
            "num_unanswerable": gc["num_unanswerable"],
        },
        "retrieval": {
            "aggregate": retrieval["aggregate"],
            "zero_relevant_queries": rc["zero_relevant_queries"],
            "per_query": retrieval["per_query"],
        },
        "generation": {
            "aggregate": generation["aggregate"],
            "per_query": generation["per_query"],
            "refusals": generation["refusals"],
        },
        "notes": NOTE,
    }


def _save(baseline: dict) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = baseline["run"]["timestamp"].replace(":", "").replace("-", "")
    run_path = RESULTS_DIR / f"baseline_{stamp}.json"
    payload = json.dumps(baseline, indent=2)
    run_path.write_text(payload)
    BASELINE_PATH.write_text(payload)  # canonical, committed reference
    return run_path, BASELINE_PATH


def _print_summary(baseline: dict) -> None:
    cfg = baseline["config"]
    r = baseline["retrieval"]["aggregate"]
    g = baseline["generation"]["aggregate"]

    print("\n" + "=" * 58)
    print("  PHASE 1 BASELINE")
    print("=" * 58)
    print(f"  corpus {cfg['corpus_chunks']} chunks | chunk {cfg['chunk_size']}/{cfg['overlap']} | "
          f"embed {cfg['embedding_model']}")
    print(f"  gen {cfg['generation_model']} | judge {cfg['judge_model']} | k={cfg['retrieval_k']} | "
          f"{cfg['num_answerable']} answerable + {cfg['num_unanswerable']} unanswerable")

    print(f"\n  RETRIEVAL (deterministic)      {'K':>3} | {'P@K':>6} | {'R@K':>6}")
    for k in cfg["k_values"]:
        print(f"  {'':30}{k:>3} | {r['precision_at_k'][str(k)]:>6.3f} | {r['recall_at_k'][str(k)]:>6.3f}")
    print(f"  {'':30}MRR (depth {cfg['mrr_depth']}): {r['mrr']:.3f}")

    print("\n  GENERATION (LLM-judged, approx)")
    for name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"    {name:<20}: {g[name]:.3f}")
    print(f"    {'refusal_accuracy':<20}: {g['refusal_accuracy']:.3f}")
    print("=" * 58)


if __name__ == "__main__":
    baseline = run()
    run_path, canonical = _save(baseline)
    _print_summary(baseline)
    print(f"\n  history : {run_path}")
    print(f"  canonical: {canonical}")
