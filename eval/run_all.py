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

from app.pipeline import BASELINE, RunConfig
from eval import generation_metrics, retrieval_metrics

RESULTS_DIR = Path("eval/results")
BASELINE_PATH = Path("eval/baseline.json")

NOTE = (
    "Retrieval/IR metrics are deterministic. Generation metrics are LLM-judged "
    "(gpt-4o-mini) and vary a few points run-to-run even at temperature 0; "
    "answer_relevancy is the noisiest. Single pilot run over 14 items."
)


def run(config: RunConfig = BASELINE) -> dict:
    retrieval = retrieval_metrics.run(config)
    generation = generation_metrics.run(config)

    rc, gc = retrieval["config"], generation["config"]
    return {
        "run": {"timestamp": datetime.now().isoformat(timespec="seconds"), "type": "baseline"},
        "config": {
            "dataset": rc["dataset"],
            "config_id": config.config_id,
            "label": config.label,
            "corpus_chunks": rc["corpus_chunks"],
            "chunker": config.chunker,
            "chunker_params": config.chunker_params,
            "embedder": config.embedder,
            "embedder_params": config.embedder_params,
            "generation_model": gc["generation_model"],
            "judge_model": gc["judge_model"],
            "retrieval_k": config.retrieval_k,
            "retrieval_mode": config.retrieval_mode,
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
        "cost_latency": generation.get("cost_latency", {}),
        "notes": NOTE,
    }


def _save(results: dict, config: RunConfig) -> tuple[Path, Path | None]:
    """Always write a timestamped history file. Only the BASELINE config also
    updates the canonical eval/baseline.json — other configs (matrix cells) must
    not clobber the baseline reference."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = results["run"]["timestamp"].replace(":", "").replace("-", "")
    payload = json.dumps(results, indent=2)

    run_path = RESULTS_DIR / f"eval_{config.config_id}_{stamp}.json"
    run_path.write_text(payload)

    canonical = None
    if config.config_id == BASELINE.config_id:
        BASELINE_PATH.write_text(payload)
        canonical = BASELINE_PATH
    return run_path, canonical


def _print_summary(baseline: dict) -> None:
    cfg = baseline["config"]
    r = baseline["retrieval"]["aggregate"]
    g = baseline["generation"]["aggregate"]

    title = "BASELINE" if cfg["config_id"] == BASELINE.config_id else "CONFIG"
    print("\n" + "=" * 58)
    print(f"  {title} — {cfg['label']} ({cfg['config_id']})")
    print("=" * 58)
    print(f"  corpus {cfg['corpus_chunks']} chunks | chunker {cfg['chunker']} {cfg['chunker_params']}")
    print(f"  embedder {cfg['embedder']} {cfg['embedder_params']}")
    print(f"  gen {cfg['generation_model']} | judge {cfg['judge_model']} | k={cfg['retrieval_k']} | "
          f"mode={cfg['retrieval_mode']} | "
          f"{cfg['num_answerable']} answerable + {cfg['num_unanswerable']} unanswerable")

    print(f"\n  RETRIEVAL (deterministic)      {'K':>3} | {'P@K':>6} | {'R@K':>6}")
    for k in cfg["k_values"]:
        print(f"  {'':30}{k:>3} | {r['precision_at_k'][str(k)]:>6.3f} | {r['recall_at_k'][str(k)]:>6.3f}")
    print(f"  {'':30}MRR (depth {cfg['mrr_depth']}): {r['mrr']:.3f}")

    print("\n  GENERATION (LLM-judged, approx)")
    for name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"    {name:<20}: {g[name]:.3f}")
    print(f"    {'refusal_accuracy':<20}: {g['refusal_accuracy']:.3f}")

    cl = results.get("cost_latency") or {}
    if cl:
        print("\n  COST & LATENCY (per query, avg)")
        print(f"    {'cost':<20}: ${cl['avg_cost_usd']:.6f}   ({cl['avg_tokens']:.0f} tokens)")
        phases = cl.get("avg_seconds", {})
        parts = "  ".join(f"{p}={phases[p]:.3f}s" for p in sorted(phases))
        print(f"    {'latency':<20}: {cl['avg_total_seconds']:.3f}s total   {parts}")
    print("=" * 58)


if __name__ == "__main__":
    config = BASELINE
    results = run(config)
    run_path, canonical = _save(results, config)
    _print_summary(results)
    print(f"\n  history : {run_path}")
    print(f"  canonical: {canonical or '(not baseline — canonical unchanged)'}")
