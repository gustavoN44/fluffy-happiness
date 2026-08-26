"""Phase 5 experiment matrix: chunking x embedding, scored on quality, cost, latency.

Runs the roadmap's 3x2 matrix. Everything except the two variables under test is held
constant: eval dataset, top-K, generation model, prompts, retrieval mode (dense).

Per cell:
  1. Ingest the corpus under that config (metered: tokens, cost, wall time).
  2. Discard a warm-up query — the first API call of a process carries client-init and
     TLS setup (~1.8s observed), which would otherwise unfairly penalise whichever
     cell runs first and corrupt cross-cell latency comparison.
  3. Retrieval metrics ONCE (deterministic — no averaging needed).
  4. Generation metrics N times (default 3; LLM-judge scores wobble run to run),
     reported as mean with min/max spread so "real difference or noise?" is answerable.

Run in the eval venv (RAGAS lives there):
    .venv-eval/bin/python -m eval.matrix                # all 6 cells
    .venv-eval/bin/python -m eval.matrix --only r512-small --gen-runs 1   # smoke test
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from app.metering import meter
from app.pipeline import RunConfig
from app.store import ingest_document
from eval import generation_metrics, retrieval_metrics

CORPUS = "data/mota-origenes.pdf"
RESULTS_DIR = Path("eval/results")
CELL_DIR = RESULTS_DIR / "cells"   # per-cell files, written as each cell completes
GEN_METRICS = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "refusal_accuracy"]

_SMALL = {"model": "text-embedding-3-small", "dim": 1536}
_VOYAGE = {"model": "voyage-4-large", "dim": 1024}

# The 3x2 matrix. Semantic takes no chunk_size (its whole point is variable sizes),
# so the two recursive sizes are what separate the SIZE effect from the STRATEGY effect.
CELLS: dict[str, RunConfig] = {
    "r512-small":  RunConfig(chunker="recursive", chunker_params={"chunk_size": 512}, embedder="openai", embedder_params=_SMALL),
    "r256-small":  RunConfig(chunker="recursive", chunker_params={"chunk_size": 256}, embedder="openai", embedder_params=_SMALL),
    "sem-small":   RunConfig(chunker="semantic",  chunker_params={"breakpoint_percentile": 95, "max_tokens": 512}, embedder="openai", embedder_params=_SMALL),
    "r512-voyage": RunConfig(chunker="recursive", chunker_params={"chunk_size": 512}, embedder="voyage", embedder_params=_VOYAGE),
    "r256-voyage": RunConfig(chunker="recursive", chunker_params={"chunk_size": 256}, embedder="voyage", embedder_params=_VOYAGE),
    "sem-voyage":  RunConfig(chunker="semantic",  chunker_params={"breakpoint_percentile": 95, "max_tokens": 512}, embedder="voyage", embedder_params=_VOYAGE),
}


def _ingest_cell(config: RunConfig) -> dict:
    """Ingest the corpus under `config`, metered. Idempotent (delete-by-source)."""
    start = time.perf_counter()
    with meter() as m:
        deleted, inserted = ingest_document(CORPUS, config, allowed_roles=["public"])
    s = m.summary()
    return {
        "chunks": inserted,
        "tokens": s["total_tokens"],
        "cost_usd": round(s["cost_usd"], 8),
        "seconds": round(time.perf_counter() - start, 3),
    }


def _warmup(config: RunConfig) -> None:
    """One discarded query so cold-start cost doesn't land in the measured numbers."""
    try:
        generation_metrics._run_pipeline("warm up the clients and connections", config)
    except Exception as exc:  # never let warm-up abort a cell
        print(f"    (warm-up failed, continuing: {type(exc).__name__})")


def _fmt(value, spec: str = ".3f") -> str:
    """Format a metric that may be None (RAGAS returns None when a judge job fails)."""
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def _summarize_gen_runs(runs: list[dict]) -> dict:
    """Mean + min/max across repeated generation-metric runs.

    `contributing_runs` records how many runs actually produced each metric, so a
    mean computed over fewer runs (because a judge job failed) is visible rather
    than silently passed off as a full-strength average."""
    out = {"num_runs": len(runs), "mean": {}, "min": {}, "max": {}, "contributing_runs": {}}
    for name in GEN_METRICS:
        vals = [r["aggregate"][name] for r in runs if r["aggregate"].get(name) is not None]
        out["contributing_runs"][name] = len(vals)
        if not vals:
            out["mean"][name] = out["min"][name] = out["max"][name] = None
            continue
        out["mean"][name] = round(sum(vals) / len(vals), 4)
        out["min"][name] = round(min(vals), 4)
        out["max"][name] = round(max(vals), 4)
    return out


def _summarize_cost_latency(runs: list[dict]) -> dict:
    """Mean per-query cost and phase latency across repeated runs."""
    cls = [r.get("cost_latency", {}) for r in runs if r.get("cost_latency")]
    if not cls:
        return {}
    phases = {p for c in cls for p in c.get("avg_seconds", {})}
    return {
        "avg_cost_usd": round(sum(c["avg_cost_usd"] for c in cls) / len(cls), 8),
        "avg_tokens": round(sum(c["avg_tokens"] for c in cls) / len(cls), 1),
        "avg_seconds": {
            p: round(sum(c["avg_seconds"].get(p, 0.0) for c in cls) / len(cls), 4)
            for p in sorted(phases)
        },
        "avg_total_seconds": round(sum(c["avg_total_seconds"] for c in cls) / len(cls), 4),
    }


def run_cell(name: str, config: RunConfig, gen_runs: int = 3, do_ingest: bool = True) -> dict:
    print(f"\n=== cell {name}  ({config.label}, {config.config_id}) ===")

    ingest = None
    if do_ingest:
        ingest = _ingest_cell(config)
        print(f"    ingest: {ingest['chunks']} chunks, {ingest['tokens']} tokens, "
              f"${ingest['cost_usd']:.6f}, {ingest['seconds']}s")

    _warmup(config)

    retrieval = retrieval_metrics.run(config)
    ra = retrieval["aggregate"]
    print(f"    retrieval: P@1={ra['precision_at_k']['1']:.3f} "
          f"R@5={ra['recall_at_k']['5']:.3f} MRR={ra['mrr']:.3f}")

    runs = []
    for i in range(gen_runs):
        try:
            r = generation_metrics.run(config)
        except Exception as exc:  # a failed run must not destroy the whole matrix
            print(f"    generation run {i + 1}/{gen_runs}: FAILED ({type(exc).__name__}: {exc})")
            continue
        runs.append(r)
        a = r["aggregate"]
        print(f"    generation run {i + 1}/{gen_runs}: faith={_fmt(a['faithfulness'])} "
              f"ans_rel={_fmt(a['answer_relevancy'])} ctx_prec={_fmt(a['context_precision'])} "
              f"ctx_rec={_fmt(a['context_recall'])} refusal={_fmt(a['refusal_accuracy'])}")

    return {
        "name": name,
        "label": config.label,
        "config_id": config.config_id,
        "chunker": config.chunker,
        "chunker_params": config.chunker_params,
        "embedder": config.embedder,
        "embedder_params": config.embedder_params,
        "retrieval_mode": config.retrieval_mode,
        "retrieval_k": config.retrieval_k,
        "ingest": ingest,
        "retrieval": ra,
        "generation": _summarize_gen_runs(runs),
        "cost_latency": _summarize_cost_latency(runs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Phase 5 experiment matrix.")
    ap.add_argument("--only", action="append", choices=sorted(CELLS),
                    help="run only these cells (repeatable); default: all")
    ap.add_argument("--gen-runs", type=int, default=3,
                    help="generation-metric repeats per cell (default 3)")
    ap.add_argument("--no-ingest", action="store_true", help="skip ingest (reuse existing tables)")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells even if a saved cell file already exists")
    args = ap.parse_args()

    names = args.only or list(CELLS)
    started = datetime.now().isoformat(timespec="seconds")
    print(f"matrix: {len(names)} cell(s), {args.gen_runs} generation run(s) each")

    # Each cell is written to disk the moment it finishes, and an already-present
    # cell file is reused instead of re-run. A crash (or a kill) therefore costs at
    # most the current cell, and re-invoking resumes where it stopped.
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    cells = []
    for n in names:
        cell_path = CELL_DIR / f"cell_{n}.json"
        if cell_path.exists() and not args.force:
            cells.append(json.loads(cell_path.read_text()))
            print(f"\n=== cell {n}: reusing existing result ({cell_path}) ===")
            continue
        try:
            cell = run_cell(n, CELLS[n], args.gen_runs, not args.no_ingest)
        except Exception as exc:
            print(f"    cell {n} FAILED ({type(exc).__name__}: {exc}) — continuing")
            continue
        cell_path.write_text(json.dumps(cell, indent=2))
        cells.append(cell)
        print(f"    saved cell -> {cell_path}")

    results = {
        "run": {"timestamp": started, "finished": datetime.now().isoformat(timespec="seconds"),
                "type": "matrix"},
        "config": {
            "corpus": CORPUS,
            "dataset": str(retrieval_metrics.DATASET_PATH),
            "generation_model": "gpt-4o-mini",
            "judge_model": generation_metrics.JUDGE_MODEL,
            "gen_runs_per_cell": args.gen_runs,
            "cells": names,
        },
        "cells": cells,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started.replace(":", "").replace("-", "")
    path = RESULTS_DIR / f"matrix_{stamp}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
