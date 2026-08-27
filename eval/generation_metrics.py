"""Generation (LLM-as-judge) evaluation via RAGAS.

Scores the generated answers — the other half of the harness from
retrieval_metrics.py. For each answerable dataset question we run the REAL
pipeline (retrieve -> generate), then judge the result with four RAGAS metrics:

  - faithfulness        : are the answer's claims supported by the retrieved context?
  - answer_relevancy    : does the answer address the question?
  - context_precision   : are the relevant contexts ranked highly? (vs the reference)
  - context_recall      : does the context contain what's needed for the reference?

Unanswerable questions are judged separately by a refusal-accuracy check (did the
pipeline correctly abstain instead of fabricating), since faithfulness/relevancy
are ill-defined for an "I don't know" response.

IMPORTANT: run with the EVAL venv (RAGAS + LangChain live there, not in the app
venv):  .venv-eval/bin/python -m eval.generation_metrics
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas import RunConfig as RagasRunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.config import settings
from app.generator import generate_answer
from app.metering import meter
from app.pipeline import BASELINE, RunConfig
from app.retriever import retrieve

warnings.filterwarnings("ignore", category=DeprecationWarning)

DATASET_PATH = Path("eval/dataset.json")
RESULTS_DIR = Path("eval/results")
JUDGE_MODEL = "gpt-4o-mini"  # LLM-as-judge; cheap, keeps repeated eval runs affordable

# RAGAS defaults to max_workers=16, which saturates the API during long back-to-back
# runs (the Phase 5 matrix) and produced 150 TimeoutErrors -> None metrics. Fewer
# concurrent judge calls plus a longer timeout trades a little speed for reliability.
_RAGAS_RUN_CONFIG = RagasRunConfig(timeout=300, max_workers=4, max_retries=10)

# RAGAS metric column name -> our clean output name
_METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


def _is_refusal(answer: str) -> bool:
    return "i don't know" in answer.lower()


def _run_pipeline(question: str, config: RunConfig) -> tuple[str, list[str], dict]:
    """Run the real pipeline under `config`, metered. Returns
    (answer, retrieved context texts, meter summary for this one query)."""
    with meter() as m:
        chunks = retrieve(question, config=config, k=config.retrieval_k)
        answer = generate_answer(question, chunks)
    return answer, [c.content for c in chunks], m.summary()


def _aggregate_meters(summaries: list[dict]) -> dict:
    """Per-query averages of cost, tokens, and phase latency across all queries."""
    n = len(summaries)
    if not n:
        return {}
    phases = {p for s in summaries for p in s["seconds"]}
    models = {mo for s in summaries for mo in s["tokens"]}
    return {
        "num_queries": n,
        "avg_cost_usd": round(sum(s["cost_usd"] for s in summaries) / n, 8),
        "avg_tokens": round(sum(s["total_tokens"] for s in summaries) / n, 1),
        "avg_tokens_by_model": {
            mo: round(sum(
                s["tokens"].get(mo, {}).get("input", 0) + s["tokens"].get(mo, {}).get("output", 0)
                for s in summaries) / n, 1)
            for mo in sorted(models)
        },
        "avg_seconds": {
            p: round(sum(s["seconds"].get(p, 0.0) for s in summaries) / n, 4)
            for p in sorted(phases)
        },
        "avg_total_seconds": round(sum(s["total_seconds"] for s in summaries) / n, 4),
    }


def _judge():
    key = settings.openai_api_key
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0, api_key=key))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=settings.embedding_model, api_key=key)
    )
    return llm, emb


def run(config: RunConfig = BASELINE) -> dict:
    dataset = _load_dataset()
    answerable = [d for d in dataset if d["answerable"]]
    unanswerable = [d for d in dataset if not d["answerable"]]

    # --- answerable: RAGAS four metrics on the live pipeline output ---
    samples, generated = [], []
    meters: list[dict] = []
    for item in answerable:
        answer, contexts, usage = _run_pipeline(item["question"], config)
        meters.append(usage)
        generated.append({"id": item["id"], "answer": answer, "contexts": contexts})
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=item["ground_truth_answer"],
        ))

    llm, emb = _judge()
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(),
                 LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm, embeddings=emb, run_config=_RAGAS_RUN_CONFIG,
    )
    df = result.to_pandas()

    per_query = []
    for i, item in enumerate(answerable):
        row = df.iloc[i]
        scores = {clean: (float(row[raw]) if row.get(raw) == row.get(raw) else None)
                  for raw, clean in _METRIC_NAMES.items()}
        per_query.append({
            "id": item["id"], "category": item["category"],
            "question": item["question"],
            "answer": generated[i]["answer"],
            "scores": scores,
        })

    def _mean(name: str) -> float:
        vals = [q["scores"][name] for q in per_query if q["scores"][name] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # --- unanswerable: refusal-accuracy ---
    refusals = []
    for item in unanswerable:
        answer, _, usage = _run_pipeline(item["question"], config)
        meters.append(usage)
        correct = _is_refusal(answer)
        refusals.append({"id": item["id"], "answer": answer, "correct_abstention": correct})
    refusal_accuracy = (round(sum(r["correct_abstention"] for r in refusals) / len(refusals), 4)
                        if refusals else None)

    return {
        "run": {"timestamp": datetime.now().isoformat(timespec="seconds"), "type": "generation"},
        "config": {
            "dataset": str(DATASET_PATH),
            "config_id": config.config_id,
            "label": config.label,
            "judge_model": JUDGE_MODEL,
            "generation_model": settings.generation_model,
            "embedder": config.embedder,
            "embedder_params": config.embedder_params,
            "k": config.retrieval_k,
            "num_answerable": len(answerable),
            "num_unanswerable": len(unanswerable),
        },
        "aggregate": {
            "faithfulness": _mean("faithfulness"),
            "answer_relevancy": _mean("answer_relevancy"),
            "context_precision": _mean("context_precision"),
            "context_recall": _mean("context_recall"),
            "refusal_accuracy": refusal_accuracy,
        },
        "cost_latency": _aggregate_meters(meters),
        "per_query": per_query,
        "refusals": refusals,
    }


def _save(results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = results["run"]["timestamp"].replace(":", "").replace("-", "")
    path = RESULTS_DIR / f"generation_{stamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def _print_summary(results: dict) -> None:
    agg, cfg = results["aggregate"], results["config"]
    print(f"\nGeneration metrics — {cfg['num_answerable']} answerable queries "
          f"(judge: {cfg['judge_model']}, gen: {cfg['generation_model']})")
    for name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        v = agg[name]
        print(f"  {name:<18}: {v:.3f}" if v is not None else f"  {name:<18}: n/a")
    print(f"\n  refusal_accuracy   : {agg['refusal_accuracy']:.3f} "
          f"({sum(r['correct_abstention'] for r in results['refusals'])}/{cfg['num_unanswerable']} "
          f"unanswerable correctly refused)")


if __name__ == "__main__":
    import argparse

    from app.pipeline import add_config_arg, resolve_config

    ap = argparse.ArgumentParser(description="Generation (LLM-judge) metrics.")
    add_config_arg(ap)
    results = run(resolve_config(ap.parse_args().config))
    path = _save(results)
    _print_summary(results)
    print(f"  saved: {path}")
