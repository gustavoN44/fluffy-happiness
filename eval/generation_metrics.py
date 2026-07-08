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
import re
import warnings
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
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
from app.retriever import DEFAULT_K, retrieve

warnings.filterwarnings("ignore", category=DeprecationWarning)

DATASET_PATH = Path("eval/dataset.json")
RESULTS_DIR = Path("eval/results")
JUDGE_MODEL = "gpt-4o-mini"  # LLM-as-judge; cheap, keeps repeated eval runs affordable

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


def _run_pipeline(question: str) -> tuple[str, list[str]]:
    """Run the real pipeline; return (answer, retrieved context texts)."""
    chunks = retrieve(question, k=DEFAULT_K)
    answer = generate_answer(question, chunks)
    return answer, [c.content for c in chunks]


def _judge():
    key = settings.openai_api_key
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0, api_key=key))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=settings.embedding_model, api_key=key)
    )
    return llm, emb


def run() -> dict:
    dataset = _load_dataset()
    answerable = [d for d in dataset if d["answerable"]]
    unanswerable = [d for d in dataset if not d["answerable"]]

    # --- answerable: RAGAS four metrics on the live pipeline output ---
    samples, generated = [], []
    for item in answerable:
        answer, contexts = _run_pipeline(item["question"])
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
        llm=llm, embeddings=emb,
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
        answer, _ = _run_pipeline(item["question"])
        correct = _is_refusal(answer)
        refusals.append({"id": item["id"], "answer": answer, "correct_abstention": correct})
    refusal_accuracy = (round(sum(r["correct_abstention"] for r in refusals) / len(refusals), 4)
                        if refusals else None)

    return {
        "run": {"timestamp": datetime.now().isoformat(timespec="seconds"), "type": "generation"},
        "config": {
            "dataset": str(DATASET_PATH),
            "judge_model": JUDGE_MODEL,
            "generation_model": settings.generation_model,
            "embedding_model": settings.embedding_model,
            "k": DEFAULT_K,
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
    results = run()
    path = _save(results)
    _print_summary(results)
    print(f"  saved: {path}")
