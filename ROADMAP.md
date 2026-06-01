# Roadmap

This document lays out the build sequence for the RAG Evaluation System and the principles guiding development. It is meant to be read alongside the [README](./README.md), which covers what the project is and why.

## Guiding principles

A few rules shape every phase below. They exist because the failure mode of a project like this is not "it doesn't work" — it's "it sprawled, it was never measured, and it never shipped."

- **Always have something that runs end-to-end.** Each phase leaves a working system, not a pile of half-finished components. Improvement, not construction, is the mode after Phase 1.
- **Measure before optimizing.** No change to chunking, embeddings, retrieval, or prompts is considered an improvement until the evaluation harness says so. This is why the eval layer is built second, before any tuning.
- **Change one variable at a time.** Results are only interpretable if everything but the variable under test is held constant. This applies to the experiment matrix and to debugging alike.
- **Quality per dollar and per second, not raw quality.** The best configuration is the one with the best quality relative to cost and latency for the use case — not the highest score on a single metric.
- **The README is a primary deliverable.** Recruiters often open the repository before the resume. The reasoning behind decisions matters as much as the code that implements them.
- **Resist scope creep.** A tight, well-measured, well-documented system beats a sprawling one. Stretch goals are explicitly marked as optional and live behind the core build.

## Build sequence

### Phase 1 — Walking skeleton

Get a single document flowing all the way through the pipeline: load → chunk (hardcoded recursive 512 / 15% overlap) → embed (OpenAI 3-small) → store in pgvector → retrieve top-K → generate a grounded answer → return it from a FastAPI endpoint. Ugly is acceptable; working is mandatory.

**Exit criterion:** one document, one query, one grounded answer, served over HTTP and verified with a curl request.

### Phase 2 — Evaluation harness

Before optimizing anything, build the instrument that tells you whether changes help. Assemble a labeled evaluation dataset (30–50 questions with ground-truth answers and labeled relevant chunks), wire up RAGAS/DeepEval for the four core metrics (faithfulness, answer relevance, context precision, context recall), and add the retrieval metrics (Precision@K, Recall@K, MRR). Run it against the Phase 1 baseline and record the numbers.

**Exit criterion:** a reproducible eval run producing a scored baseline that every later change is measured against.

### Phase 3 — Swappable interfaces

Refactor chunking and embedding so strategy and model are selected by configuration, not hardcoded. Define a common interface for each. Add the second chunking strategy (semantic) and a second embedding model behind these interfaces. This is unglamorous plumbing, but it is what makes the experiment matrix possible without rewriting the pipeline.

**Exit criterion:** chunking strategy and embedding model can be swapped via config, with no pipeline code changes.

### Phase 4 — Hybrid retrieval and access control

Add BM25 keyword search and fuse it with dense retrieval — the differentiating feature, and one that tends to produce measurable gains, especially on exact terms like codes, SKUs, and names. Then add the RBAC layer: a query carries a user identity, and retrieval filters to documents that user is permitted to see. Prove it with a negative test confirming a restricted user cannot retrieve a forbidden chunk.

**Exit criterion:** hybrid retrieval is available as a mode, and access control is enforced and verified by a passing negative test.

### Phase 5 — Experiment matrix

With everything swappable and measurable, run the full chunking × embedding matrix, collect results, and identify the winning configuration on this corpus. This is the centerpiece of the project and of the README.

**Exit criterion:** a completed results table with a defensible, data-backed choice of configuration.

### Phase 6 — Frontend and deployment

Build the React interface showing the answer alongside retrieved passages and their relevance scores. Containerize all services with Docker Compose. Wire the evaluation suite into GitHub Actions so a regression fails CI.

**Exit criterion:** the system runs via a single compose command, and CI runs tests plus evaluation on every push.

### Phase 7 — Documentation

Treated as a primary deliverable, not a wrap-up task. Finalize the README with an architecture diagram, the experiment matrix results as a table, and — most importantly — the reasoning: why the winning configuration won, what was surprising, what the trade-offs were.

**Exit criterion:** a reader can understand what was built, what was measured, and why the decisions were made, without running the code.

## The experiment matrix (Phase 5 detail)

Two variables, everything else held constant (eval dataset, top-K, generation model, prompts).

- **Chunking (3):** Recursive-512 (15% overlap), Recursive-256, Semantic. Two recursive sizes separate the *size* effect from the *strategy* effect.
- **Embedding (2):** OpenAI text-embedding-3-small (cheap baseline) and a quality contender (Voyage for code/technical corpora, Cohere for multilingual).

This is a 3 × 2 = 6-cell matrix. Each cell is scored on retrieval quality, generation quality, cost per million tokens, and latency, so the trade-offs are explicit rather than hidden behind a single number.

**Optional stretch:** compare hybrid vs. dense retrieval on the winning cell only, quantifying the hybrid gain without expanding the matrix to twelve cells.

## Suggested pace

A realistic cadence alongside a job search, assuming part-time effort:

- **Week 1:** Phases 1–2 (skeleton + eval harness)
- **Week 2:** Phases 3–4 (swappable components + hybrid/RBAC)
- **Week 3:** Phases 5–7 (matrix, deployment, documentation)

The schedule is a guide, not a contract. The non-negotiable is the ordering: the eval harness comes before tuning, and swappable interfaces come before the matrix.

## Out of scope (for now)

Kept off the critical path to protect focus. Candidates for a v2 if the core lands well:

- Reranking as a second retrieval stage
- Contextual or late chunking strategies
- Multi-corpus / multi-tenant support
- Continuous evaluation on live query logs
- Context trustworthiness checks (freshness, lineage, ownership) as a fifth eval dimension
