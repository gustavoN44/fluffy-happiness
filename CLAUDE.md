# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repo is in the **planning stage**. The only files are [README.md](README.md) (what the project is and why) and [ROADMAP.md](ROADMAP.md) (the build sequence and principles). There is no code, no build tooling, and no tests yet. When adding the first code, you are executing Phase 1 of the roadmap — read ROADMAP.md before starting, and update this file with real build/test/run commands once they exist.

## What this project is

A Retrieval-Augmented Generation (RAG) pipeline whose centerpiece is a **first-class evaluation harness**, not the pipeline itself. The pipeline (load → chunk → embed → store → retrieve → generate) is deliberately conventional. The substance — and the thing any change is judged against — lives in three places: independent evaluation of retrieval vs. generation, a controlled experiment comparing chunking/embedding choices, and role-based access control on retrieval.

Planned stack: Python + FastAPI + Pydantic; PostgreSQL with pgvector for the vector store; dense vector search with optional hybrid (dense + BM25) retrieval; RAGAS/DeepEval plus custom IR metrics for evaluation; React frontend; Docker Compose + GitHub Actions.

## Non-negotiable constraints

These come from ROADMAP.md and override convenience. Violating the ordering produces a project that looks finished but proves nothing — which defeats the entire purpose.

- **The evaluation harness (Phase 2) is built before any tuning.** No change to chunking, embeddings, retrieval, or prompts counts as an "improvement" until the eval harness measures it as one. Do not optimize anything that cannot yet be scored.
- **Swappable interfaces (Phase 3) come before the experiment matrix (Phase 5).** The matrix must run by swapping config, with zero pipeline code changes.
- **Every phase leaves a system that runs end-to-end.** After Phase 1 the mode is improvement, never construction-in-pieces. Don't break the working path to build the next feature.
- **Change one variable at a time** — in the experiment matrix and in debugging alike. Results are only interpretable if everything but the variable under test is held constant.
- **Optimize for quality per dollar and per second**, not raw quality. Score quality, cost, and latency together; the "best" config is the one the eval data selects, not the most sophisticated-sounding one.
- **Retrieval and generation are evaluated separately** — LLM-as-judge metrics (faithfulness, answer relevance, context precision/recall) for the pipeline, classic IR metrics (Precision@K, Recall@K, MRR) for retrieval against labeled relevance judgments.
- **Access control is a requirement, not an add-on.** RBAC filters retrieval by user identity and must be backed by a negative test proving a restricted user cannot reach a forbidden passage.
- **The README is a primary deliverable.** The reasoning behind decisions (especially the Phase 5 matrix results) matters as much as the code. Keep it current as the project grows.

## Phase 1 starting point (the walking skeleton)

The roadmap specifies the exact baseline config so Phase 1 isn't a design exercise: hardcoded recursive chunking at **512 / 15% overlap**, **OpenAI text-embedding-3-small**, store in pgvector, retrieve top-K, generate a grounded answer, return from a FastAPI endpoint. Exit criterion: one document, one query, one grounded answer over HTTP, verified with curl. Ugly is acceptable; working is mandatory.

## Scope discipline

ROADMAP.md lists explicit out-of-scope items (reranking, contextual/late chunking, multi-tenant, continuous eval on live logs). Treat these as off the critical path. Before adding anything not in the current phase, check it against the "Resist scope creep" principle and the out-of-scope list.
