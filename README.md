# RAG Evaluation System

A production-style Retrieval-Augmented Generation (RAG) pipeline built around a first-class evaluation harness, with role-based access control on retrieval. The point of this project is not to demonstrate that RAG *works* — that's a solved tutorial — but to demonstrate the judgment that separates a system that passes a demo from one that survives production: measuring retrieval and generation quality systematically, comparing design choices on real data, and documenting the decisions behind them.

## What this is

Most RAG portfolio projects are a single notebook that loads a PDF, embeds it, and calls an LLM. They prove you can wire the pieces together. They prove nothing about whether the result is any good, where it breaks, or what trade-offs were made along the way.

This project inverts that emphasis. The pipeline itself is deliberately conventional — load, chunk, embed, store, retrieve, generate. The substance is in three places that tutorials skip:

- **An evaluation layer** that scores retrieval and generation as separate stages, so failures can be diagnosed rather than guessed at.
- **A systematic comparison** of chunking strategies and embedding models, run as a controlled experiment on this corpus rather than chosen by default or by reputation.
- **Access control on retrieval**, so that *who is asking* constrains *what can be retrieved* — a requirement in any real enterprise deployment and a dimension most demos ignore entirely.

## What it does

A user submits a natural-language question through an API. The system retrieves the most relevant passages from an indexed document corpus, filters them to those the user is permitted to see, and generates an answer grounded strictly in the retrieved context. The accompanying interface surfaces not just the answer but the passages it was built from and their relevance scores, so the retrieval step is transparent rather than a black box.

Underneath, every meaningful configuration choice — chunk size and strategy, embedding model, retrieval depth, dense versus hybrid search — is swappable and measurable, so the "best" configuration is the one the evaluation data selects, not the one that sounded most sophisticated.

## What it aims to demonstrate

The system is designed to send specific signals to anyone reading the repository:

- **Retrieval and generation are evaluated independently**, using both LLM-as-judge metrics (faithfulness, answer relevance, context precision, context recall) and classic information-retrieval metrics (Precision@K, Recall@K, MRR) backed by labeled relevance judgments.
- **Design decisions are made by measurement.** Chunking strategies and embedding models are compared head-to-head in a controlled experiment matrix, scored on quality, cost, and latency together — because the best system is the best quality *per dollar and per second*, not the highest raw score.
- **Quality is defended over time.** The evaluation suite runs in CI, so a change that degrades retrieval or introduces hallucination fails the build instead of reaching users.
- **Security is treated as a requirement, not an add-on.** Role-based access control filters retrieval by user identity, verified with negative tests proving a restricted user cannot reach a forbidden passage.
- **The whole thing is deployable.** It runs as containerized services, not as a notebook that only works on the author's laptop.

## Tech stack

- **Pipeline & API:** Python, FastAPI, Pydantic
- **Vector store:** PostgreSQL with the pgvector extension
- **Retrieval:** dense vector search with an optional hybrid (dense + BM25 keyword) mode
- **Evaluation:** RAGAS / DeepEval for pipeline metrics, plus custom retrieval metrics
- **Frontend:** React (answer, source passages, and relevance scores)
- **Infrastructure:** Docker Compose, GitHub Actions (tests + evaluation in CI)

## Status

In active development. Build is sequenced in phases, beginning with an end-to-end "walking skeleton" (single document, single query, single answer over HTTP) and progressively adding the evaluation harness, swappable components, hybrid retrieval and access control, the experiment matrix, and deployment.

---

*This README will grow as the project does. The experiment matrix results and the reasoning behind the chosen configuration will be added once that phase is complete — that analysis is the centerpiece of the project, not an appendix to it.*
