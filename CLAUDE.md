# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**Phase 1 (walking skeleton) is complete.** The pipeline runs end-to-end:
load → chunk → embed → store → retrieve → generate → HTTP. The exit criterion
(one document, one query, one grounded answer over HTTP, verified with curl) is
met. The mode from here is **improvement, not construction** — next is Phase 2
(the evaluation harness), which must be built before any tuning.

Read these before changing anything:
- [ROADMAP.md](ROADMAP.md) — the build sequence and non-negotiable ordering.
- [DECISIONS.md](DECISIONS.md) — why specific choices were made and where they deviate from the roadmap. **Add an entry when you make a decision or deviation.**
- [CONCEPTS.md](CONCEPTS.md) — how each pipeline component works, in plain language.

## Commands

All Python commands use the project venv (`.venv`). Prefix with `.venv/bin/` or activate it first.

```bash
# One-time environment setup
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Database (Postgres + pgvector) — must be running for ingest/retrieve/API
docker compose up -d                      # start; schema + extension auto-init on first run
docker compose down                       # stop (keeps data volume)
docker compose down -v                    # stop AND wipe the data volume (destructive)
.venv/bin/python scripts/check_db.py      # verify Python can reach the DB + pgvector

# Ingest a document (idempotent per source: re-running replaces that doc's rows)
.venv/bin/python -m app.store data/<document>

# Run the API, then query it
.venv/bin/uvicorn app.main:app --reload   # serves on http://127.0.0.1:8000
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"...","k":5}'

# Each pipeline stage has a manual check via its __main__:
.venv/bin/python -m app.loader    data/<document>   # raw text + stats
.venv/bin/python -m app.chunker   data/<document>   # chunk count, token sizes
.venv/bin/python -m app.embedder                    # live embedding smoke test (billed)
.venv/bin/python -m app.retriever "your question"   # top-K chunks + scores
.venv/bin/python -m app.generator "your question"   # grounded answer + sources
```

There is **no test suite yet** — verification is currently via the per-module
`__main__` checks and curl. A real test harness arrives with Phase 2.

## Architecture

The pipeline is a chain of small, single-purpose modules under [app/](app/), each
pure and unaware of the others' internals so they stay swappable (Phase 3):

- [app/loader.py](app/loader.py) — `load_document(path)` → raw text. PDF (pypdf) + txt/md.
- [app/chunker.py](app/chunker.py) — `chunk_text(text)` → `list[Chunk]`. Recursive, token-based (tiktoken), 512 / 15% overlap. See DECISIONS.md D1.
- [app/embedder.py](app/embedder.py) — `embed_texts()` / `embed_text()` → 1536-dim vectors via OpenAI. Batched, order-preserving.
- [app/store.py](app/store.py) — `ingest_document()` / `store_chunks()`. Delete-by-source then insert; populates `metadata` JSONB.
- [app/retriever.py](app/retriever.py) — `retrieve(query, k)` → `list[RetrievedChunk]`. Cosine-distance search, returns distance + similarity.
- [app/generator.py](app/generator.py) — `generate_answer(question, chunks)`. Grounded answer (gpt-4o-mini), refuses when context lacks the answer.
- [app/main.py](app/main.py) — FastAPI: `POST /query`, `GET /health`.
- [app/config.py](app/config.py) — typed `Settings` (pydantic-settings); single source for keys, DB URL, model names. Baseline config is hardcoded here, becomes swappable in Phase 3.
- [app/db.py](app/db.py) — `connect()`; one place that registers the pgvector adapter.

Data layer: a single `chunks` table ([db/init/02-schema.sql](db/init/02-schema.sql)) —
`id`, `content`, `embedding vector(1536)`, `metadata jsonb`. No vector index yet
(brute-force scan is correct at this scale). The `db/init/` scripts only run on a
**fresh** volume; schema changes to an existing DB must be applied manually.

Config/secrets: `.env` (gitignored) holds `OPENAI_API_KEY`, `DATABASE_URL`, and
the `POSTGRES_*` vars docker-compose reads. `.env.example` documents them.

## Non-negotiable constraints

These come from ROADMAP.md and override convenience. Violating the ordering produces a project that looks finished but proves nothing — which defeats the entire purpose.

- **The evaluation harness (Phase 2) is built before any tuning.** No change to chunking, embeddings, retrieval, or prompts counts as an "improvement" until the eval harness measures it as one. Do not optimize anything that cannot yet be scored.
- **Swappable interfaces (Phase 3) come before the experiment matrix (Phase 5).** The matrix must run by swapping config, with zero pipeline code changes.
- **Every phase leaves a system that runs end-to-end.** Don't break the working path to build the next feature.
- **Change one variable at a time** — in the experiment matrix and in debugging alike.
- **Optimize for quality per dollar and per second**, not raw quality. Score quality, cost, and latency together.
- **Retrieval and generation are evaluated separately** — IR metrics (Precision@K, Recall@K, MRR) for retrieval; LLM-as-judge (faithfulness, answer relevance, context precision/recall) for generation.
- **Access control is a requirement, not an add-on.** RBAC (Phase 4) filters retrieval by user identity and must be backed by a negative test proving a restricted user cannot reach a forbidden passage. The `metadata` JSONB column is where ownership fields will live.
- **The README is a primary deliverable.** The reasoning behind decisions (especially the Phase 5 matrix results) matters as much as the code.

## Phase 1 baseline config (implemented)

The baseline that later phases measure against, all currently hardcoded in
[app/config.py](app/config.py) / [app/chunker.py](app/chunker.py): recursive
chunking **512 tokens / 15% overlap**, **text-embedding-3-small** (1536-dim),
pgvector store, top-K=5 retrieval by cosine distance, **gpt-4o-mini** generation
at temperature 0. These become config-swappable in Phase 3, not before.

## Scope discipline

ROADMAP.md lists explicit out-of-scope items (reranking, contextual/late chunking, multi-tenant, continuous eval on live logs). Treat these as off the critical path. Before adding anything not in the current phase, check it against the "Resist scope creep" principle and the out-of-scope list.
