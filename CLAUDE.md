# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**Phases 1–5 are complete.** The pipeline runs end-to-end (load → chunk → embed →
store → retrieve → generate → HTTP), the evaluation harness scores retrieval and
generation separately, components are swappable by config, RBAC is enforced and
proven by a negative test, and the Phase 5 experiment matrix has run to completion
with a data-backed choice of configuration.

The served configuration is **no longer the Phase 1 baseline**. It was chosen by
measurement — see `PRODUCTION` in [app/pipeline.py](app/pipeline.py) and DECISIONS.md
D9/D10.

What remains: the **README**, which ROADMAP.md treats as a primary deliverable.

Read these before changing anything:
- [ROADMAP.md](ROADMAP.md) — the build sequence and non-negotiable ordering.
- [DECISIONS.md](DECISIONS.md) — why specific choices were made and where they deviate from the roadmap (D1–D10). **Add an entry when you make a decision or deviation.**
- [CONCEPTS.md](CONCEPTS.md) — how each pipeline component works, in plain language.
- [FINDINGS.md](FINDINGS.md) — what the measurements actually showed (F1–F7), including results that were surprising or that revised an earlier belief. **Add an entry when data contradicts an expectation.**

## Two virtualenvs — this matters

RAGAS pins an incompatible LangChain line, so eval dependencies are isolated
(DECISIONS.md D5). Using the wrong one produces confusing `ModuleNotFoundError`s:

| venv | requirements | use for |
|---|---|---|
| `.venv` | `requirements.txt` | the app, ingest, retrieval metrics, RBAC test |
| `.venv-eval` | `requirements-eval.txt` | anything importing RAGAS — `eval.generation_metrics`, `eval.matrix`, `eval.run_all` |

Anything that imports `eval.matrix` transitively imports RAGAS, so it needs
`.venv-eval`. Scripts run from outside the package need `PYTHONPATH=.`.

## Commands

```bash
# One-time environment setup
python3 -m venv .venv       && .venv/bin/python      -m pip install -r requirements.txt
python3 -m venv .venv-eval  && .venv-eval/bin/python -m pip install -r requirements-eval.txt

# Database (Postgres + pgvector) — must be running for ingest/retrieve/API
docker compose up -d                      # start; schema + extension auto-init on first run
docker compose down                       # stop (keeps data volume)
docker compose down -v                    # stop AND wipe the data volume (destructive)
.venv/bin/python scripts/check_db.py      # verify Python can reach the DB + pgvector

# Ingest a document (idempotent per source: re-running replaces that doc's rows)
.venv/bin/python -m app.store data/<document>

# Run the API (serves PRODUCTION), then query it
.venv/bin/uvicorn app.main:app --reload   # serves on http://127.0.0.1:8000
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"...","k":5}'
# RBAC: roles come from a header; absent means public
curl -s -X POST http://127.0.0.1:8000/query -H "X-User-Roles: admin" ...

# Evaluation
.venv/bin/python      -m eval.retrieval_metrics    # IR metrics (deterministic, cheap)
.venv-eval/bin/python -m eval.generation_metrics   # RAGAS judge (billed, slow)
.venv-eval/bin/python -m eval.run_all              # both

# Experiment matrix (Phase 5) — 6 cells, resumable
.venv-eval/bin/python -m eval.matrix                        # all cells; reuses saved ones
.venv-eval/bin/python -m eval.matrix --only r512-small --force --gen-runs 3
.venv-eval/bin/python -m eval.matrix --no-ingest             # reuse existing tables

# RBAC negative test (Phase 4 exit criterion) — covers BASELINE and PRODUCTION
.venv/bin/python -m tests.test_rbac

# Per-stage manual checks via __main__:
.venv/bin/python -m app.loader    data/<document>   # raw text + stats
.venv/bin/python -m app.chunker   data/<document>   # chunk count, token sizes
.venv/bin/python -m app.embedder                    # live embedding smoke test (billed)
.venv/bin/python -m app.retriever "your question"   # top-K chunks + scores
.venv/bin/python -m app.generator "your question"   # grounded answer + sources
```

**There is no pytest suite.** `tests/test_rbac.py` is a standalone script with a
`main()` returning an exit code — run it with `python -m tests.test_rbac`, not
pytest (which isn't installed in either venv). Verification is that test, the eval
harness, and the per-module `__main__` checks.

## Architecture

Small, single-purpose modules under [app/](app/), each unaware of the others'
internals so they stay swappable:

- [app/loader.py](app/loader.py) — `load_document(path)` → raw text. PDF (pypdf) + txt/md.
- [app/interfaces.py](app/interfaces.py) — `Chunk` dataclass; `Chunker` / `Embedder` runtime-checkable Protocols. The seam everything else plugs into (D6).
- [app/chunker.py](app/chunker.py) — `RecursiveChunker`. Token-based (tiktoken), configurable size, 15% overlap. See D1 and **D4** (a real overlap-seam bug found by measurement).
- [app/semantic_chunker.py](app/semantic_chunker.py) — `SemanticChunker`. Splits at embedding-distance breakpoints; its breakpoint model is deliberately decoupled from the storage embedder.
- [app/embedder.py](app/embedder.py) / [app/voyage_embedder.py](app/voyage_embedder.py) — `OpenAIEmbedder`, `VoyageEmbedder`. Batched, order-preserving; both report real usage to the meter. Voyage uses asymmetric `input_type` (document vs query).
- [app/pipeline.py](app/pipeline.py) — **`RunConfig`, the registries, and the `BASELINE` / `PRODUCTION` configs.** Start here to understand how a config becomes a running pipeline.
- [app/store.py](app/store.py) — `ingest_document(path, config, allowed_roles)`. Delete-by-source then insert; writes `metadata` JSONB including ACL.
- [app/retriever.py](app/retriever.py) — `retrieve(query, config, k, user)`. Dispatches dense or hybrid; both RBAC-filtered.
- [app/lexical.py](app/lexical.py) — BM25 keyword search (`rank_bm25`), RBAC-filtered corpus load. Index rebuilt per query — fine at this scale, first thing to cache at a larger one.
- [app/rbac.py](app/rbac.py) — `User`, `PUBLIC`, `ADMIN`, `where_clause`. Default-deny via `jsonb_exists_any`.
- [app/metering.py](app/metering.py) — `meter()` ContextVar-scoped manager, `record_usage`, `timed`. **Token counts are stored raw**, so re-pricing never requires a re-run.
- [app/generator.py](app/generator.py) — `generate_answer(question, chunks)`. Grounded answer (gpt-4o-mini), refuses when context lacks the answer.
- [app/main.py](app/main.py) — FastAPI: `POST /query`, `GET /health`. `ACTIVE_CONFIG = PRODUCTION`; reads `X-User-Roles`.
- [app/db.py](app/db.py) — `connect()`, `config_table()`, `register_config()`. Registers the pgvector adapter.

Eval harness under [eval/](eval/): `dataset.json` (35 items), `retrieval_metrics.py`
(deterministic IR), `generation_metrics.py` (RAGAS), `matrix.py` (Phase 5 runner),
`run_all.py`. Results land in `eval/results/`, with per-cell files in
`eval/results/cells/` — the matrix **reuses a saved cell instead of re-running it**,
so a crash or a quota exhaustion costs at most the current cell. Use `--force` to
re-run one.

**Data layer: table-per-config.** Each `RunConfig` gets its own `chunks_<config_id>`
table (embedding dimensionality varies between models, so one shared table is
impossible), plus a `configs` registry. `config_id` is a hash of chunker + embedder
settings; **`retrieval_mode` is deliberately excluded** so dense and hybrid read the
same table (D6). No vector index — brute-force scan is correct at this scale.
`db/init/` scripts run only on a **fresh** volume; schema changes to a live DB are manual.

Config/secrets: `.env` (gitignored) holds `OPENAI_API_KEY`, `VOYAGE_API_KEY`,
`DATABASE_URL`, and the `POSTGRES_*` vars docker-compose reads. `.env.example` documents them.

## The two configs, and why both exist

```python
BASELINE    # recursive-512 x text-embedding-3-small, dense  — Phase 1
PRODUCTION  # recursive-256 x voyage-4-large,         hybrid — Phase 5 winner, served
```

**Do not re-point `BASELINE` at the winner.** It means "what we measure against";
changing it would silently invalidate every stored result and every document that
cites it. Add a new named config instead.

## Non-negotiable constraints

These come from ROADMAP.md and override convenience.

- **Nothing is an "improvement" until the harness measures it as one.** No change to chunking, embeddings, retrieval, or prompts counts until scored.
- **The matrix runs by swapping config, with zero pipeline code changes.** Preserve that property.
- **Every phase leaves a system that runs end-to-end.**
- **Change one variable at a time** — in experiments and debugging alike.
- **Optimize for quality per dollar and per second**, not raw quality.
- **Retrieval and generation are evaluated separately** — deterministic IR metrics and LLM-as-judge are a cross-check, not redundancy.
- **Access control is a requirement, not an add-on.** RBAC must be backed by a negative test, and that test covers **every config the system depends on** — a table the API serves but the test skips is untested access control.
- **The README is a primary deliverable.**

## Evaluation discipline (learned the hard way)

Costly lessons already paid for — see FINDINGS.md:

- **A benchmark everything passes measures nothing** (F2). Check the score distribution before trusting a weak measured effect.
- **State the scoring rule before looking at results**, and show the conclusion survives rejecting it (F5, D9).
- **30 questions rank configurations; they do not certify the ranking** (F6). Report McNemar / bootstrap alongside means, and don't claim significance the sample can't support.
- **Wall-clock latency silently absorbs client retries.** A rate-limited run reported 5.26s generation vs a true 0.91s while its quality metrics were unaffected. Treat latency from a throttled run as invalid.
- **Cost intuition does not predict quality importance** (F3). The cheapest component was the most decisive.
- **RAGAS returns `None` when a judge job fails** — never format a metric without a None guard, and never let one failed run destroy a whole matrix.

## Scope discipline

ROADMAP.md lists explicit out-of-scope items (reranking, contextual/late chunking,
multi-tenant, continuous eval on live logs). Treat these as off the critical path.
Before adding anything not in the current phase, check it against the "Resist scope
creep" principle and the out-of-scope list.
