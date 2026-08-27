# Decisions & Deviations

A running log of decisions taken while building the RAG Evaluation System, and
any deviations from [ROADMAP.md](ROADMAP.md). Each entry records *what* was
decided, *why*, the *tradeoffs* accepted, the **ROADMAP phase and aspect**
affected, and the **scripts** that implement it.

This is a deliberate deliverable: per the roadmap, "the reasoning behind
decisions matters as much as the code." Entries are append-only and ordered
newest-last; supersede rather than rewrite when something changes.

> Note on the word "improvement": the roadmap forbids calling anything an
> improvement before the Phase 2 eval harness can score it. Entries here that
> predate the harness document **decisions and verified behavior**, not quality
> claims. Whether a choice is actually *good* for this corpus is deferred to the
> eval data.

---

## Entry template

```
## D<N> — <short title>
- **Date:** YYYY-MM-DD
- **Status:** Accepted | Superseded by D<M> | Revisit in Phase <X>
- **ROADMAP reference:** Phase <N> — <phase name> → <specific aspect/line>
- **Type:** Decision (roadmap silent) | Deviation (differs from roadmap)
- **Implemented in:** <files / functions>

**Context** — what prompted the decision.
**Decision** — what was chosen.
**Why** — the reasoning.
**Tradeoffs** — what we gave up / risks accepted.
**Verification** — how we confirmed the behavior (not the quality).
```

---

## D1 — Chunker: token-based recursion with a global overlap pass

- **Date:** 2026-06-10
- **Status:** Accepted (chunking strategy/params become config-swappable in Phase 3; current values are the Phase 1 baseline)
- **ROADMAP reference:** Phase 1 — Walking skeleton → "chunk (hardcoded recursive 512 / 15% overlap)" ([ROADMAP.md](ROADMAP.md) line 20). Related: Phase 3 (swappable interfaces) and Phase 5 (chunking variable in the experiment matrix, [ROADMAP.md](ROADMAP.md) line 64).
- **Type:** Mostly Decision (roadmap specifies "recursive 512 / 15% overlap" but not the unit or the algorithm); one **Deviation** in how overlap is realized (see below).
- **Implemented in:** [app/chunker.py](app/chunker.py) — `chunk_text()`, `_recursive_split()`, `_merge()`, `_add_overlap()`, `_tail_tokens()`. Manual check via `python -m app.chunker <doc>`.

**Context**
The roadmap fixes the Phase 1 chunker as "recursive 512 / 15% overlap" but
leaves three things unspecified: (a) whether "512" counts characters or tokens,
(b) whether to hand-write the splitter or pull in a library, and (c) exactly how
overlap is produced. These had to be decided to write the module.

**Decision**
1. **Measure in tokens, not characters.** "512" and the 15% overlap are counted
   with `tiktoken`'s `cl100k_base` encoding — the same tokenizer
   `text-embedding-3-small` uses.
2. **Hand-write the recursive splitter** rather than adopt LangChain's
   `RecursiveCharacterTextSplitter`.
3. **Realize overlap as a global stride pass.** Build non-overlapping base
   chunks at a stride budget of `512 − 77 = 435` tokens, then prepend each chunk
   with the trailing 77 tokens of its predecessor (`_add_overlap`). Final chunks
   stay ≤ 512.

**Why**
1. *Tokens:* the project optimizes for **quality per dollar and per second**
   ([ROADMAP.md](ROADMAP.md) line 12), and cost plus the embedding model's input
   limit are both denominated in tokens. A character budget is only a loose
   proxy; a token budget means "512" is the same quantity the model and the
   bill see.
2. *Hand-written:* avoids pulling LangChain's large dependency tree in for one
   function, keeps full control over separators and overlap behavior, and makes
   the Phase 3 swap to a common chunker interface straightforward.
3. *Global overlap pass (the deviation):* the first implementation wove overlap
   into the recursive descent (the standard piece-bounded approach LangChain
   uses — overlap is retained as whole trailing pieces). Measurement showed it
   produced **zero overlap at 12 of 30 boundaries**, because wherever a chunk
   was a single paragraph larger than the 77-token overlap budget, the whole
   piece was dropped instead of carried forward. That defeats the purpose of
   overlap (not losing facts that straddle a boundary) and misses the stated
   "15% overlap" at 40% of boundaries. The stride model guarantees the overlap
   at every boundary regardless of paragraph size.

**Tradeoffs**
- **Overlap regions may begin mid-sentence.** The 77-token seed is sliced from
  the token stream, so the *duplicated* prefix of a chunk can start mid-word.
  Accepted: it is a recall safety margin, not primary content, and the chunk's
  substantive (end) boundaries remain aligned to natural separators.
- **Stride semantics shift slightly.** Each chunk now holds ≤ 435 tokens of new
  content + ≤ 77 repeated, rather than ≤ 512 new. This yields a few more chunks
  (34 vs ~30 on the test PDF) — marginally more vectors to store and scan, which
  is irrelevant at Phase 1 scale (no vector index, brute-force scan).
- **Token slicing cost.** `_add_overlap` re-encodes each chunk tail once; trivial
  at this scale.
- **Not yet validated for quality.** Per the roadmap, whether 512 / 15% / token
  overlap is *good* for this corpus is unknown until the Phase 2 harness scores
  it. This entry documents behavior, not merit.

**Verification**
On `data/mota-origenes.pdf` (13-page PDF, 50,430 chars) after the change:
34 chunks; tokens/chunk min 94 / max 512 / avg 418; **0 chunks over the 512
budget**; overlap present at **33/33 boundaries** (avg 74 tokens, target 77).
On `data/sample.txt` (187 tokens): collapses to 1 chunk with no overlap, as
expected for sub-budget input.

> **Update (2026-06-28, see D4):** the overlap-seam join described here had a bug
> — `.strip()` dropped the boundary whitespace, so the seed collided with the next
> chunk's first word (`"in"`+`"early"` → `"inearly"`) in 33/34 chunks. Fixed in D4.

---

## D2 — Phase 1 baseline: generation, retrieval depth, and API contract

- **Date:** 2026-06-28
- **Status:** Accepted (these are the Phase 1 baseline; generation/retrieval params become config-swappable in Phase 3 and the generation model is held constant across the Phase 5 matrix)
- **ROADMAP reference:** Phase 1 — Walking skeleton → "retrieve top-K → generate a grounded answer → return it from a FastAPI endpoint" ([ROADMAP.md](ROADMAP.md) line 20). Related: Phase 2 (these are what the harness first scores), Phase 5 (generation model held constant, [ROADMAP.md](ROADMAP.md) line 62).
- **Type:** Decision (roadmap fixes the *embedding* model and chunker params but leaves generation model, K, prompt, and API shape open).
- **Implemented in:** [app/generator.py](app/generator.py), [app/main.py](app/main.py), [app/retriever.py](app/retriever.py) (`DEFAULT_K`), [app/config.py](app/config.py) (`generation_model`, `generation_temperature`). Checks: `python -m app.generator "..."`, `uvicorn app.main:app` + curl.

**Context**
The roadmap pins the embedding model (text-embedding-3-small) and the chunker
(recursive 512 / 15%) but says only "retrieve top-K → generate a grounded answer
→ FastAPI endpoint." The generation model, K, the grounding prompt, and the HTTP
contract all had to be chosen to finish the skeleton.

**Decision**
1. **Generation model: `gpt-4o-mini`, temperature 0.0.**
2. **Retrieval depth: top-K = 5**, as an overridable argument (`DEFAULT_K`), not hardcoded in SQL.
3. **Grounding prompt:** a system prompt instructing the model to answer from the provided context only, ignore prior knowledge, and reply exactly *"I don't know based on the provided context."* when the answer isn't present.
4. **API contract:** `POST /query` `{question, k=5}` → `{answer, sources[]}` where each source carries `source`, `chunk_index`, `similarity`, `distance`, `content`; plus `GET /health`. Typed with Pydantic.

**Why**
1. *gpt-4o-mini / temp 0:* mirrors the "cheap baseline" logic of the embedding
   choice — Phase 2 will run the eval suite repeatedly, so a cheap, fast model
   keeps iteration affordable; temperature 0 makes answers reproducible so eval
   scores reflect the pipeline, not sampling noise. Quality-per-dollar, not raw
   quality ([ROADMAP.md](ROADMAP.md) line 12).
2. *K = 5:* a conventional RAG default and a sane starting point; the *right* K is
   an empirical question the Phase 2 harness will answer, so it's a parameter, not
   a constant.
3. *Grounding prompt:* "grounded answer" is the explicit Phase 1 requirement, and
   the refusal behavior is precisely what the Phase 2 faithfulness metric scores.
4. *Sources in the response:* the README requires retrieval be transparent
   (passages + scores surfaced), so the contract returns them from day one.

**Tradeoffs**
- **Baseline ≠ best.** None of these are validated as good yet — per the roadmap,
  that's deferred to the eval harness. gpt-4o-mini may underperform a larger model
  on faithfulness; K=5 may over- or under-retrieve for this corpus. Documented as
  starting points, not conclusions.
- **Single-shot prompt, no citations-in-text.** The model is told which passages
  it used (numbered context) but isn't required to cite inline. Kept simple for
  Phase 1; revisit if faithfulness scoring wants span-level attribution.
- **Temperature 0 is not fully deterministic.** OpenAI outputs can still vary
  slightly run-to-run; temp 0 minimizes but doesn't eliminate it.
- **No auth / rate limiting on the endpoint.** Out of scope for Phase 1; RBAC on
  retrieval arrives in Phase 4.

**Verification**
`POST /query` with "Where was cannabis first domesticated?" returned a grounded
answer ("...first domesticated in early Neolithic times in East Asia") with 3
scored sources. The grounding guard held: an off-topic question ("capital of
France?") returned "I don't know based on the provided context." despite chunks
being retrieved. `GET /health` returns `{"status":"ok"}`.

---

## D3 — Evaluation dataset: span-based gold labels, LLM-drafted + human-reviewed

- **Date:** 2026-06-28
- **Status:** Accepted (pilot of 14 items; scales to the roadmap's 30–50 in a later Phase 2 pass)
- **ROADMAP reference:** Phase 2 — Evaluation harness → "Assemble a labeled evaluation dataset (30–50 questions with ground-truth answers and labeled relevant chunks)" ([ROADMAP.md](ROADMAP.md) line 26). Related: Phase 5 (chunking is swapped in the experiment matrix, [ROADMAP.md](ROADMAP.md) line 64), which drove the span-based labeling choice.
- **Type:** Decision (roadmap requires a labeled dataset but leaves authoring method, label representation, size cadence, and negatives open).
- **Implemented in:** [eval/dataset.json](eval/dataset.json) (14 items). Validation: each gold span is confirmed to occur in the source (whitespace-normalized) via an inline check; unanswerable items are asserted to have no spans.

**Context**
Phase 2 needs a labeled dataset to score retrieval and generation. Four choices
were open: how questions/answers are authored, how "relevant" content is labeled,
how big to go first, and whether to include unanswerable questions.

**Decision**
1. **Authoring: LLM-drafted, human-reviewed.** Claude drafted questions,
   ground-truth answers, and gold spans grounded in the corpus; the user reviewed
   and approved the set as the gold standard.
2. **Relevance labels: span-based gold passages.** Each answerable item stores one
   or more *verbatim text spans* from the source document. A chunk is counted
   "relevant" (for IR metrics) if it contains a gold span, matched with whitespace
   normalized on both sides.
3. **Size: pilot ~10–15 first, then scale.** Built 14 items (12 answerable — 10
   factual + 2 multi-hop — plus 2 unanswerable) to prove the harness end-to-end
   before investing in the full 30–50.
4. **Include unanswerable questions.** ~14% (2 of 14) are on-topic but genuinely
   unanswerable from the corpus; their ground truth is the exact refusal string.

**Why**
1. *LLM-drafted + reviewed:* a human writing 30–50 grounded items with exact
   quoted spans is slow; unreviewed LLM output is an untrustworthy ruler. Drafting
   then human-approving balances effort against the fidelity the gold set demands.
2. *Span-based labels (the consequential one):* Phase 5 swaps the chunker
   (Recursive-512 / 256 / semantic), which changes chunk boundaries and would
   invalidate any chunk-index labels — forcing a relabel per chunking. Spans are
   chunking-independent: the same gold passage maps to whichever chunks contain it,
   so one labeling survives the whole experiment matrix. Cost: a small
   span→chunk mapping step at metric time (built in the retrieval-metrics step).
3. *Pilot first:* de-risks the metrics tooling on a small set before heavy
   labeling; the roadmap's 30–50 is the target, reached once the harness works.
4. *Unanswerable items:* the pipeline deliberately refuses when context lacks the
   answer (D2); without negatives the eval never exercises that path, and
   faithfulness/precision would be measured only on the easy case.

**Tradeoffs**
- **Pilot size (14) is below the roadmap's 30–50.** Intentional and temporary —
  baseline numbers from 14 items are indicative, not final; the set must grow
  before the Phase 5 matrix conclusions lean on it. Flagged so the small n isn't
  mistaken for the finished dataset.
- **Span matching is containment-based.** A chunk counts as relevant if it
  contains a gold span; very long spans split across two chunks by a given chunker
  could match neither. Mitigated by keeping spans short and self-contained; the
  metric code will need a documented rule for multi-chunk spans.
- **LLM-authored questions can carry subtle bias** toward what the model finds
  salient. Human review is the check; a larger, more diverse expansion later
  reduces it further.
- **Answers are terse.** Fine for LLM-as-judge (tolerates phrasing), but not a
  reference for exact-match scoring — which we are not using.

**Verification**
All 14 items validated: 14 gold spans present in the source (whitespace-normalized,
0 missing), and both unanswerable items confirmed to carry no spans. User reviewed
and approved the set on 2026-06-28.

---

## D4 — Chunker fix: preserve word boundaries across overlap seams

- **Date:** 2026-06-28
- **Status:** Accepted. Fixes a bug in the D1 overlap implementation; this is now the true Phase 1 chunking baseline.
- **ROADMAP reference:** Phase 1 — chunker ([ROADMAP.md](ROADMAP.md) line 20). Surfaced while building Phase 2 Step 2 (retrieval metrics).
- **Type:** Bug fix (deviation from D1's original `_add_overlap`).
- **Implemented in:** [app/chunker.py](app/chunker.py) — `_add_overlap`. Regression check: every chunk must be a whitespace-normalized substring of the source document.

**Context**
The Step 2 span-based relevance mapping flagged q01 as having zero relevant chunks
despite its gold span validating against the full document. Tracing it revealed
that **33 of 34 chunks** were corrupted at their overlap seam: base chunks are
`.strip()`-ed (D1's global overlap pass), which removes the whitespace that
separated a chunk's end from the next chunk's start. When `_add_overlap`
re-concatenated the overlap seed with the next chunk, the two adjacent words
collided — e.g. `"...domesticated in"` + `"early Neolithic..."` → `"inearly"`.

**Decision**
Join the overlap seed and the next chunk with a single space:
`(seed + " " + curr).strip()`.

**Why**
The lost boundary was always a whitespace separator (paragraph / line / sentence /
word — all whitespace), so reinserting one space faithfully reconstructs the word
boundary after whitespace-normalization. This is a **correctness fix, not tuning**:
`"inearly"` is genuine corruption of the stored chunk text that degrades its
embedding and any answer generated from it — the eval failure was just the symptom
that exposed it. Fixing the data is strictly better than tolerating the corruption
and working around it in the metric.

**Tradeoffs**
- **Char-level boundaries.** If a base boundary ever fell mid-word (the `""`
  terminal separator, only reachable for 435 tokens of unbroken non-whitespace),
  the inserted space would be wrong. Negligible in prose; no occurrences here.
- **Re-baseline.** Required re-ingesting the corpus; embeddings for seam regions
  changed slightly. Since no official baseline had been recorded yet (this was the
  first eval run, and it was flagged buggy), the timing is ideal — the corrected
  chunker is what all Phase 2+ numbers are measured against. Chunk count unchanged (34).

**Verification**
After the fix: 0/34 chunks seam-corrupted (all are clean normalized substrings of
the source), 0 chunks over the 512-token budget, q01 maps to a relevant chunk at
rank 1, and the retrieval-metrics zero-relevant warning cleared.

---

## D5 — Generation eval: RAGAS in an isolated venv, pinned to LangChain 0.3.x

- **Date:** 2026-07-04
- **Status:** Accepted.
- **ROADMAP reference:** Phase 2 — Evaluation harness → "wire up RAGAS/DeepEval for the four core metrics (faithfulness, answer relevance, context precision, context recall)" ([ROADMAP.md](ROADMAP.md) line 26). Related: Phase 6 (containerize services, [ROADMAP.md](ROADMAP.md) line 50) — a key reason to isolate.
- **Type:** Decision (framework + environment) with a forced version pin (deviation from "just install the latest").
- **Implemented in:** [requirements-eval.txt](requirements-eval.txt) (separate `.venv-eval`), [eval/generation_metrics.py](eval/generation_metrics.py).

**Context**
Phase 2 needs LLM-as-judge metrics. RAGAS was chosen as the framework (canonical
RAG-eval library, names these four metrics 1:1). Two problems surfaced on install:
(1) RAGAS pulls ~60 transitive packages — the entire LangChain + LangGraph stack,
pandas, pyarrow, scipy, huggingface_hub — and would downgrade `jiter`/`websockets`
in the app venv; (2) RAGAS 0.4.3 declares **unpinned** langchain deps, so pip
grabbed LangChain 1.x, whose `langchain_community` removed a module RAGAS 0.4.3
imports at load time (`chat_models.vertexai`) — RAGAS failed to import at all.

**Decision**
1. **Isolate RAGAS in a separate virtualenv** (`.venv-eval`, from
   `requirements-eval.txt = -r requirements.txt + ragas + pinned langchain`). The
   app's `.venv` and its tested runtime are untouched.
2. **Pin the LangChain 0.3.x line** RAGAS 0.4.3 actually works against
   (`langchain==0.3.30`, `langchain-core==0.3.86`, `langchain-community==0.3.31`,
   `langchain-openai==0.3.35`, `langchain-text-splitters==0.3.11`), which keeps
   `openai` at 2.x (no SDK downgrade). Removed orphaned langgraph/langchain-classic
   packages left by the initial 1.x resolution; `pip check` is clean.
3. **Judge model: gpt-4o-mini, temperature 0** (D2 logic — cheap, repeatable).
4. **Unanswerable questions scored by a separate refusal-accuracy check**, not
   RAGAS (faithfulness/relevancy are ill-defined for an "I don't know" response).

**Why**
- *Isolation:* eval is a dev/CI-time concern never touched by `/query`. Keeping its
  heavy, conflict-prone stack out of the app venv preserves the app's lean, tested
  runtime and — critically for Phase 6 — keeps the API container image small
  (no LangChain/pandas/pyarrow shipped to production). The eval venv is a *superset*
  of the app venv (app deps + RAGAS), so eval code can still `import app.*`.
- *Version pin:* RAGAS's loose deps are a reproducibility hazard — a fresh install
  silently grabs an incompatible LangChain and breaks. Pinning the known-good set
  makes `requirements-eval.txt` reproducible.

**Tradeoffs**
- **Two environments to maintain** (`.venv`, `.venv-eval`) and two requirements
  files. Documented; the eval venv is only needed to run the eval harness.
- **Pinned to older LangChain.** RAGAS 0.4.3 lags the current LangChain 1.x; we're
  frozen on 0.3.x until RAGAS supports 1.x. Acceptable — eval is offline and the
  pins are explicit.
- **RAGAS deprecation warnings.** 0.4.3 warns that metric imports move to
  `ragas.metrics.collections` in v1.0; suppressed for now, revisit on upgrade.
- **answer_relevancy is noisy for terse text** (RAGAS generates questions from the
  answer; short answers score erratically) and gpt-4o-mini sometimes ignores its
  n-generations request. Interpret that metric with more caution than the others.

**Verification**
RAGAS imports and runs end-to-end on our stack (`openai` 2.38.0 retained,
`pip check` clean). Baseline on the 14-item set: faithfulness 0.833, answer
relevancy 0.862, context precision 0.893, context recall 0.917, refusal accuracy
1.000 (2/2). RAGAS context_recall (0.917) tracks the independent span-based
Recall@5 (0.875), cross-validating the retrieval signal.

---

## D6 — Phase 3 architecture: Protocol interfaces, table-per-config storage, Voyage

- **Date:** 2026-07-08
- **Status:** Accepted; implementation spans Phase 3 (Steps 1–5).
- **ROADMAP reference:** Phase 3 — Swappable interfaces → "Refactor chunking and embedding so strategy and model are selected by configuration... Define a common interface for each" ([ROADMAP.md](ROADMAP.md) lines 30–34). Enables Phase 5's 3×2 matrix ([ROADMAP.md](ROADMAP.md) lines 60–67).
- **Type:** Decision (architecture; roadmap mandates swappability but not the mechanism).
- **Implemented in (planned):** `app/interfaces.py` (Protocols), `app/pipeline.py` (RunConfig + registry + factory), refactored `app/chunker.py` / `app/embedder.py`; Step 2 makes `app/store.py` / `app/retriever.py` / the eval harness config-aware.

**Context**
Phase 5 needs to sweep chunking {Recursive-512, Recursive-256, Semantic} × embedding
{text-embedding-3-small, a contender} = 6 cells by config alone. Three design
questions had to be settled first: how components are abstracted, how vectors of
differing dimensionality/chunking are stored, and which second embedder to add.

**Decision**
1. **Interfaces via `typing.Protocol` + a registry + a factory.** `Chunker` and
   `Embedder` are structural Protocols; a registry maps config names to classes; a
   `RunConfig` builds the pipeline's components. Recursive-512 vs -256 are just
   constructor params, not separate strategies.
2. **Table-per-config storage.** Each config gets its own table `chunks_<config_id>`
   (config_id = stable hash of chunking+embedding params) with its own
   `vector(dim)`. All cells coexist; no re-ingest between evaluations.
3. **Second embedder: Voyage `voyage-3-large`** (~1024-dim), retrieval-tuned for
   technical/scientific text like the corpus. Needs a `VOYAGE_API_KEY` + `voyageai`
   client.

**Why**
1. *Protocol + registry:* structural typing keeps implementations decoupled (no
   inheritance), the registry turns "add a strategy" into "add one entry," and the
   factory is what lets Phase 5 iterate configs with zero pipeline edits — the exit
   criterion.
2. *Table-per-config (chosen over sequential single-store):* different embedders
   emit different dims (3-small 1536, Voyage 1024), so a single fixed
   `vector(1536)` column can't hold them; and different chunkings produce different
   rows. Per-config tables let all six cells live simultaneously, so the matrix
   doesn't re-ingest between evals. Cost: more schema/state machinery, and the read
   path (retriever, eval) must target a config's table and embed queries with that
   config's embedder.
3. *Voyage:* the roadmap's suggested quality contender for technical corpora; a true
   cross-provider comparison (vs a same-provider size bump like 3-large).

**Tradeoffs**
- **More DB state.** N coexisting tables to create/track; needs a clear naming +
  lifecycle scheme and a way to list/drop configs.
- **Query-embedder coupling.** Retrieval MUST embed the query with the same embedder
  that populated the table — a mismatch yields meaningless distances. The config has
  to thread through the whole read path (retriever, eval harness, and the API's
  notion of an "active" config).
- **Third-party dependency + key.** Voyage adds an external provider, API key, and
  client library; a cell can't run without the key.
- **Schema no longer purely in db/init.** Tables are created per-config at ingest
  time (dim known only from config), so schema creation moves partly into app code.

**Verification (per step) — Phase 3 COMPLETE (2026-07-17)**
- Step 1: interfaces + factory, **no behavior change** (baseline reproduces identical chunks/dims/IR).
- Step 2: table-per-config storage + `configs` registry; baseline re-ingested to `chunks_33fd8fca2f`, static `chunks` dropped; IR byte-identical.
- Step 3: `SemanticChunker` added + registered; ingested + retrieval-evaluated as its own coexisting config.
- Step 4: `VoyageEmbedder` (voyage-3-large, 1024-dim) added + registered; live smoke test passed; no dependency conflicts in either venv.
- Step 5 (exit criterion): `recursive512__voyage-3-large` ingested and **fully evaluated end-to-end (retrieval + generation) by config change alone**, zero pipeline edits. Three configs now coexist at different dims (1536/1536/1024). Baseline still reproduces exactly (P@1 0.917 / R@5 0.875 / MRR 0.917).

**Exit criterion met:** chunking strategy and embedding model are both swappable via a `RunConfig` with no pipeline code changes. Early (non-authoritative) data point — recursive512 × voyage-3-large scored R@5 0.917 vs the baseline's 0.875; the systematic comparison is Phase 5.

---

## D7 — Phase 4 hybrid retrieval: rank_bm25 + Reciprocal Rank Fusion (RBAC pending)

- **Date:** 2026-07-17
- **Status:** Accepted (hybrid retrieval). RBAC decisions **deferred to Step 3** — see the pending note below; this entry will be extended when they're made.
- **ROADMAP reference:** Phase 4 — Hybrid retrieval and access control → "Add BM25 keyword search and fuse it with dense retrieval... Then add the RBAC layer" ([ROADMAP.md](ROADMAP.md) lines 36–40). Related: Phase 5 optional stretch (hybrid vs dense on the winning cell, [ROADMAP.md](ROADMAP.md) line 69).
- **Type:** Decision (engine + fusion algorithm; roadmap mandates BM25 + fusion but not the mechanism).
- **Implemented in (planned):** `app/lexical.py` (BM25 search), `app/retriever.py` (fusion + `retrieval_mode`), `requirements.txt` (rank_bm25).

**Context**
Dense retrieval alone can miss exact terms (names, codes, rare words) because an
embedding blurs them. Phase 4 adds BM25 keyword search and fuses it with dense as a
selectable mode. Two design questions: which BM25 engine, and how to fuse.

**Decision**
1. **BM25 engine: `rank_bm25` (Python), in-memory over a config's chunk texts.**
2. **Fusion: Reciprocal Rank Fusion (RRF)** — `score = Σ 1/(k + rank_i)`, k≈60.
3. **Delivered as a mode**: `retrieval_mode ∈ {dense, hybrid}`, dense stays default.

**Why — BM25 engine (options compared)**

| Option | True BM25? | In-DB? | Infra cost | Verdict |
|---|---|---|---|---|
| **rank_bm25 (Python)** ✅ | yes | no (app-side) | none — pure-Python lib | **Chosen.** Genuine BM25, zero infra churn, fine at our scale (tens of chunks). |
| ParadeDB pg_search | yes | yes | swap Docker image, **destructive** volume re-init, re-ingest all configs | Rejected: heavy infra change + wipe for a tiny corpus; most production-grade but overkill now. |
| Postgres native FTS (ts_rank) | **no** | yes | none (built-in) | Rejected: `ts_rank`/`ts_rank_cd` is cover-density ranking, not BM25 — fails the roadmap's explicit "BM25". |

Trade-off accepted with rank_bm25: the lexical index lives in Python (rebuilt from
the config's table per search), so fusion and later RBAC filtering happen in app
code rather than a single SQL query. Acceptable given the corpus size and the goal
of zero infra churn.

**Why — fusion (options compared)**

| Option | Needs score normalization? | Params | Verdict |
|---|---|---|---|
| **RRF** ✅ | no (rank-based) | just k (~60) | **Chosen.** Cosine distance and BM25 scores are on totally different scales; RRF sidesteps that by combining ranks. Robust, standard default. |
| Weighted score blend (α·dense + (1−α)·bm25) | yes | α weight + normalization | Rejected for now: more tunable (α is itself a knob) but sensitive to score-distribution quirks and needs careful normalization. Could revisit as a Phase 5 knob. |

**Tradeoffs**
- **In-memory lexical index** rebuilt per query — negligible at tens of chunks, but
  not how you'd scale it; a real deployment would use an in-DB BM25 (ParadeDB) or a
  search engine. Documented as a scale-bounded choice.
- **Fusion/RBAC in app code**, not one SQL query — more app logic, but keeps the DB
  image unchanged.
- **New dependency** (rank_bm25) in both venvs.

**RBAC — DECIDED (Phase 4 Step 3, 2026-07-17)**
- **Implemented in:** [app/rbac.py](app/rbac.py) (`User`, `acl_condition`, `where_clause`), [app/store.py](app/store.py) (`allowed_roles` stamped into metadata), [app/lexical.py](app/lexical.py) + [app/retriever.py](app/retriever.py) (filtered reads), [app/main.py](app/main.py) (`X-User-Roles` header), [tests/test_rbac.py](tests/test_rbac.py) (negative test), [data/confidential.md](data/confidential.md).

1. **ACL model: role-based `allowed_roles`.** Each document's chunks store
   `metadata.allowed_roles` (e.g. `["public"]`, `["admin"]`); a `User` has roles; a
   chunk is visible when the sets intersect (Postgres `jsonb_exists_any`).
   *Chosen over* classification levels (hierarchical but less flexible, and not
   strictly "role-based" as the roadmap specifies) and owner-based ACLs (models
   per-user ownership, a poor fit for "documents a user may see").
2. **Identity via `X-User-Roles` header → `User`.** The API reads roles from a
   header (absent ⇒ `public`) and passes a `User` into `retrieve()`. *Chosen over* a
   field in the request body, because in real systems identity comes from the auth
   layer, not the query payload. Real auth/JWT remains out of scope.
3. **Demo data: a dedicated admin-only document.** [data/confidential.md](data/confidential.md)
   (codename "Project Bluefin") is ingested with `allowed_roles=["admin"]` beside the
   public paper. *Chosen over* tagging a subset of the paper's chunks, which would
   split one document across access tiers — artificial and a murkier test.

**Enforcement design**
- **Default-deny for real users:** an untagged chunk (no `allowed_roles`) is
  invisible to any `User`. `user=None` means an unrestricted/trusted call, which is
  what the eval harness uses — so the harness needed no changes and the baseline
  still reproduces exactly.
- **Filtered at the DB read on BOTH paths.** Dense adds the ACL to its `WHERE`;
  lexical filters the corpus *before* building the BM25 index — so forbidden
  documents don't even influence lexical statistics (IDF, average length). No leak
  through scores, not just through results.
- **Negative test is self-cleaning:** it ingests the confidential doc, asserts, then
  deletes it, leaving the eval table paper-only.

**RBAC tradeoffs**
- No real authentication — a caller can claim any role via the header. Acceptable:
  the roadmap scopes Phase 4 to *retrieval filtering*, and an auth gateway would
  supply verified identity in production.
- Document-level (not passage-level) ACLs; every chunk of a document shares its ACL.
- `user=None` bypassing the filter is a deliberate trusted-path escape hatch; it must
  never be reachable from an HTTP request (the API always constructs a `User`).

**Verification — Phase 4 exit criterion MET (2026-07-17)**
Hybrid retrieval available as a mode, and access control enforced + proven:
`tests/test_rbac.py` passes — on **both** dense and hybrid, the public user is denied
the confidential passage while an **admin positive control** retrieves it (proving the
denial is access control, not a broken query), and the generator refuses ("I don't
know based on the provided context.") instead of leaking. Baseline IR unchanged
(P@1 0.917 / R@5 0.875 / MRR 0.917). Measured hybrid-vs-dense gain on the same table:
R@5 0.875 → 0.958, MRR 0.917 → 0.938.

---

## D8 — Eval dataset expanded to 35 items, with engineered difficulty and an exact-term category

- **Date:** 2026-08-25
- **Status:** Accepted; user reviewed and approved all 21 new items.
- **ROADMAP reference:** Phase 2 — "30–50 questions with ground-truth answers and labeled relevant chunks" ([ROADMAP.md](ROADMAP.md) line 26), fulfilled late, as [D3](#d3--evaluation-dataset-span-based-gold-labels-llm-drafted--human-reviewed) said it must be before Phase 5 conclusions relied on it. Directly serves Phase 5 ([ROADMAP.md](ROADMAP.md) lines 42–46).
- **Type:** Decision (benchmark design) + completion of a deferred obligation.
- **Implemented in:** [eval/dataset.json](eval/dataset.json) (35 items).

**Context**
D3 built a 14-item pilot and explicitly flagged that it must grow before the matrix
leaned on it. Reviewing the pilot before Phase 5 surfaced a second, sharper problem:
a **ceiling effect**. 11 of 12 answerable questions put a relevant chunk at rank 1
(MRR 0.917). A benchmark nearly everything passes cannot *discriminate* — and
discrimination between six configs is the entire job of the experiment matrix.

**Decision**
1. **Expand to 35 items** (30 answerable + 5 unanswerable, 14% negatives) — inside
   the roadmap's 30–50 band. Authored LLM-drafted → human-reviewed, same as D3.
2. **Engineer a difficulty spread** (14 easy / 13 medium / 8 hard) rather than
   accepting whatever difficulty naturally arose.
3. **Add an `exact-term` category** (9 items: specific figures, dates, proper names,
   Latin binomials, a genus name) alongside `factual` and `multi-hop`, to exercise
   lexical matching and make the dense-vs-hybrid comparison measurable.
4. **Recategorize q04** (110 accessions) `factual` → `exact-term`; it turns on a
   specific figure and was the pilot's one dense-retrieval miss. Wording and spans
   unchanged.

**Why**
- *Engineered difficulty:* a dataset where all six matrix cells score ~0.92 yields no
  defensible winner. Deliberately including items current retrieval plausibly fails
  is what gives the matrix statistical room to separate configs.
- *Exact-term category:* Phase 4 added hybrid retrieval, but the pilot barely tested
  it. Exact-term questions are where BM25 beats dense, so without them the hybrid
  gain (and the Phase 5 stretch goal) would be invisible.
- *Chosen over* a "natural mix" of realistic user questions, which risked repeating
  the ceiling effect.

**Tradeoffs**
- **Engineered ≠ natural.** The set is a *benchmark*, tuned to discriminate, not a
  faithful sample of real user queries. Documented deliberately: absolute scores on
  it are not a claim about real-world experience, only a basis for comparing configs.
- **Difficulty labels are intent, not measurement.** They record what was *aimed at*.
  Reality partly disagrees: dense missed `q26` at top-5 despite an "easy" label,
  while some "hard" items were retrieved at rank 1. Treat the field as a design aid.
- **The 14-item baseline is invalidated** — old numbers are not comparable. Expected:
  the baseline cell is re-measured as part of the Phase 5 matrix run.
- Corpus is still a single document, so coverage breadth is bounded by that paper.

**Verification**
All 35 items validated programmatically: every gold span occurs verbatim in the
source (whitespace-normalized), all unanswerable items carry zero spans, ids unique,
no answerable question maps to zero relevant chunks. The design goal was met
measurably — see [FINDINGS.md](FINDINGS.md) F2.

---

## D9 — Phase 5 scoring rule: 60/40 retrieval-generation, faithfulness-weighted, un-normalized

- **Date:** 2026-08-27
- **Status:** Accepted; weights chosen by the user and fixed **before** the composite was computed.
- **ROADMAP reference:** Phase 5 — the experiment matrix and its exit criterion, "a completed results table with a defensible, data-backed choice of configuration" ([ROADMAP.md](ROADMAP.md) lines 42–46). Also serves the standing constraint "optimize for quality per dollar and per second, not raw quality."
- **Type:** Decision (evaluation methodology).
- **Implemented in:** [eval/matrix.py](eval/matrix.py) (data source); scoring applied over `eval/results/cells/*.json`.

**Context**
Six configurations were measured on three dimensions (quality, cost, latency).
Picking a winner requires collapsing several quality metrics into one comparable
number, and *every* such collapse embeds a value judgement. The methodological risk
is choosing weights **after** seeing which configuration they favour. The rule was
therefore agreed in Phase 5 scoping and the specific weights fixed before computing
anything.

**Decision**
1. **60% retrieval / 40% generation.** Retrieval = the deterministic IR metrics
   (P@1, Recall@5, MRR), equally weighted at 20% of the total each.
2. **Within generation, faithfulness 70% / answer relevancy 30%** — i.e. 28% and 12%
   of the total.
3. **`context_precision` and `context_recall` are excluded from the score** and
   reported instead as an independent cross-check of the retrieval bucket.
4. **`refusal_accuracy` is excluded** from the score.
5. **No normalization.** Metrics enter the weighted average as raw 0–1 values; no
   min-max rescaling or z-scoring across the six cells.
6. **Cost and latency stay outside the composite**, reported as explicit
   quality-per-dollar and quality-per-second ratios, with one-time ingest cost kept
   separate from recurring per-query cost.

**Why**
- *Retrieval weighted higher:* it is **deterministic** — zero LLM-judge noise, so the
  same config always yields the same number — and it is where the configurable
  machinery actually lives (chunker, embedder, K, retrieval mode). Generation
  metrics inherit judge variance, and the generation model and prompt are held
  constant across all six cells, so that half of the score has less to say about the
  variables under test.
- *Faithfulness privileged:* recorded honestly as a **value judgement, not a
  statistical result.** The expectation was that answer relevancy would prove
  noisier and could be downweighted on those grounds. **Measurement contradicted
  it** — mean run-to-run spread was 0.0081 for answer relevancy vs 0.0099 for
  faithfulness, and answer relevancy discriminated better (signal/noise 14.0× vs
  7.3×). The weighting stands on a different basis: faithfulness measures what this
  system *promises* — answers anchored to retrieved context — and hallucination is
  the failure mode the project owner most wants suppressed. It is also the metric the
  grounding prompt exists to enforce, and the one that matters most alongside RBAC,
  where a fabricating system can assert content it never retrieved.
- *Excluding `context_*` from the score:* both metrics judge **retrieval**, not
  generation — [CONCEPTS.md](CONCEPTS.md) calls them the LLM-judged cousins of
  Precision@K and Recall@K. Scoring them inside the generation bucket would have
  given retrieval roughly **80% effective weight** while the stated rule said 60%,
  silently violating the agreed split. Reporting them as a cross-check preserves the
  "two independent lenses" value they were adopted for.
- *Excluding `refusal_accuracy`:* it is **1.000 in all six cells**. A metric constant
  across every candidate contributes an identical term to every score and cannot
  change a ranking, whatever weight it carries. It is reported as a floor that all
  configurations met, not as a selection criterion.
- *No normalization, chosen over min-max/z-score:* a normalized score is defined
  **relative to the pool of six** — adding a seventh configuration would change every
  existing cell's score, and the number stops being interpretable as "quality" in any
  absolute sense. The raw weighted average stays interpretable and stable as the
  matrix grows.

**Tradeoffs**
- **Weights do not equal influence.** In an un-normalized average, a metric's effect
  on the ranking is *weight × spread across cells*. Faithfulness carries the largest
  single weight (28%) yet moves the composite less than P@1 does at 20%, because the
  cells separate by only 0.072 on faithfulness against 0.333 on P@1:

  | metric | weight | range across cells | effective influence |
  |---|---|---|---|
  | P@1 | 0.20 | 0.333 | **0.067** |
  | MRR | 0.20 | 0.225 | 0.045 |
  | Recall@5 | 0.20 | 0.117 | 0.023 |
  | faithfulness | 0.28 | 0.072 | 0.020 |
  | answer relevancy | 0.12 | 0.114 | 0.014 |

  So the composite is dominated by retrieval by more than the nominal 60/40 — a
  direct consequence of declining to normalize. This is accepted and disclosed rather
  than corrected, because the alternative costs interpretability (above).
- **Equal weights within the retrieval bucket are a simplification.** P@1 and MRR are
  strongly correlated (both reward rank-1 hits), so rank-1 accuracy is effectively
  counted twice against Recall@5's coverage view.
- **The composite is a ranking aid, not a measurement.** Only the underlying metrics
  are measurements; the score is a declared preference applied to them. The full
  per-metric table is reported alongside it so a reader who rejects these weights can
  re-rank from the raw numbers.

**Verification**
A sensitivity sweep across 15 weightings (retrieval 40–80%, faithfulness share
50–90%) returned **`r256-voyage` as the winner in 14 of 15**. The single exception —
retrieval 40% *and* no faithfulness privilege — flipped to `r512-voyage` by 0.0017,
well inside judge noise. The conclusion therefore does not depend on the specific
weights chosen here, which is the strongest available defence against the charge that
the rule was fitted to a preferred answer.

---

## D10 — Hybrid retrieval adopted as the production default

- **Date:** 2026-08-27
- **Status:** Accepted.
- **ROADMAP reference:** Phase 5 stretch goal — hybrid vs dense on the winning configuration ([ROADMAP.md](ROADMAP.md) lines 42–46). Also completes the Phase 4 hybrid work ([D7](#d7--phase-4-hybrid-retrieval-rank_bm25--reciprocal-rank-fusion-rbac-pending)), which was built and measured but never promoted.
- **Type:** Decision (configuration), backed by measurement.
- **Implemented in:** [app/pipeline.py](app/pipeline.py) (`PRODUCTION.retrieval_mode = "hybrid"`), served via [app/main.py](app/main.py); evidence in `eval/results/cells/cell_r256-voyage-hybrid.json`.

**Context**
D7 built hybrid retrieval (BM25 + RRF) and F2 measured a large gain — **+0.111 MRR**
— but only against the Phase 1 baseline. Phase 5 then replaced the baseline with a
much stronger dense retriever (`recursive-256 × voyage-4-large`, P@1 0.900). The open
question was whether hybrid still earns its place once dense retrieval is good, or
whether its value was an artefact of a weak baseline. F2's own ceiling-effect
mechanism predicted the gain should shrink.

**Decision**
Serve `retrieval_mode="hybrid"` in `PRODUCTION`. `BASELINE` is untouched and remains
dense, preserving it as the fixed comparison point.

**Why**
- **It wins on every metric measured — nine for nine.** P@1 0.900→0.933, Recall@5
  0.967→**1.000**, MRR 0.925→0.957, faithfulness 0.901→0.931, answer relevancy
  0.841→0.871, context precision 0.929→0.945, context recall 0.967→**1.000**, at
  equal cost and equal latency. There is no metric on which dense is preferable.
- **The generation gains clear the noise floor.** Faithfulness +0.031 and answer
  relevancy +0.030 against run-to-run spreads of 0.003–0.006 — a 5–10× signal-to-noise
  ratio. Context recall is deterministic across all six runs (spread 0.0000) and
  reaches exactly 1.000.
- **Faithfulness improved with no change to the generator or prompt.** Only the
  retrieved context changed, which isolates the causal path: better context produces
  better-grounded answers. It is also the metric weighted highest under D9.
- **Recall@5 = 1.000 changes what failures are possible.** A relevant chunk now
  reaches the top-5 for all 30 answerable questions, so no remaining error can be
  attributed to missing evidence.
- **It is nearly free.** +$0.000003 per query (+1.4%), no additional API calls, and no
  re-ingest — hybrid reads the same table its dense twin populated, because D6
  deliberately excluded `retrieval_mode` from `config_id`. Measured end-to-end latency
  did not increase (1.090s → 1.058s).
- *Chosen over* keeping dense for simplicity: the only real cost is the BM25 code
  path, which already exists, is already RBAC-enforced, and is already tested.

**Tradeoffs**
- **The improvement is not statistically significant.** McNemar on paired hit@1 gives
  *p* = 1.000 (one discordant question); the bootstrap 95% CI on ΔMRR is
  [+0.000, +0.073]. The defensible claim is that hybrid **dominates on every metric
  and never underperforms**, not that its advantage is proven. Adoption rests on the
  asymmetry — nine-for-nine at near-zero cost — rather than on a p-value. See
  [FINDINGS.md](FINDINGS.md) F6 and F7.
- **The gain is much smaller than F2 reported** (+0.032 MRR vs +0.111): 29% of the
  original effect. This is the predicted ceiling effect, not a contradiction — see F7.
- **BM25 rebuilds its index per query** from the RBAC-filtered corpus. Fine at 63
  chunks; it is the first thing that would need caching at a larger corpus size, and
  `search` latency already doubled (0.025s → 0.051s) even though the absolute cost
  stayed negligible.
- **Two retrieval paths to maintain** — both must stay RBAC-correct. Mitigated by the
  negative test, which exercises dense and hybrid across both configs.

**Verification**
`python -m tests.test_rbac` passes for `BASELINE` and `PRODUCTION`, dense and hybrid —
10 assertions, including the generator refusing to leak to a public user. The live API
was queried over HTTP under the new config and returned a grounded answer with sources.

---

## D11 — CI runs unit tests only; evaluation is manual and reports without failing

- **Date:** 2026-08-27
- **Status:** Accepted; the user chose both halves explicitly after being shown the trade-off.
- **ROADMAP reference:** Phase 6 — "Wire the evaluation suite into GitHub Actions so a regression fails CI", exit criterion "CI runs tests plus evaluation on every push" ([ROADMAP.md](ROADMAP.md) lines 48–52).
- **Type:** **Deviation** from the roadmap's stated exit criterion.
- **Implemented in:** [.github/workflows/ci.yml](.github/workflows/ci.yml), [.github/workflows/evaluation.yml](.github/workflows/evaluation.yml), [tests/test_units.py](tests/test_units.py), [ruff.toml](ruff.toml).

**Context**
Taken literally, the roadmap asks for the full evaluation suite on every push, failing
the build on regression. Three facts make that a bad gate in practice:

1. **Cost and flakiness.** One generation-metrics run is ~350 OpenAI requests. During
   Phase 5 the daily quota was exhausted **twice**, killing runs mid-flight. A gate
   that goes red because of a rate limit rather than a code change teaches everyone to
   ignore the gate — which is worse than not having one.
2. **No zero-cost evaluation tier exists.** Even the "cheap, deterministic" IR metrics
   embed 35 queries, so they need a paid API key and a live Postgres. There is no
   evaluation subset that runs for free.
3. **Repository secrets are unavailable to pull requests from forks**, by GitHub's
   design. Any API-dependent job is structurally red on outside contributions.

**Decision**
1. **`ci.yml` on every push and PR:** ruff lint, 36 unit tests, and a Docker build of
   both images. No API key, no database, no network calls to a paid provider.
2. **`evaluation.yml` on `workflow_dispatch` + a weekly schedule:** spins up Postgres,
   ingests, and runs the metrics **against `--config production`** — the configuration
   the API actually serves, not `BASELINE`.
3. **The evaluation workflow reports; it never fails the build.** Results go to the
   GitHub run summary and are uploaded as an artifact.
4. **A real unit-test suite was written to make (1) possible** — none existed before,
   since `tests/test_rbac.py` needs live API and DB access.

**Why**
- *Splitting the gates:* the push gate should answer "did this change break the code?"
  — a question answerable deterministically and for free. "Did this change degrade
  retrieval quality?" is a different question with a different cadence and a real
  bill; conflating them makes the fast gate slow and the slow gate noisy.
- *Reporting rather than failing:* a numeric threshold was considered and rejected.
  With 35 questions, differences between good configurations are **not statistically
  separable** ([FINDINGS.md](FINDINGS.md) F6) — McNemar *p* = 1.000 between the winner
  and its runner-up. A floor set inside that noise band would fail on sampling
  variation, and a floor set outside it would not catch anything a human wouldn't
  already notice. Reporting the numbers where a human reads them is the honest option
  until the dataset is large enough for a threshold to mean something.
- *Testing `production`, not `baseline`:* watching the wrong configuration is worse
  than watching nothing, because it produces the appearance of coverage.
- *Committing [ruff.toml](ruff.toml):* without a checked-in lint config, ruff applies
  whatever its installed version defaults to, so CI and local disagree and a version
  bump silently changes what fails. Found while wiring this up.

**Tradeoffs**
- **The roadmap's exit criterion is not met as written.** CI does not run evaluation on
  every push, and no regression fails the build. A quality regression reaches `main`
  and is caught at the next manual or weekly run. This is the cost of the decision and
  is stated rather than papered over.
- **The weekly schedule spends money unattended.** Modest (~$0.02/run for retrieval
  only), but it is a recurring charge with no human watching.
- **Unit tests cover logic, not behaviour.** They cannot catch a regression that only
  shows up as worse retrieval — which is precisely the failure mode this project
  exists to measure. The suite guards the two places bugs actually appeared (chunker
  seams, `config_id`), not answer quality.

**Verification**
`ruff check app eval tests` clean; `pytest tests/test_units.py` — 36 passed in 0.51s
with no API key and no database. `docker compose up -d` brings up db + api + web, all
healthy; a query issued through the frontend's nginx proxy returned a grounded answer,
and the role switch was confirmed end-to-end (public refused, admin answered).

**Side effect worth knowing**
Wiring the lint config surfaced six `zip()` calls without `strict=`. Two were latent
correctness risks, not style: `store.py` zips chunks with their vectors (a length
mismatch would silently drop chunks), and the three `cosine()` helpers zip two vectors
(a 1536-vs-1024 mismatch would score over the shorter prefix instead of erroring).
All are now `strict=True`; the genuinely-unequal pairwise loops use `itertools.pairwise`.
