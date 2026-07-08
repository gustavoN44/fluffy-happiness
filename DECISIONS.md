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
