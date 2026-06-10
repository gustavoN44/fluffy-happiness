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
