# Findings

Empirical observations produced by measuring the system. This is the third
companion doc: [DECISIONS.md](DECISIONS.md) records *choices we made and why*,
[CONCEPTS.md](CONCEPTS.md) explains *how components work*, and this file records
*what the data actually showed* — including results that were surprising or that
changed how we think about a trade-off.

Each finding states its evidence and, importantly, **how much weight it can bear**.
A one-query observation and a 35-question matrix result are not the same kind of
claim, and entries say which they are.

---

## F1 — Generation dominates per-query cost; the embedding model barely registers

- **Date:** 2026-08-25
- **Phase:** 5, Step 1 (cost + latency instrumentation)
- **Evidence:** one metered query ("Where was cannabis first domesticated?") across three configs, via [app/metering.py](app/metering.py). Token counts are **real API usage**, not estimates; costs derive from the price table (currently UNVERIFIED — see caveat).
- **Confidence:** the headline is **structural and robust**; the secondary observation is **tentative (single query)**.

### What we measured

Per-query token breakdown for `recursive512 × text-embedding-3-small` (dense):

| source | tokens | share |
|---|---|---|
| query embedding (`text-embedding-3-small`, input) | **7** | 0.3% |
| generation prompt (`gpt-4o-mini`, input) | **2,317** | 99.0% |
| generation output (`gpt-4o-mini`, output) | **17** | 0.7% |
| **total** | **2,341** | |

Translated to cost, the imbalance is even starker, because embedding tokens are
also cheaper per token than generation tokens:

| config | total cost | embedding's share of cost |
|---|---|---|
| recursive512 × 3-small (dense) | $0.000358 | **~0.04%** |
| recursive512 × voyage-3-large (dense) | $0.000249 | **~0.4%** |

Even Voyage — roughly 9× pricier per token than 3-small — accounts for **under half
a percent** of the cost of answering one question.

### Why this matters

The naive expectation going into the experiment matrix is that the "cheap vs
premium embedding model" axis is a **cost** trade-off: pick the expensive embedder,
pay more per query. The data says that framing is wrong for the *recurring* cost.

- **At query time, the embedder's price is nearly irrelevant.** It embeds one short
  question (~7 tokens). Choosing between embedders should therefore be argued on
  **retrieval quality and latency**, essentially not on per-query cost.
- **What actually drives per-query cost is context size** — how many tokens the
  retrieved chunks add up to, since they become the generation prompt. That is a
  function of **chunk size and how many chunks we retrieve (top-K)**, not of which
  embedding model produced the vectors.
- **Ingest is the opposite regime.** A one-time ingest of the corpus embedded
  14,204 tokens ($0.000284 at 3-small). There the embedder's price *is* the whole
  cost, and it scales with corpus size and chunker overlap. So the embedding model's
  price matters for **ingest economics**, not for **query economics** — two
  different budgets that the matrix should report separately.

**Consequence for the matrix:** cost per query is expected to track the *chunking*
axis (chunk size → context size) far more than the *embedding* axis. If that holds
across 35 questions, the "quality per dollar" argument for choosing an embedder
mostly collapses into "quality per second" plus one-time ingest cost.

### A tentative, related observation (NOT yet a claim)

On this single query the Voyage cell was **cheaper overall** ($0.000249 vs
$0.000358) despite the pricier embedder — because its top-5 chunks totalled 1,587
generation-input tokens versus 2,317 for 3-small. Both cells use the *same*
`recursive-512` chunker and therefore the same chunk table, so this is not a
chunking difference: the two embedders simply **ranked different chunks into the
top-5**, and chunk lengths vary (not every chunk is a full 512 tokens).

This is **one query**. It could easily be sampling noise rather than a systematic
effect. It is recorded here because, if the matrix confirms it across ~35
questions, it is a genuinely counterintuitive result — *the more expensive embedder
can be cheaper end-to-end* — and worth the README. Until then it is a hypothesis to
test, not a finding to cite.

### Caveats

- **Prices VERIFIED 2026-08-25** against the OpenAI and Voyage pricing pages; all
  four originally-estimated values were correct, so the figures above stand
  unchanged. Re-check before publication — prices change. Token counts are stored
  raw, so re-pricing never requires a re-run.
- Measured with top-K = 5 and `gpt-4o-mini` generation. A much larger K, or a
  pricier generation model, would shift the absolute numbers — but both would push
  *further* in the same direction (generation's share grows).

---

## F2 — An easy benchmark hid a large hybrid-retrieval gain

- **Date:** 2026-08-25
- **Phase:** 5, Step 2 (dataset expansion)
- **Evidence:** deterministic IR metrics (zero LLM-judge noise) on the same corpus,
  same configs, comparing the 14-item pilot against the 35-item expanded set.
- **Confidence:** **high** for the direction and mechanism; the exact magnitudes are
  specific to this corpus and dataset.

### What we measured

Identical pipeline, identical chunks and embeddings — only the *question set* changed:

| dataset | dense P@1 | dense MRR | rank-1 hits | hybrid MRR | **gap** |
|---|---|---|---|---|---|
| 14-item pilot | 0.917 | 0.917 | 11/12 (92%) | 0.938 | **+0.021** |
| 35-item expanded | 0.700 | 0.789 | 21/30 (70%) | 0.900 | **+0.111** |

Recall@5 moved the same way: the dense→hybrid gain grew from +0.083 to +0.100, and
hybrid reached **0.983**.

### The finding

**The pilot's ceiling effect was concealing most of hybrid retrieval's value.** On
the easy set, hybrid looked like a marginal +0.02 MRR improvement — arguably not
worth the extra machinery. On a set with a real difficulty spread, the same
unchanged code shows **+0.111 MRR and +0.133 P@1**: a five-fold larger effect.

The mechanism is not mysterious, which is why confidence is high. When dense
retrieval already ranks the right chunk first on 92% of questions, there is almost
no headroom for a second retriever to contribute — any method scores ~0.92. The
questions that *distinguish* retrieval strategies are precisely the ones an easy
benchmark lacks. Adding exact-term questions (specific figures, dates, proper names)
created that headroom, and BM25 filled it: of the three questions dense retrieval
misses entirely at top-5, two are exact-term items.

### Why it matters

- **For this project:** the Phase 5 matrix would have been close to unusable on the
  pilot. Six configs clustering within noise of each other cannot produce the
  "defensible, data-backed choice" the roadmap requires. The expansion was not
  box-ticking against the 30–50 target; it was a precondition for the phase working
  at all.
- **Generally:** *a benchmark that everything passes measures nothing.* A weak
  measured gain can mean the improvement is weak — or that the benchmark cannot see
  it. Those are very different conclusions, and only inspecting the score
  distribution (here: 92% of questions at rank 1) tells them apart.

### Caveat

The expanded set is deliberately engineered to discriminate (see
[DECISIONS.md](DECISIONS.md) D8), so its *absolute* scores understate real-world
performance on typical questions. Its purpose is comparing configurations, not
estimating user-facing accuracy — and the hybrid gain reported here should be read
as "on questions that separate retrieval strategies," not "on all questions."
