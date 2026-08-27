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

---

## F3 — The embedder decided the outcome, and it is the cheapest thing in the pipeline

- **Date:** 2026-08-27
- **Phase:** 5, Step 4 (matrix analysis)
- **Evidence:** the full 6-cell matrix, 35-item dataset, 3 generation runs per cell
  ([eval/results/cells/](eval/results/cells/)). Retrieval metrics are deterministic.
- **Confidence:** **high** for the direction and size; see F6 for what the sample size
  does and does not license.

### What we measured

Averaging the three cells on each side of the embedding axis — so chunking varies
*within* each average and cancels out:

| metric | text-embedding-3-small | voyage-4-large | delta |
|---|---|---|---|
| P@1 | 0.667 | 0.867 | **+0.200** |
| MRR | 0.772 | 0.912 | **+0.140** |
| Recall@5 | 0.889 | 0.961 | +0.072 |
| faithfulness | 0.851 | 0.902 | +0.051 |
| answer relevancy | 0.799 | 0.857 | +0.058 |

**All three Voyage cells beat all three OpenAI cells on every single metric.** There
is no overlap between the two groups — the worst Voyage configuration outscores the
best OpenAI one.

### The finding, and how it revises F1

F1 established that the embedding model is **~0.2–0.4% of per-query cost**, because
it embeds one short question while generation processes ~2,400 context tokens. That
holds — it was confirmed across all six cells.

But F1 drew a second inference from it: that the embedder choice should therefore be
argued "on retrieval quality and latency, essentially not on per-query cost." That
was correct as far as it went, and **understated the conclusion**. The embedder did
not merely fail to be a cost lever — it turned out to be **the single most decisive
variable in the matrix**, worth roughly 3× more MRR movement than the chunking axis
it was compared against.

So the honest summary is: *the cheapest component to run was the most expensive one
to get wrong.* Its price is nearly irrelevant; its choice is not.

### Why it matters

- **Cost intuitions do not predict quality importance.** The natural way to prioritise
  tuning effort is "where does the money go" — which would have pointed at chunk size
  and top-K, and away from the embedder. That heuristic would have picked the wrong
  variable to optimise first.
- **The one-time/recurring split is what makes the premium embedder cheap.** Voyage
  costs ~6.5× more to ingest ($0.00184 vs $0.00028) but *less* per query, because its
  better ranking pulls tighter contexts. The ingest premium amortises after **~9
  queries**. For any system answering more than a handful of questions, the
  "expensive" embedder is the cheaper one end-to-end — the hypothesis F1 flagged as
  untested on one query, now confirmed across 30.

### Caveats

- Two embedders, one corpus, one domain (a cannabis-genomics paper). This is evidence
  that *the embedder axis dominates the chunking axis here*, not a general ranking of
  embedding models.
- The gap may be partly a **domain-vocabulary effect**: the corpus is dense with Latin
  binomials, accession numbers, and genetics terminology. A newer, larger model
  plausibly has an outsized advantage on exactly that vocabulary.

---

## F4 — There is no best chunk size, only a best chunk-size/embedder pair

- **Date:** 2026-08-27
- **Phase:** 5, Step 4 (matrix analysis)
- **Evidence:** the recursive cells of the matrix — 512 vs 256 tokens, held identical
  in every other respect, under each of the two embedders. Deterministic IR metrics.
- **Confidence:** **high** that the direction reverses; the magnitudes are specific to
  this corpus.

### What we measured

The *same* configuration change — halve the chunk size from 512 to 256 tokens —
applied under each embedder:

| embedder | MRR: 512 → 256 | P@1: 512 → 256 |
|---|---|---|
| text-embedding-3-small | 0.789 → 0.700 = **−0.089** | 0.700 → 0.567 = **−0.133** |
| voyage-4-large | 0.900 → 0.925 = **+0.025** | 0.833 → 0.900 = **+0.067** |

### The finding

**The same change degrades one pipeline and improves the other.** Under 3-small,
256-token chunks are the worst configuration in the entire matrix (P@1 0.567); under
Voyage, they are the best (P@1 0.900). This is a genuine **interaction effect**: the
chunk-size variable has no context-free direction.

The plausible mechanism: smaller chunks trade *evidence completeness* for *ranking
precision*. A 256-token chunk is more likely to split a fact across a boundary, but
when it does contain the fact, it contains less surrounding dilution. Whether that
trade pays depends on how well the embedder handles short, narrow passages — and the
two models clearly differ there.

### Why it matters

- **It invalidates transferable chunk-size advice.** "Use 512" and "smaller chunks
  retrieve better" are both defensible-sounding claims, and each is *empirically
  wrong* for one of the two pipelines we measured. Chunk size cannot be tuned in
  isolation from the embedder, or copied from another project's blog post.
- **It vindicates the roadmap's insistence on two recursive sizes.** With only one
  size in the matrix, this effect is invisible: we would have concluded "256 is worse"
  or "256 is better" — a claim that is half false either way. The extra cells existed
  to separate the size effect from the strategy effect, and they earned their cost.
- **It is an argument against one-variable-at-a-time tuning in general.** The
  roadmap's "change one variable at a time" rule is right for *attribution*, but a
  purely sequential search (fix the best chunk size, then pick the embedder) would
  have locked in 512 under 3-small and never found the winning cell.

### Caveat

Two points per line is enough to establish that the sign differs, not to characterise
the curve. Where the optimum sits for either embedder, and whether the reversal holds
at 128 or 1024 tokens, is unmeasured.

---

## F5 — The winner does not depend on the scoring weights

- **Date:** 2026-08-27
- **Phase:** 5, Step 4 (winner selection)
- **Evidence:** sensitivity sweep over 15 weightings of the D9 composite score.
- **Confidence:** **high** — it is an exhaustive recomputation over the stated
  parameter range, not an estimate.

### What we measured

The D9 rule requires two judgement calls: the retrieval/generation split (set at
60/40) and faithfulness's share of the generation bucket (set at 70/30). Both are
declared preferences, so the obvious objection is that the winner is an artefact of
them. Re-ranking all six cells across the grid:

| retrieval weight | 40% | 50% | 60% | 70% | 80% |
|---|---|---|---|---|---|
| faithfulness 50% | `r512-voyage` | r256-voyage | r256-voyage | r256-voyage | r256-voyage |
| faithfulness 70% | r256-voyage | r256-voyage | **r256-voyage** ← D9 | r256-voyage | r256-voyage |
| faithfulness 90% | r256-voyage | r256-voyage | r256-voyage | r256-voyage | r256-voyage |

**`r256-voyage` wins 14 of 15.** The single exception requires retrieval weighted
*below* generation **and** no faithfulness privilege, and it flips by **0.0017** —
roughly a fifth of the run-to-run judge noise on a single generation metric.

### Why it matters

A composite score is a declared preference applied to measurements, and the standard
critique is that the preference was fitted to a desired answer. The defence is not to
argue the weights are correct — they are a value judgement and D9 says so — but to
show **the conclusion survives rejecting them.** A reader who disagrees with 60/40,
or with privileging faithfulness, still arrives at the same configuration.

This is cheap to run and worth doing whenever a decision rests on chosen weights: it
converts "trust my weighting" into "the weighting doesn't matter here."

### Caveat

Robustness to *weights* is not robustness to *sampling* — a distinct and, here, less
favourable question. See F6.

---

## F6 — The winner is robust to weighting but statistically indistinguishable from the runners-up

- **Date:** 2026-08-27
- **Phase:** 5, Step 4 (winner selection)
- **Evidence:** exact McNemar tests on paired per-question hit@1 outcomes (n = 30
  answerable questions), plus 20,000-resample paired bootstrap CIs on the P@1 and MRR
  differences. Retrieval is deterministic, so all variance here is *sampling*
  variance, not run-to-run noise.
- **Confidence:** **high** — this is a direct calculation on the actual per-question
  results.

### What we measured

| comparison | discordant pairs (win:lose) | McNemar exact *p* | P@1 delta, 95% CI |
|---|---|---|---|
| `r256-voyage` vs `r512-small` (baseline) | 7 : 1 | **0.070** | [+0.033, +0.367] |
| `r256-voyage` vs `r512-voyage` | 2 : 0 | 0.500 | [+0.000, +0.167] |
| `r256-voyage` vs `sem-voyage` | 3 : 2 | 1.000 | [−0.100, +0.167] |

**Not one comparison clears *p* < 0.05.** Even the headline result — a +0.200 P@1
improvement over the Phase 1 baseline, which looks decisive in the results table —
lands at *p* = 0.070. The bootstrap CI for that same comparison excludes zero
([+0.033, +0.367]); the two tests disagree because McNemar conditions on only the 8
discordant questions and is conservative at that count. The honest reading is
"probably real, not demonstrated."

Against the runners-up there is no case at all: `r256-voyage` and `sem-voyage` differ
on 5 of 30 questions, 3 one way and 2 the other. That is a coin flip.

### The finding

**A 30-question benchmark can rank configurations; it cannot certify the ranking.**
The matrix produced a clear, weight-robust ordering (F5) whose top three cells are
statistically inseparable from each other. Both statements are true simultaneously,
and reporting only the first would be misleading.

Power analysis on the observed discordance rates gives the price of certainty:

| comparison | questions needed for 80% power |
|---|---|
| vs baseline (7:1 discordant) | **~60–75** |
| vs `r512-voyage` (2:0) | **~150** |
| vs `sem-voyage` (3:2) | **~1,200** |

The cost scales with how *similar* the configurations are: separating the winner from
the Phase 1 baseline needs roughly double our dataset, separating it from the second-
best Voyage cell needs 40× more.

### Why it matters

- **It bounds what Phase 5 is allowed to claim.** The defensible statement is "the
  Voyage cells clearly outperform the OpenAI cells, and `r256-voyage` is the best
  choice among them on quality, cost, and latency jointly" — not "`r256-voyage` is
  significantly better than `sem-voyage`."
- **The decision is still correct.** Under statistical uncertainty between the top
  three, the tie-breakers are the dimensions with no sampling error at all:
  `r256-voyage` is the cheapest per query and the fastest end-to-end. Choosing it
  costs nothing even if its quality edge is illusory.
- **F3 survives this; F5's ordering partly does not.** The embedder effect is a
  group-level comparison — three cells vs three cells, no overlap on any metric —
  which is a far stronger pattern than any single pairwise gap.

### Caveats

- **The *p*-values assume the questions are a random sample from a population of
  questions. Ours are not** — D8 engineered the dataset to discriminate. Strictly,
  these tests describe the sample rather than support inference to a wider
  population; the power figures should be read as orders of magnitude.
- Statistical significance is about sampling within *this* corpus. Whether any of it
  generalises to another document or domain is external validity, which no number of
  questions from one paper can establish.

---

## F7 — A better retriever hides hybrid's value the same way an easy benchmark did

- **Date:** 2026-08-27
- **Phase:** 5, Step 5 (hybrid vs dense on the winning cell)
- **Evidence:** deterministic IR metrics plus 3 generation runs, dense vs hybrid, on
  the same table (`cell_r256-voyage.json` vs `cell_r256-voyage-hybrid.json`). Only the
  read-time ranking strategy differs — same chunks, same embeddings, same generator.
- **Confidence:** **high** for the shrinkage and its mechanism; **moderate** for the
  hybrid advantage itself, which is consistent but not statistically significant.

### What we measured

The same dense→hybrid switch, applied to a weak and a strong dense retriever:

| dense retriever | dense MRR | hybrid MRR | gain |
|---|---|---|---|
| `r512-small` (Phase 1 baseline) | 0.789 | 0.900 | **+0.111** |
| `r256-voyage` (Phase 5 winner) | 0.925 | 0.957 | **+0.032** |

On the winning cell, hybrid nonetheless improved **every metric measured — nine for
nine**: P@1 0.900→0.933, Recall@5 0.967→**1.000**, MRR 0.925→0.957, faithfulness
0.901→0.931, answer relevancy 0.841→0.871, context precision 0.929→0.945, context
recall 0.967→**1.000**, at +1.4% cost and no latency increase.

### The finding

**F2's mechanism runs in both directions.** F2 showed an *easy benchmark* concealed
hybrid's value by leaving no headroom. The symmetric result is that a *strong dense
retriever* conceals it the same way, for the identical reason: an improvement can only
materialise in the space the baseline leaves unoccupied. Hybrid's gain fell to **29%**
of its former size not because hybrid got worse, but because there were only three
questions left to fix — and it fixed all three.

This makes "hybrid retrieval is worth +0.11 MRR" a meaningless claim in isolation. The
honest form is always *relative to a stated baseline*, and the same technique will be
measured as large or negligible depending entirely on what it is layered on.

### A second-order effect worth separating out

Faithfulness rose **+0.031** and answer relevancy **+0.030** — against run-to-run
spreads of 0.003–0.006, a 5–10× signal-to-noise ratio — **with no change to the
generator, the prompt, or the model.** Only the retrieved context changed.

That isolates a causal path the harness was built to expose: retrieval quality
propagates into generation quality. It is the cleanest confirmation we have of the
blame chain in [CONCEPTS.md](CONCEPTS.md) — the generator did not improve, what it was
handed did. It also happens to move the metric weighted highest under D9.

Reaching **Recall@5 = 1.000** is the qualitative part: a relevant chunk is now in the
top-5 for all 30 answerable questions, so no remaining error can be blamed on missing
evidence. Every residual failure is now the generator's.

### Where the gain comes from: one question

Only three questions changed rank, and the decisive one is **q04** (the "110
accessions" item). D8 recategorised it from `factual` to `exact-term` precisely
because it was the pilot's single dense-retrieval miss. It is *still* the only
question dense misses at top-5 — **even with the best embedder in the matrix** — and
BM25 rescues it to rank 5.

The reason is structural: a bare figure like "110" has no semantic neighbourhood for
an embedding model to place it in, but it is a high-IDF exact token for BM25. This is
the entire argument for hybrid retrieval, reduced to a single observable case, and it
is why Recall@5 cannot reach 1.000 without it.

### Caveats

- **Not statistically significant.** McNemar on paired hit@1: *p* = 1.000, one
  discordant question. Bootstrap 95% CI on ΔMRR: [+0.000, +0.073] — bounded at zero
  from below (hybrid never lost in 20,000 resamples), but not separated from it. The
  supportable statement is "dominates on every metric, never underperforms," not
  "proven better." Per F6, separating configurations this close would need on the
  order of 1,000 questions.
- **Latency fell slightly (1.090s → 1.058s)** even though BM25 doubled `search`
  (0.025s → 0.051s), because generation time dropped. Plausible mechanism: tighter
  context yields shorter answers. Recorded as an **observation, not a finding** — 26ms
  is close to measurement noise and was not repeated enough to confirm.
- One corpus, 63 chunks. BM25 rebuilds its index per query, which is affordable here
  and would not be at scale; the comparison does not model that regime.
