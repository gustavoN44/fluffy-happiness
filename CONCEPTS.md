# Concepts

A plain-language reference for how each component of the pipeline works and why
it exists. This is the "how it works" companion to [DECISIONS.md](DECISIONS.md)
(which records *why specific choices* were made) and [ROADMAP.md](ROADMAP.md)
(the build sequence). Sections are added as components are built.

The pipeline so far: **load → chunk → embed → store** → (retrieve → generate).

---

## Load

**File:** [app/loader.py](app/loader.py) · **Entry:** `load_document(path) -> str`

**Function in the pipeline.** The entry point: take one source document off disk
and hand the rest of the pipeline a single clean string of text. Everything
downstream operates on text, so this is the only component that cares about file
formats.

**How it works.**
- Dispatches on file extension: `.pdf` is read with `pypdf` (extract text from
  every page, join pages with blank lines); `.txt` / `.md` / `.markdown` are read
  directly as UTF-8.
- Returns the document's full raw text as one string — no cleaning, no splitting.
- Guards against silent failure: missing file → `FileNotFoundError`; unsupported
  extension → `ValueError`; a file that loads but yields no text (empty, or a
  scanned/image-only PDF) → `ValueError`. That last one matters because a
  scanned PDF *looks* fine but extracts nothing, and embedding empty strings
  would quietly poison retrieval.

**Why it's deliberately dumb.** PDF extraction leaves artifacts — page numbers,
headers, reference-list formatting. We do *not* clean them in Phase 1 ("ugly is
acceptable, working is mandatory"); whether they hurt retrieval is a question for
the Phase 2 eval harness, not a guess made now.

---

## Chunk

**File:** [app/chunker.py](app/chunker.py) · **Entry:** `chunk_text(text) -> list[Chunk]`

**Function in the pipeline.** Split the document's text into smaller, overlapping
pieces ("chunks"). Chunks are the *unit of retrieval*: we embed and later fetch
chunks, not whole documents. Why split at all? An embedding compresses its whole
input into one fixed-size vector — embed a 50-page document as one vector and all
specificity is averaged away. Chunks keep each vector about one focused span of
text, so retrieval can return the precise passage that answers a question.

**How it works.** Phase 1 baseline: **recursive** splitting, **512 tokens** per
chunk, **15% overlap**, measured in tokens via `tiktoken` (`cl100k_base`, the
tokenizer `text-embedding-3-small` uses).
- *Recursive* = try to split on the coarsest natural boundary first
  (paragraph → line → sentence → word → character), descending to a finer
  boundary only when a piece is still over budget. This keeps chunk *endings*
  aligned to natural breaks instead of slicing mid-word.
- *Overlap* = consecutive chunks repeat ~77 tokens of each other. A fact that
  straddles a boundary then appears whole in at least one chunk, so retrieval
  doesn't lose it. Implemented as a global "stride" pass: build non-overlapping
  base chunks at 512−77 = 435 tokens, then prepend each with the previous
  chunk's trailing 77 tokens (see [DECISIONS.md](DECISIONS.md) D1 for why this
  replaced the first, piece-bounded approach).
- Output is `Chunk(text, chunk_index)` objects — the index records each chunk's
  position in the document, which will populate the `metadata` column at store time.

**Why tokens, not characters.** Cost and the embedding model's input limit are
both denominated in tokens, and the project optimizes for quality *per token*. A
token budget means "512" is the same quantity the model and the bill see.

---

## Embed

**File:** [app/embedder.py](app/embedder.py) · **Entry:** `embed_texts(list[str]) -> list[list[float]]`

**Function in the pipeline.** Turn each chunk's text into a **vector** — a list
of 1536 numbers — that captures its meaning. These vectors are what get stored
and searched. Retrieval works by embedding the user's question the same way and
finding the chunk vectors closest to it, so embedding is the bridge between
"text" and "math we can compare."

**How it works.**
- Sends chunk text to OpenAI's `text-embedding-3-small` model, which returns a
  **1536-dimensional** vector per input. The model is trained so that texts with
  similar meaning land near each other in this 1536-D space, even when they share
  no words.
- "Closeness" is measured by **cosine similarity** (the angle between two
  vectors): ~1.0 = very similar meaning, ~0 = unrelated. Our smoke test showed a
  related pair at 0.815 vs an unrelated pair at 0.044.
- Requests are **batched** (up to 100 inputs per call) and returned in input
  order (sorted by the API's `index` field), so vectors line up 1:1 with chunks.
- A **dimensionality guard** rejects any vector that isn't 1536 dims — the number
  must exactly match the `chunks.embedding vector(1536)` column or inserts fail.
- The module is pure (text in, vectors out); it knows nothing about chunks or the
  database. The same `embed_text()` is reused at query time to embed the question.

**Why this model.** `text-embedding-3-small` is the cheap, capable Phase 1
baseline. Phase 5's experiment matrix compares it against a quality contender;
until the eval harness exists, it's a starting point, not a validated "best."

### Deep dive: what "batched" means

The naive way to embed 34 chunks is 34 separate HTTP requests — one per chunk.
Each request carries fixed overhead (network round-trip, TLS, queueing, auth), so
doing it 34 times is slow and wasteful. **Batching** means sending many inputs in
a *single* request: OpenAI's embeddings endpoint accepts a list of texts and
returns a list of vectors, so 34 chunks become one call returning 34 vectors.

```python
_BATCH_SIZE = 100
for start in range(0, len(texts), _BATCH_SIZE):
    batch = texts[start : start + _BATCH_SIZE]   # up to 100 at a time
    response = _client.embeddings.create(input=batch, ...)
```

With 34 chunks and a batch size of 100, everything fits in one request; 250
chunks would make three (0–99, 100–199, 200–249). The cap exists because (1) the
API rejects requests whose inputs exceed a token/size ceiling, so one giant
request would fail, and (2) if a request fails you lose only that batch, not the
whole job.

**Ordering subtlety:** a batch response can come back out of order, so each item
carries an `index` (its position in the input). Sorting by it guarantees
`vectors[i]` matches `texts[i]` — essential, because the store step pairs each
vector with the *correct* chunk.

```python
for item in sorted(response.data, key=lambda d: d.index):
    vectors.append(item.embedding)
```

In short: batching = fewer, larger requests instead of many tiny ones — faster,
less overhead, and bounded so it scales to large documents.

### Deep dive: cosine similarity

Cosine similarity measures how similar two vectors are *in direction*, ignoring
their length — it's the cosine of the angle between them. An embedding encodes
meaning as a *direction* in 1536-D space: texts that mean the same thing point the
same way, even if one vector has a larger magnitude. Cosine ignores magnitude and
asks only whether they point the same way.

Scale (typical for text embeddings): **1.0** = same direction / same meaning;
**~0.0** = perpendicular / unrelated; **negative** = opposite (rare with these
models). Our smoke test: related pair **0.815**, unrelated pair **0.044**.

```
cos(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

- `A · B` (dot product): multiply element-wise and sum — large when big values
  line up in the same positions.
- `‖A‖` (magnitude): `sqrt(sum of squares)`.
- Dividing by both magnitudes normalizes into the −1…1 range; that division is
  what removes "length" and leaves only "direction."

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))   # A · B
    na  = math.sqrt(sum(x * x for x in a))   # ‖A‖
    nb  = math.sqrt(sum(y * y for y in b))   # ‖B‖
    return dot / (na * nb)
```

Geometric intuition (shown in 2D; the math is identical with 1536 terms):

```
        ^                              ^
        |   B                          | B
        |  /                           |  \
        | /  small angle               |   \  ~90° angle
        |/__________ A   → ~1          |____\______ A  → ~0
       similar meaning                  unrelated
```

**Connection to retrieval (Step 7):** the user's question is embedded into the
same space, and the chunks with the highest cosine similarity to it are the
passages most likely to answer it.

**pgvector note:** the database ranks by *distance*, not similarity. pgvector's
operators are `<=>` (cosine distance = `1 − cosine_similarity`; 0 = identical),
`<->` (L2/Euclidean), and `<#>` (negative inner product). For meaning-based
ranking we use `<=>` and sort ascending — smallest distance = closest match. Same
concept as cosine similarity, expressed as "lower is better" because that's what
the scan/index operators are built around.

---

## Store

**File:** [app/store.py](app/store.py) · **Entries:** `ingest_document(path)`, `store_chunks(conn, source, chunks, vectors)`

**Function in the pipeline.** Persist each chunk's text, its 1536-dim vector, and
its metadata into the `chunks` table, turning the in-memory pipeline output into a
durable, queryable index. This is the boundary between "build the index" (load →
chunk → embed → store) and "use the index" (retrieve → generate). `ingest_document`
is the one-command path that runs the whole left half for a single document.

**How it works.**
- **The table** (`chunks`, defined in [db/init/02-schema.sql](db/init/02-schema.sql)):
  `id` (auto), `content` (the chunk text), `embedding` (`vector(1536)`), and
  `metadata` (`JSONB`). Metadata currently holds `{source, chunk_index}`; the
  RBAC ownership fields join it in Phase 4 — the column exists now so that's not a
  migration later.
- **The pgvector adapter.** `register_vector(conn)` teaches `psycopg` the
  `vector` type, so a Python `list[float]` round-trips to/from the column without
  hand-formatting pgvector's `'[...]'` string syntax.
- **Idempotent per source.** Before inserting, all rows whose
  `metadata->>'source'` matches this document are deleted, then the new chunks are
  inserted — delete + insert in one transaction. Re-ingesting a document never
  duplicates, and it leaves other documents untouched (verified: a second ingest
  reported "deleted 34, inserted 34").
- **No vector index yet** — at ~34 rows a brute-force exact scan is faster and
  simpler than an approximate HNSW/IVFFlat index. Added only when scan latency
  shows up in eval (a deliberate Phase 1 choice).

**The assignment-cast gotcha.** pgvector defines an *assignment* cast from
array → vector, so an `INSERT` silently accepts a plain Python list (assignment
context). But an operator expression like `embedding <=> $1` is **not** an
assignment context, so the same list arrives as `double precision[]` and
`vector <=> float8[]` doesn't exist — the query errors. Inserts "just work";
similarity queries must cast the parameter (`%s::vector`) or pass a
`pgvector.Vector`. This is why retrieval handles the query vector explicitly
rather than relying on the insert-time magic.
