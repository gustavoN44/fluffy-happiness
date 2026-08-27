"""BM25 lexical (keyword) search over a config's chunks (Phase 4).

The lexical half of hybrid retrieval. Dense search matches meaning but can blur
exact terms (names, codes, rare words); BM25 matches them literally. This module
is standalone — Step 2 fuses its ranking with the dense ranking (retriever.py).

Scale note (DECISIONS.md D7): the BM25 index is built in memory from the config's
chunk table on each search. Fine at tens of chunks; not how you'd scale it.
"""

import re
from dataclasses import dataclass

from psycopg import sql
from rank_bm25 import BM25Okapi

from app.db import config_table, connect
from app.pipeline import BASELINE, RunConfig
from app.rbac import User, where_clause

DEFAULT_K = 5
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens — the standard simple BM25 tokenization."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class LexicalHit:
    content: str
    source: str
    chunk_index: int
    bm25_score: float


def _load_chunks(config: RunConfig, user: User | None = None) -> list[tuple[str, str, int]]:
    """Chunks in the config's table VISIBLE to `user`, as (content, source,
    chunk_index). Filtering here means forbidden chunks never enter the BM25 index
    — no leak through lexical statistics, not just through results."""
    where, params = where_clause(user)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT content, metadata->>'source', (metadata->>'chunk_index')::int "
                "FROM {tbl} {where}"
            ).format(tbl=config_table(config.config_id), where=where),
            params,
        )
        return cur.fetchall()


def keyword_search(
    query: str, config: RunConfig = BASELINE, k: int = DEFAULT_K, user: User | None = None
) -> list[LexicalHit]:
    """Return the top-k chunks for `query` by BM25 score over the config's chunks
    that `user` is permitted to see."""
    rows = _load_chunks(config, user)
    if not rows:
        return []

    corpus = [_tokenize(content) for content, _, _ in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(rows, scores, strict=True), key=lambda rs: rs[1], reverse=True)[:k]
    return [
        LexicalHit(content=content, source=source, chunk_index=idx, bm25_score=float(score))
        for (content, source, idx), score in ranked
    ]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "CRISPR/Cas9"
    hits = keyword_search(query)
    print(f"query: {query!r}  (config: {BASELINE.label})\ntop-{len(hits)} by BM25:")
    for h in hits:
        print(f"  [{h.source}#{h.chunk_index}] bm25={h.bm25_score:.2f}\n      {h.content.strip()[:110]!r}")
