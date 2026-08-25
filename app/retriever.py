"""Retrieve: given a question, return the top-K most relevant chunks.

Two modes, selected by config.retrieval_mode (Phase 4):
  - "dense"  : vector similarity over the config's table (the Phase 1-3 path).
  - "hybrid" : dense + BM25 lexical, fused by Reciprocal Rank Fusion (RRF).

Both embed the query with the config's own embedder and read the config's table;
hybrid additionally runs a BM25 pass over that table's text. Mode is a read-time
choice, so a hybrid config reuses its dense twin's table — no re-ingest.
"""

from dataclasses import dataclass

from pgvector import Vector
from psycopg import sql

from app.db import config_table, connect
from app.lexical import keyword_search
from app.pipeline import BASELINE, RunConfig
from app.rbac import User, where_clause

DEFAULT_K = 5
_RRF_K = 60         # RRF constant: dampens the weight of top ranks (standard ~60)
_CANDIDATES = 20    # top-N pulled from each retriever before fusing


@dataclass
class RetrievedChunk:
    content: str
    source: str
    chunk_index: int
    score: float                  # ranking score: cosine similarity (dense) or RRF (hybrid)
    distance: float | None = None    # cosine distance, when the chunk came from dense
    similarity: float | None = None  # 1 - distance, when available


def _dense_retrieve(query: str, config: RunConfig, k: int, user: User | None = None) -> list[RetrievedChunk]:
    query_vector = Vector(config.build_embedder().embed_text(query))
    tbl = config_table(config.config_id)
    where, acl_params = where_clause(user)  # RBAC filter (empty when user is None)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT content,"
                "       metadata->>'source'             AS source,"
                "       (metadata->>'chunk_index')::int AS chunk_index,"
                "       embedding <=> %s                AS distance "
                "FROM {tbl} {where} ORDER BY distance ASC LIMIT %s"
            ).format(tbl=tbl, where=where),
            [query_vector, *acl_params, k],
        )
        rows = cur.fetchall()
    return [
        RetrievedChunk(content=c, source=s, chunk_index=i,
                       score=1.0 - d, distance=d, similarity=1.0 - d)
        for c, s, i, d in rows
    ]


def _hybrid_retrieve(query: str, config: RunConfig, k: int, user: User | None = None) -> list[RetrievedChunk]:
    """Fuse dense and BM25 rankings via RRF: each chunk scores Σ 1/(_RRF_K + rank)
    over the two lists, so a chunk ranked highly by either (or both) rises. The
    RBAC filter is applied inside both sub-retrievers."""
    dense = _dense_retrieve(query, config, _CANDIDATES, user)
    lexical = keyword_search(query, config, _CANDIDATES, user)

    fused: dict[tuple[str, int], dict] = {}
    for rank, r in enumerate(dense, start=1):
        e = fused.setdefault(
            (r.source, r.chunk_index),
            {"content": r.content, "source": r.source, "chunk_index": r.chunk_index,
             "score": 0.0, "distance": r.distance, "similarity": r.similarity},
        )
        e["score"] += 1.0 / (_RRF_K + rank)
    for rank, h in enumerate(lexical, start=1):
        e = fused.setdefault(
            (h.source, h.chunk_index),
            {"content": h.content, "source": h.source, "chunk_index": h.chunk_index,
             "score": 0.0, "distance": None, "similarity": None},
        )
        e["score"] += 1.0 / (_RRF_K + rank)

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:k]
    return [RetrievedChunk(**e) for e in ranked]


def retrieve(
    query: str, config: RunConfig = BASELINE, k: int | None = None, user: User | None = None
) -> list[RetrievedChunk]:
    """Return the k most relevant chunks, dense or hybrid per config.retrieval_mode,
    filtered to what `user` may see (user=None = unrestricted/trusted call)."""
    k = k if k is not None else config.retrieval_k
    if config.retrieval_mode == "hybrid":
        return _hybrid_retrieve(query, config, k, user)
    return _dense_retrieve(query, config, k, user)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Where was cannabis first domesticated?"
    for mode in ("dense", "hybrid"):
        cfg = RunConfig(retrieval_mode=mode)
        print(f"\n=== {mode} ===")
        for r in retrieve(query, cfg):
            extra = f"sim={r.similarity:.3f}" if r.similarity is not None else "sim=—"
            print(f"  #{r.chunk_index} score={r.score:.4f} {extra}  {r.content.strip()[:70]!r}")
