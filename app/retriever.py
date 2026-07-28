"""Retrieve: given a question, return the top-K most relevant chunks.

Config-aware (Phase 3): retrieval embeds the query with the *same* embedder that
populated the config's table and searches `chunks_<config_id>`. Using a different
embedder than the one that built the table would make the cosine distances
meaningless, so the embedder is taken from the config, not assumed.
"""

from dataclasses import dataclass

from pgvector import Vector
from psycopg import sql

from app.db import config_table, connect
from app.pipeline import BASELINE, RunConfig

DEFAULT_K = 5


@dataclass
class RetrievedChunk:
    content: str
    source: str
    chunk_index: int
    distance: float    # cosine distance from pgvector: lower = closer
    similarity: float  # 1 - distance: higher = more relevant (human-friendly)


def retrieve(query: str, config: RunConfig = BASELINE, k: int | None = None) -> list[RetrievedChunk]:
    """Embed `query` with the config's embedder and return the k nearest chunks
    from the config's table by cosine distance."""
    k = k if k is not None else config.retrieval_k
    query_vector = Vector(config.build_embedder().embed_text(query))
    tbl = config_table(config.config_id)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT content,"
                "       metadata->>'source'             AS source,"
                "       (metadata->>'chunk_index')::int AS chunk_index,"
                "       embedding <=> %s                AS distance "
                "FROM {tbl} "
                "ORDER BY distance ASC "
                "LIMIT %s"
            ).format(tbl=tbl),
            (query_vector, k),
        )
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            content=content, source=source, chunk_index=chunk_index,
            distance=distance, similarity=1.0 - distance,
        )
        for content, source, chunk_index, distance in rows
    ]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Where was cannabis first domesticated?"
    results = retrieve(query)
    print(f"query: {query!r}  (config: {BASELINE.label})\ntop-{len(results)}:")
    for r in results:
        print(
            f"  [{r.source}#{r.chunk_index}] sim={r.similarity:.3f} "
            f"dist={r.distance:.3f}\n      {r.content.strip()[:110]!r}"
        )
