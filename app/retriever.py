"""Retrieve: given a question, return the top-K most relevant chunks.

The "use the index" entry point. The query is embedded with the *same* model used
at ingestion (so the vectors live in the same space and are comparable), then the
chunks table is ranked by cosine distance. Results carry their source, position,
and a relevance score so retrieval is transparent rather than a black box.
"""

from dataclasses import dataclass

from pgvector import Vector

from app.db import connect
from app.embedder import embed_text

DEFAULT_K = 5


@dataclass
class RetrievedChunk:
    content: str
    source: str
    chunk_index: int
    distance: float    # cosine distance from pgvector: lower = closer
    similarity: float  # 1 - distance: higher = more relevant (human-friendly)


def retrieve(query: str, k: int = DEFAULT_K) -> list[RetrievedChunk]:
    """Embed `query` and return the k nearest chunks by cosine distance."""
    # Wrap as pgvector.Vector so psycopg sends a true `vector` (the <=> operator
    # has no overload for a plain float array — see DECISIONS/CONCEPTS notes).
    query_vector = Vector(embed_text(query))

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT content,
                   metadata->>'source'              AS source,
                   (metadata->>'chunk_index')::int  AS chunk_index,
                   embedding <=> %s                  AS distance
            FROM chunks
            ORDER BY distance ASC
            LIMIT %s
            """,
            (query_vector, k),
        )
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            content=content,
            source=source,
            chunk_index=chunk_index,
            distance=distance,
            similarity=1.0 - distance,
        )
        for content, source, chunk_index, distance in rows
    ]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Where was cannabis first domesticated?"
    results = retrieve(query)
    print(f"query: {query!r}\ntop-{len(results)}:")
    for r in results:
        print(
            f"  [{r.source}#{r.chunk_index}] sim={r.similarity:.3f} "
            f"dist={r.distance:.3f}\n      {r.content.strip()[:110]!r}"
        )
