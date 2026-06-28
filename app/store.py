"""Store: persist chunks + vectors + metadata into the `chunks` table, and the
ingest entry point that runs the full load -> chunk -> embed -> store path.

Re-ingesting a document is idempotent per source: existing rows for that source
are deleted before the new ones are inserted, so re-running never duplicates and
survives having multiple documents in the table.
"""

from pathlib import Path

import psycopg
from psycopg.types.json import Json

from app.chunker import Chunk, chunk_text
from app.db import connect
from app.embedder import embed_texts
from app.loader import load_document


def store_chunks(
    conn: psycopg.Connection,
    source: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> tuple[int, int]:
    """Replace all rows for `source` with the given chunks/vectors. Returns
    (rows_deleted, rows_inserted). One transaction: delete + insert commit together."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch"
        )

    rows = [
        (c.text, v, Json({"source": source, "chunk_index": c.chunk_index}))
        for c, v in zip(chunks, vectors)
    ]

    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE metadata->>'source' = %s", (source,))
        deleted = cur.rowcount
        cur.executemany(
            "INSERT INTO chunks (content, embedding, metadata) VALUES (%s, %s, %s)",
            rows,
        )
    conn.commit()
    return deleted, len(rows)


def ingest_document(path: str | Path) -> tuple[int, int]:
    """Run the full pipeline for one document and store it. `source` in metadata
    is the path as given. Returns (rows_deleted, rows_inserted)."""
    source = str(path)
    chunks = chunk_text(load_document(path))
    vectors = embed_texts([c.text for c in chunks])
    with connect() as conn:
        return store_chunks(conn, source, chunks, vectors)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.store <path-to-document>")

    deleted, inserted = ingest_document(sys.argv[1])
    print(f"ingested {sys.argv[1]}: deleted {deleted} old row(s), inserted {inserted}")
