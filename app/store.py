"""Store: persist chunks + vectors + metadata into a config's table, and the
ingest entry point that runs the full load -> chunk -> embed -> store path.

Config-aware (Phase 3): each RunConfig writes to its own table `chunks_<config_id>`
using its own chunker and embedder. Re-ingesting a document is idempotent per
(config, source): existing rows for that source in that table are deleted before
the new ones are inserted.
"""

from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Json

from app.db import config_table, connect, ensure_config_table, ensure_registry, register_config
from app.interfaces import Chunk
from app.loader import load_document
from app.pipeline import BASELINE, RunConfig


def store_chunks(
    conn: psycopg.Connection,
    config: RunConfig,
    source: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> tuple[int, int]:
    """Replace all rows for `source` in this config's table. Returns
    (rows_deleted, rows_inserted); delete + insert commit together."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch"
        )

    tbl = config_table(config.config_id)
    rows = [
        (c.text, v, Json({"source": source, "chunk_index": c.chunk_index}))
        for c, v in zip(chunks, vectors)
    ]

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DELETE FROM {} WHERE metadata->>'source' = %s").format(tbl),
            (source,),
        )
        deleted = cur.rowcount
        cur.executemany(
            sql.SQL("INSERT INTO {} (content, embedding, metadata) VALUES (%s, %s, %s)").format(tbl),
            rows,
        )
    conn.commit()
    return deleted, len(rows)


def ingest_document(path: str | Path, config: RunConfig = BASELINE) -> tuple[int, int]:
    """Run the full pipeline for one document under `config` and store it in the
    config's table. Returns (rows_deleted, rows_inserted)."""
    source = str(path)
    chunker = config.build_chunker()
    embedder = config.build_embedder()

    chunks = chunker.chunk(load_document(path))
    vectors = embedder.embed_texts([c.text for c in chunks])

    with connect() as conn:
        ensure_registry(conn)
        ensure_config_table(conn, config.config_id, embedder.dim)
        register_config(conn, config, embedder.dim)
        return store_chunks(conn, config, source, chunks, vectors)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.store <path-to-document>")

    deleted, inserted = ingest_document(sys.argv[1])
    print(f"ingested {sys.argv[1]} into config {BASELINE.label} ({BASELINE.config_id}): "
          f"deleted {deleted} old row(s), inserted {inserted}")
