"""Database access: connections, and the table-per-config schema helpers (Phase 3).

Each RunConfig stores its chunks in its own table `chunks_<config_id>` whose
`vector(dim)` matches that config's embedder, so configs with different embedding
dimensionalities coexist (DECISIONS.md D6). A small `configs` registry table
records what each config_id is (label, params, dim) so the set is self-describing.

Table names are built with psycopg.sql.Identifier (config_id is a hex hash, but we
never string-format identifiers into SQL regardless).
"""

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Json

from app.config import settings

_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS configs (
    config_id       TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    chunker         TEXT NOT NULL,
    chunker_params  JSONB NOT NULL,
    embedder        TEXT NOT NULL,
    embedder_params JSONB NOT NULL,
    dim             INT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def connect() -> psycopg.Connection:
    """Open a connection with the pgvector adapter registered, so Python lists
    map to/from `vector` columns without manual string formatting."""
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def config_table(config_id: str) -> sql.Identifier:
    """The (safely-quoted) chunk-table identifier for a config."""
    return sql.Identifier(f"chunks_{config_id}")


def ensure_registry(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_REGISTRY_DDL)


def ensure_config_table(conn: psycopg.Connection, config_id: str, dim: int) -> None:
    """Create this config's chunk table (sized to `dim`) if it doesn't exist."""
    ddl = sql.SQL(
        "CREATE TABLE IF NOT EXISTS {tbl} ("
        "  id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
        "  content   TEXT NOT NULL,"
        "  embedding vector({dim}) NOT NULL,"
        "  metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb"
        ")"
    ).format(tbl=config_table(config_id), dim=sql.SQL(str(int(dim))))
    with conn.cursor() as cur:
        cur.execute(ddl)


def register_config(conn: psycopg.Connection, config, dim: int) -> None:
    """Upsert the config's metadata into the registry (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO configs "
            "(config_id, label, chunker, chunker_params, embedder, embedder_params, dim) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (config_id) DO UPDATE SET label = EXCLUDED.label",
            (config.config_id, config.label, config.chunker, Json(config.chunker_params),
             config.embedder, Json(config.embedder_params), dim),
        )


def list_configs(conn: psycopg.Connection) -> list[dict]:
    """All registered configs with a live row count of their chunk table."""
    with conn.cursor() as cur:
        cur.execute("SELECT config_id, label, dim FROM configs ORDER BY created_at")
        rows = cur.fetchall()
        out = []
        for cid, label, dim in rows:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(config_table(cid)))
            (n,) = cur.fetchone()
            out.append({"config_id": cid, "label": label, "dim": dim, "chunks": n})
    return out
