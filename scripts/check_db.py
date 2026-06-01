"""Connectivity check: can the Python app reach the dockerized Postgres?

Run before writing any pipeline logic. A green check here means the driver,
the connection string in .env, and the pgvector extension all line up.

    .venv/bin/python scripts/check_db.py
"""

import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()

dsn = os.getenv("DATABASE_URL")
if not dsn:
    sys.exit("DATABASE_URL is not set (check your .env)")

try:
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            (version,) = cur.fetchone()

            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            row = cur.fetchone()
            vector_version = row[0] if row else None

            # Round-trip a vector through the driver to prove the type works
            # end-to-end, not just that the extension is installed.
            cur.execute("SELECT '[1,2,3]'::vector <-> '[3,2,1]'::vector;")
            (distance,) = cur.fetchone()
except Exception as exc:  # noqa: BLE001 - surface any connectivity failure plainly
    sys.exit(f"Could not connect / query: {exc}")

print("Connected.")
print(f"  {version.split(',')[0]}")
if vector_version:
    print(f"  pgvector extension: v{vector_version}")
    print(f"  sample vector distance: {distance}")
else:
    sys.exit("Connected, but the 'vector' extension is NOT enabled.")
