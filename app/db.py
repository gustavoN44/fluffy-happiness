"""Database connection helper.

One place that opens a psycopg connection with the pgvector adapter registered,
so the `vector` type round-trips correctly for everything (store and retrieve)
without duplicating setup.
"""

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn
