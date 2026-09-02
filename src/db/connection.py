# src/db/connection.py
"""Postgres connection helper for the World Cup RAG project."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg


def get_database_url() -> str:
    """Build a connection URL from environment variables (with safe defaults)."""
    user = os.getenv("POSTGRES_USER", "worldcup")
    password = os.getenv("POSTGRES_PASSWORD", "worldcup")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "worldcup")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Yield a psycopg connection that is closed automatically."""
    conn = psycopg.connect(get_database_url())
    try:
        yield conn
    finally:
        conn.close()
