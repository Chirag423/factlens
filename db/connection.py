"""
FactLens — db/connection.py

Low-level connection management.

Responsibilities
----------------
• Open / close the psycopg2 connection.
• Provide _tx() — a context manager that yields a cursor inside a
  commit/rollback transaction.
• Support use as a context manager (with Connection(...) as conn).

Repositories receive a Connection instance and call _tx(); they never
touch the raw psycopg2 connection directly.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


class Connection:
    """
    Wraps a single psycopg2 connection.

    Usage
    -----
        conn = Connection()                 # reads DATABASE_URL from env
        conn = Connection("postgresql://...") # explicit DSN

        # as context manager
        with Connection() as conn:
            ...

        # in repositories
        with conn._tx() as cur:
            cur.execute(...)
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        url = database_url or os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set "
                "and no database_url argument was provided."
            )
        self._conn: psycopg2.extensions.connection = psycopg2.connect(
            url, cursor_factory=psycopg2.extras.RealDictCursor
        )
        log.info("Connected to database.")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._conn.closed:
            self._conn.close()
            log.info("Database connection closed.")

    @contextmanager
    def _tx(self) -> Generator[psycopg2.extensions.cursor, None, None]:
        """
        Yield a cursor inside a commit/rollback transaction block.

        The ``with self._conn:`` idiom commits on exit and rolls back on
        exception — this is psycopg2's connection-as-context-manager
        behaviour, NOT the same as autocommit.
        """
        with self._conn:
            with self._conn.cursor() as cur:
                yield cur

    @property
    def raw(self) -> psycopg2.extensions.connection:
        """
        Expose the raw connection for callers that need it (e.g. SchemaManager).
        Use sparingly — prefer _tx() for all DML.
        """
        return self._conn
