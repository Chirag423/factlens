"""
FactLens — db/repositories/chunks.py

DB operations for the `chunks` table (text segments pre/post embedding).
"""

from __future__ import annotations

import logging
from typing import Optional

import psycopg2.extras

from db.connection import Connection
from db.models import ChunkRow

log = logging.getLogger(__name__)


class ChunkRepository:
    """Insert, bulk-insert, query, and mark-embedded for the chunks table."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def insert(self, chunk: ChunkRow) -> Optional[int]:
        """
        Insert a single chunk.
        Returns the new chunk id, or None if (article_id, chunk_index) already exists.
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                INSERT INTO chunks
                    (article_id, chunk_index, content, start_char, end_char)
                VALUES
                    (%(article_id)s, %(chunk_index)s, %(content)s,)
                ON CONFLICT (article_id, chunk_index) DO NOTHING
                RETURNING id
                """,
                {
                    "article_id":  chunk.article_id,
                    "chunk_index": chunk.chunk_index,
                    "content":     chunk.content,
                },
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def bulk_insert(self, chunks: list[ChunkRow]) -> None:
        """Insert all chunks for an article in a single round-trip. Skips duplicates."""
        rows = [
            (c.article_id, c.chunk_index, c.content, c.start_char, c.end_char)
            for c in chunks
        ]
        with self._conn._tx() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks
                    (article_id, chunk_index, content)
                VALUES %s
                ON CONFLICT (article_id, chunk_index) DO NOTHING
                """,
                rows,
            )

    def get_unembedded(self, limit: int = 500) -> list[dict]:
        """
        Fetch chunks not yet pushed to Qdrant, oldest-first.
        Uses SKIP LOCKED so multiple embedding workers can run in parallel.
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT id, article_id, chunk_index, content
                FROM   chunks
                WHERE  is_embedded = FALSE
                ORDER  BY created_at
                LIMIT  %s
                FOR UPDATE SKIP LOCKED
                """,
                (limit,),
            )
            return cur.fetchall()

    def mark_embedded(self, chunk_ids: list[int]) -> None:
        """Set is_embedded = TRUE for a batch of chunk ids after a successful Qdrant upsert."""
        with self._conn._tx() as cur:
            cur.execute(
                "UPDATE chunks SET is_embedded = TRUE WHERE id = ANY(%s)",
                (chunk_ids,),
            )

    def all_embedded(self, article_id: int) -> bool:
        """Return True when every chunk for the given article has been embedded."""
        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                                     AS total,
                    SUM(CASE WHEN is_embedded THEN 1 ELSE 0 END) AS done
                FROM chunks
                WHERE article_id = %s
                """,
                (article_id,),
            )
            row = cur.fetchone()
            return row["total"] > 0 and row["total"] == row["done"]
