"""
FactLens — db/repositories/chunks.py

DB operations for both the `parent_chunks` and `chunks` tables.

Parent-child architecture
--------------------------
  ParentChunkRepository  — paragraph-level chunks stored in PostgreSQL.
      Each row has a deterministic ``parent_id`` (UUID-v5) so re-runs are
      idempotent via ON CONFLICT (parent_id) DO NOTHING.

  ChunkRepository        — legacy / generic chunk table; retained for future
      use (e.g. simple single-level chunking pipelines).  Not used by the
      parent-child embedding pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import psycopg2.extras

from db.connection import Connection
from db.models import ChunkRow, ParentChunkRow

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ParentChunkRepository
# ═══════════════════════════════════════════════════════════════════════════

class ParentChunkRepository:
    """
    Persistence layer for paragraph-level parent chunks.

    Parent chunks are stored in PostgreSQL so they can be retrieved during
    answer synthesis (the child chunk's Qdrant payload carries ``parent_id``
    which is used to look up the full parent text here).
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def bulk_insert(self, chunks: list[ParentChunkRow]) -> None:
        """
        Insert all parent chunks for a batch of articles in one round-trip.

        Existing rows with the same ``parent_id`` are silently skipped
        (ON CONFLICT DO NOTHING), so the method is safe to call on re-runs.

        Parameters
        ----------
        chunks : list[ParentChunkRow]
            May span multiple articles — typically the accumulated output of
            one processing batch (up to BATCH_SIZE * avg_parents_per_article rows).
        """
        if not chunks:
            return

        rows = [
            (c.parent_id, c.article_id, c.chunk_index, c.content)
            for c in chunks
        ]

        with self._conn._tx() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO parent_chunks (parent_id, article_id, chunk_index, content)
                VALUES %s
                ON CONFLICT (parent_id) DO NOTHING
                """,
                rows,
                page_size=500,
            )

        log.debug("ParentChunkRepository: inserted/skipped %d parent chunk(s).", len(rows))

    def insert(self, chunk: ParentChunkRow) -> Optional[int]:
        """
        Insert a single parent chunk.

        Returns
        -------
        int
            The new ``id`` if the row was inserted.
        None
            If a row with the same ``parent_id`` already exists.
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                INSERT INTO parent_chunks (parent_id, article_id, chunk_index, content)
                VALUES (%(parent_id)s, %(article_id)s, %(chunk_index)s, %(content)s)
                ON CONFLICT (parent_id) DO NOTHING
                RETURNING id
                """,
                {
                    "parent_id":   chunk.parent_id,
                    "article_id":  chunk.article_id,
                    "chunk_index": chunk.chunk_index,
                    "content":     chunk.content,
                },
            )
            row = cur.fetchone()
            if row is None:
                return None
            try:
                return row["id"]   # RealDictCursor
            except (TypeError, KeyError):
                return row[0]

    def get_by_article(self, article_id: int) -> list[dict]:
        """
        Fetch all parent chunks for a given article, ordered by chunk_index.
        Useful for answer synthesis when a child chunk's parent_id is known.
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT id, parent_id, article_id, chunk_index, content, created_at
                FROM   parent_chunks
                WHERE  article_id = %s
                ORDER  BY chunk_index
                """,
                (article_id,),
            )
            return cur.fetchall()

    def get_by_parent_id(self, parent_id: str) -> Optional[dict]:
        """
        Fetch a single parent chunk by its deterministic UUID string.
        Returns None if not found (e.g. before the chunker has run for this article).
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT id, parent_id, article_id, chunk_index, content, created_at
                FROM   parent_chunks
                WHERE  parent_id = %s
                LIMIT  1
                """,
                (parent_id,),
            )
            return cur.fetchone()
        
    """
FactLens — addition for db/repositories/chunks.py
==================================================
Add the method below to your existing ``ParentChunkRepository`` class.

It follows the exact same pattern as ArticleRepository (psycopg2
RealDictCursor, Connection._tx() context manager, dict return type).

Nothing else in chunks.py needs to change.
"""

# ── Paste this method into the ParentChunkRepository class body ──────────────

    def get_by_parent_ids(self, parent_ids: list[str]) -> dict[str, str]:
        """
        Fetch the ``content`` for each parent_id in *parent_ids*.

        Called by RetrievalService after a hybrid Qdrant search to retrieve
        the full parent-paragraph texts that surround the matched child
        (sentence-level) chunks.

        Parameters
        ----------
        parent_ids : list[str]
            UUID-v5 strings identifying the parent_chunks rows to fetch.
            Ordering is intentionally ignored — the caller preserves RRF
            rank order using the original parent_ids_ordered list.

        Returns
        -------
        dict[str, str]
            Mapping of ``parent_id → content``.
            parent_ids that are not present in the table are silently absent
            from the dict; the caller logs a warning for each missing id.
        """
        if not parent_ids:
            return {}

        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT parent_id,
                       content
                FROM   parent_chunks
                WHERE  parent_id = ANY(%s)
                """,
                (parent_ids,),
            )
            rows = cur.fetchall()

        result: dict[str, str] = {}
        for row in rows:
            try:
                result[row["parent_id"]] = row["content"]   # RealDictCursor
            except (TypeError, KeyError):
                result[row[0]] = row[1]                      # plain cursor fallback

        log.debug(
            "get_by_parent_ids: requested %d, found %d.",
            len(parent_ids),
            len(result),
        )
        return result


# ═══════════════════════════════════════════════════════════════════════════
# ChunkRepository  (legacy / generic)
# ═══════════════════════════════════════════════════════════════════════════

class ChunkRepository:
    """
    Insert, bulk-insert, query, and mark-embedded for the generic chunks table.

    Not used by the parent-child embedding pipeline; retained for other
    single-level chunking workflows that may share this codebase.
    """

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
                INSERT INTO chunks (article_id, chunk_index, content)
                VALUES (%(article_id)s, %(chunk_index)s, %(content)s)
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
            if row is None:
                return None
            try:
                return row["id"]
            except (TypeError, KeyError):
                return row[0]

    def bulk_insert(self, chunks: list[ChunkRow]) -> None:
        """Insert chunks in a single round-trip. Skips duplicates."""
        if not chunks:
            return

        rows = [
            (c.article_id, c.chunk_index, c.content)
            for c in chunks
        ]

        with self._conn._tx() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks (article_id, chunk_index, content)
                VALUES %s
                ON CONFLICT (article_id, chunk_index) DO NOTHING
                """,
                rows,
                page_size=500,
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
        if not chunk_ids:
            return
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
                    COUNT(*)                                      AS total,
                    SUM(CASE WHEN is_embedded THEN 1 ELSE 0 END) AS done
                FROM chunks
                WHERE article_id = %s
                """,
                (article_id,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            # RealDictCursor returns a dict; plain cursor returns a tuple.
            try:
                total = row["total"]
                done  = row["done"]
            except (TypeError, KeyError):
                total, done = row[0], row[1]

            return total > 0 and total == done
