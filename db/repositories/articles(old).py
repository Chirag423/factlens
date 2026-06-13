"""
FactLens — db/repositories/articles.py

All DB operations for the `articles` hypertable.

## Dedup contract

Always call exists(url) before insert(). The ON CONFLICT clause in insert()
is a safety-net for races, not the primary dedup gate.
"""

from __future__ import annotations

import logging
from typing import Optional

import psycopg2.extras

from db.connection import Connection
from db.models import ArticleRow

log = logging.getLogger(__name__)


class ArticleRepository:
    """CRUD + status helpers for the articles hypertable."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def exists(self, url: str) -> bool:
        """
        Return True when url is already present.
        """
        with self._conn._tx() as cur:
            cur.execute(
                "SELECT 1 FROM articles WHERE url = %s LIMIT 1",
                (url,),
            )
            return cur.fetchone() is not None

    def insert(self, article: ArticleRow) -> Optional[int]:
        """
        Insert a new article.

        Returns:
            int: inserted article id
            None: if article already exists
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                INSERT INTO articles (
                    url,
                    title,
                    category,
                    fetched_at,
                    published_at,
                    source_domain,
                    language,
                    state,
                    raw_content,

                    entity_person,
                    entity_norp,
                    entity_fac,
                    entity_org,
                    entity_gpe,
                    entity_loc,
                    entity_product,
                    entity_event,
                    entity_work_of_art,
                    entity_law,
                    entity_language,
                    entity_date,
                    entity_time,
                    entity_percent,
                    entity_money,
                    entity_quantity,
                    entity_ordinal,
                    entity_cardinal,

                    extra
                )
                VALUES (
                    %(url)s,
                    %(title)s,
                    %(category)s,
                    %(fetched_at)s,
                    %(published_at)s,
                    %(source_domain)s,
                    %(language)s,
                    %(state)s,
                    %(raw_content)s,

                    %(entity_person)s,
                    %(entity_norp)s,
                    %(entity_fac)s,
                    %(entity_org)s,
                    %(entity_gpe)s,
                    %(entity_loc)s,
                    %(entity_product)s,
                    %(entity_event)s,
                    %(entity_work_of_art)s,
                    %(entity_law)s,
                    %(entity_language)s,
                    %(entity_date)s,
                    %(entity_time)s,
                    %(entity_percent)s,
                    %(entity_money)s,
                    %(entity_quantity)s,
                    %(entity_ordinal)s,
                    %(entity_cardinal)s,

                    %(extra)s
                )
                ON CONFLICT (url, fetched_at) DO NOTHING
                RETURNING id
                """,
                {
                    "url": article.url,
                    "title": article.title,
                    "category": article.category,
                    "fetched_at": article.fetched_at,
                    "published_at": article.published_at,
                    "source_domain": article.source_domain,
                    "language": article.language,
                    "state": article.state,
                    "raw_content": article.raw_content,

                    "entity_person": article.entity_person,
                    "entity_norp": article.entity_norp,
                    "entity_fac": article.entity_fac,
                    "entity_org": article.entity_org,
                    "entity_gpe": article.entity_gpe,
                    "entity_loc": article.entity_loc,
                    "entity_product": article.entity_product,
                    "entity_event": article.entity_event,
                    "entity_work_of_art": article.entity_work_of_art,
                    "entity_law": article.entity_law,
                    "entity_language": article.entity_language,
                    "entity_date": article.entity_date,
                    "entity_time": article.entity_time,
                    "entity_percent": article.entity_percent,
                    "entity_money": article.entity_money,
                    "entity_quantity": article.entity_quantity,
                    "entity_ordinal": article.entity_ordinal,
                    "entity_cardinal": article.entity_cardinal,

                    "extra": (
                        psycopg2.extras.Json(article.extra)
                        if article.extra is not None
                        else None
                    ),
                },
            )

            row = cur.fetchone()

            if row is None:
                log.debug("article skipped (duplicate): %s", article.url)
                return None

            try:
                art_id = row["id"]  # RealDictCursor
            except (TypeError, KeyError):
                art_id = row[0]  # regular cursor

            log.debug(
                "article inserted id=%d url=%s",
                art_id,
                article.url,
            )

            return art_id

    def set_status(self, article_id: int, status: str) -> None:
        """
        Update processing_status.
        Sets processed_at when status becomes processed/embedded.
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                UPDATE articles
                SET processing_status = %s,
                    processed_at = CASE
                        WHEN %s IN ('processed', 'embedded')
                        THEN NOW()
                        ELSE processed_at
                    END
                WHERE id = %s
                """,
                (status, status, article_id),
            )

    def get_by_state(self, query: str) -> list[dict]:
        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT *
                FROM articles
                WHERE state = %s
                """,
                (query,),
            )

            return cur.fetchall()

    def get_pending(self, limit: int = 100) -> list[dict]:
        """
        Fetch pending articles for workers.
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    url,
                    raw_content,
                    source_domain
                FROM articles
                WHERE processing_status = 'pending'
                ORDER BY fetched_at
                LIMIT %s
                FOR UPDATE SKIP LOCKED
                """,
                (limit,),
            )
            return cur.fetchall()