"""
FactLens — db.py
Full database layer: schema deployment + insert/upsert functions for all tables.

Usage:
    python db.py                    # deploy schema (create tables, hypertable, indexes)
    python db.py --reset            # DROP all tables then redeploy (destructive!)

From code:
    from db import Database

    db = Database()                 # reads DATABASE_URL from env
    db.deploy_schema()              # idempotent — safe to call on every startup

    feed_id  = db.insert_feed(url="https://feeds.feedburner.com/ndtvnews-top-stories", name="NDTV", category="politics")
    art_id   = db.insert_article(url="https://...", raw_content="...", feed_id=feed_id, ...)
    ent_id   = db.upsert_entity(name="Narendra Modi", entity_type="PERSON", aliases=["Modi", "PM Modi"])
    db.insert_entity_mention(article_id=art_id, entity_id=ent_id, mention_text="PM Modi")
    db.upsert_entity_relation(from_entity_id=1, relation_type="MEMBER_OF", to_entity_id=2, article_id=art_id)
    chunk_id = db.insert_chunk(article_id=art_id, chunk_index=0, content="...", start_char=0, end_char=1200)
    db.mark_chunks_embedded(chunk_ids=[chunk_id])
    db.close()

TimescaleDB note:
    TimescaleDB requires that every UNIQUE / PRIMARY KEY constraint on a hypertable
    includes the partition column (fetched_at).  To avoid this restriction while keeping
    clean dedup semantics we:
      • Remove the inline PRIMARY KEY / UNIQUE constraints from the articles DDL.
      • Create UNIQUE indexes on (id, fetched_at) and (url_hash, fetched_at) instead.
      • Guard against same-URL re-inserts with a fast article_exists() pre-check on
        url_hash alone (plain B-tree index, not a constraint) BEFORE calling
        insert_article(), so the INSERT…ON CONFLICT path is a true "never happens"
        fallback rather than the primary dedup gate.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator, Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (what callers pass in — no raw dicts)
# ---------------------------------------------------------------------------

@dataclass
class FeedRow:
    url: str
    name: Optional[str] = None
    category: Optional[str] = None
    is_active: bool = True


@dataclass
class ArticleRow:
    url: str
    raw_content: str
    feed_id: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    source_domain: Optional[str] = None
    language: str = "en"
    extra: Optional[dict] = None


@dataclass
class EntityRow:
    name: str
    entity_type: str                    # PERSON | ORG | GPE | LAW | EVENT
    aliases: list[str] = field(default_factory=list)
    extra: Optional[dict] = None


@dataclass
class ChunkRow:
    article_id: int
    chunk_index: int
    content: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_EXTENSION = "CREATE EXTENSION IF NOT EXISTS timescaledb;"

_DDL_FEEDS = """
CREATE TABLE IF NOT EXISTS feeds (
    id            SERIAL PRIMARY KEY,
    url           TEXT        NOT NULL UNIQUE,
    name          TEXT,
    category      TEXT,
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
    last_fetched  TIMESTAMPTZ,
    fetch_errors  INTEGER     NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# ---------------------------------------------------------------------------
# NOTE: No PRIMARY KEY or UNIQUE constraints on articles.
#
# TimescaleDB partitions the table by fetched_at.  Postgres enforces unique
# constraints with indexes, and those indexes must span every partition shard,
# which requires the partition column to be part of the key.  Rather than
# make (id, fetched_at) the PK — which would cascade complexity to every FK
# on entity_mentions and chunks — we drop the constraints here and replace
# them with UNIQUE indexes that include fetched_at (see _DDL_INDEXES).
#
# Dedup strategy:
#   • article_exists(url) does a fast lookup on idx_articles_url_hash_only
#     (a plain non-unique index on url_hash alone) before any INSERT.
#   • insert_article() still carries ON CONFLICT (url_hash, fetched_at) as a
#     safety net for races, but the pre-check is the primary dedup gate.
# ---------------------------------------------------------------------------

_DDL_ARTICLES = """
CREATE TABLE IF NOT EXISTS articles (
    id                BIGSERIAL,
    url               TEXT        NOT NULL,
    url_hash          CHAR(64)    NOT NULL,
    feed_id           INTEGER     REFERENCES feeds(id) ON DELETE SET NULL,
    title             TEXT,
    author            TEXT,
    published_at      TIMESTAMPTZ,
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_domain     TEXT,
    language          VARCHAR(10) NOT NULL DEFAULT 'en',
    raw_content       TEXT,
    content_chars     INTEGER,
    processing_status TEXT        NOT NULL DEFAULT 'pending'
                          CHECK (processing_status IN
                                 ('pending','processing','processed','embedded','failed')),
    processed_at      TIMESTAMPTZ,
    extra             JSONB
);
"""

_DDL_ARTICLES_HYPERTABLE = """
SELECT create_hypertable(
    'articles',
    'fetched_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);
"""

_DDL_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
    id             SERIAL      PRIMARY KEY,
    name           TEXT        NOT NULL,
    entity_type    TEXT        NOT NULL
                       CHECK (entity_type IN ('PERSON','ORG','GPE','LAW','EVENT')),
    aliases        TEXT[]      NOT NULL DEFAULT '{}',
    first_seen     TIMESTAMPTZ,
    last_seen      TIMESTAMPTZ,
    mention_count  INTEGER     NOT NULL DEFAULT 0,
    extra          JSONB,
    UNIQUE (name, entity_type)
);
"""

_DDL_ENTITY_MENTIONS = """
CREATE TABLE IF NOT EXISTS entity_mentions (
    id            BIGSERIAL   PRIMARY KEY,
    article_id    BIGINT      NOT NULL,
    entity_id     INTEGER     NOT NULL REFERENCES entities(id)  ON DELETE CASCADE,
    mention_text  TEXT,
    mention_count INTEGER     NOT NULL DEFAULT 1,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (article_id, entity_id)
);
"""

# NOTE: entity_mentions.article_id deliberately has no FK to articles(id).
# A FK referencing a hypertable column that is not part of a unique *constraint*
# (only a unique index) is not supported in PostgreSQL.  Referential integrity
# is maintained by the application layer (insert_article → insert_entity_mention).

_DDL_ENTITY_RELATIONS = """
CREATE TABLE IF NOT EXISTS entity_relations (
    id                   BIGSERIAL PRIMARY KEY,
    from_entity_id       INTEGER   NOT NULL REFERENCES entities(id),
    relation_type        TEXT      NOT NULL,
    to_entity_id         INTEGER   NOT NULL REFERENCES entities(id),
    confidence           REAL      NOT NULL DEFAULT 0.5,
    weight               INTEGER   NOT NULL DEFAULT 1,
    first_seen           TIMESTAMPTZ,
    last_seen            TIMESTAMPTZ,
    evidence_article_ids BIGINT[]  NOT NULL DEFAULT '{}',
    extra                JSONB,
    UNIQUE (from_entity_id, relation_type, to_entity_id)
);
"""

_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL   PRIMARY KEY,
    article_id   BIGINT      NOT NULL,
    chunk_index  INTEGER     NOT NULL,
    content      TEXT        NOT NULL,
    start_char   INTEGER,
    end_char     INTEGER,
    is_embedded  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (article_id, chunk_index)
);
"""

# NOTE: chunks.article_id also has no FK for the same reason as entity_mentions.

_DDL_INDEXES = [
    # --- articles: TimescaleDB-compatible unique indexes (include fetched_at) ---
    # Used to assert row-level uniqueness across hypertable shards.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_id_fetched        ON articles (id, fetched_at);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_hash_fetched  ON articles (url_hash, fetched_at);",
    # Plain index on url_hash alone — used by article_exists() for fast O(log n) dedup
    # without needing to know fetched_at at call time.
    "CREATE INDEX IF NOT EXISTS idx_articles_url_hash_only        ON articles (url_hash);",
    # --- standard lookup indexes ---
    "CREATE INDEX IF NOT EXISTS idx_articles_fetched_at           ON articles (fetched_at);",
    "CREATE INDEX IF NOT EXISTS idx_articles_published_at         ON articles (published_at);",
    "CREATE INDEX IF NOT EXISTS idx_articles_feed_id              ON articles (feed_id);",
    "CREATE INDEX IF NOT EXISTS idx_articles_processing_status    ON articles (processing_status);",
    "CREATE INDEX IF NOT EXISTS idx_articles_source_domain        ON articles (source_domain);",
    "CREATE INDEX IF NOT EXISTS idx_entity_mentions_article_id    ON entity_mentions (article_id);",
    "CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity_id     ON entity_mentions (entity_id);",
    "CREATE INDEX IF NOT EXISTS idx_entity_relations_from         ON entity_relations (from_entity_id);",
    "CREATE INDEX IF NOT EXISTS idx_entity_relations_to           ON entity_relations (to_entity_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_article_id             ON chunks (article_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_is_embedded            ON chunks (is_embedded);",
]

_DDL_DROP_ALL = """
DROP TABLE IF EXISTS chunks            CASCADE;
DROP TABLE IF EXISTS entity_relations  CASCADE;
DROP TABLE IF EXISTS entity_mentions   CASCADE;
DROP TABLE IF EXISTS entities          CASCADE;
DROP TABLE IF EXISTS articles          CASCADE;
DROP TABLE IF EXISTS feeds             CASCADE;
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """
    Single entry point for all FactLens DB operations.
    Maintains one persistent connection; call close() when done,
    or use it as a context manager.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        url = database_url or os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is not set.")
        self._conn: psycopg2.extensions.connection = psycopg2.connect(
            url, cursor_factory=psycopg2.extras.RealDictCursor
        )
        log.info("Connected to database.")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if not self._conn.closed:
            self._conn.close()
            log.info("Database connection closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _tx(self) -> Generator[psycopg2.extensions.cursor, None, None]:
        """Yield a cursor inside a commit/rollback transaction."""
        with self._conn:
            with self._conn.cursor() as cur:
                yield cur

    def _autocommit(self, sql: str, label: str) -> None:
        """Execute DDL that must run outside a transaction."""
        old = self._conn.autocommit
        self._conn.autocommit = True
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql)
                log.info("DDL OK — %s", label)
        except psycopg2.Error as exc:
            log.error("DDL FAIL — %s — %s", label, exc.pgerror or exc)
            raise
        finally:
            self._conn.autocommit = old

    # ------------------------------------------------------------------
    # Schema deployment
    # ------------------------------------------------------------------

    def deploy_schema(self) -> None:
        """Create all tables, hypertable, and indexes. Safe to call on every startup."""
        log.info("Deploying schema...")
        self._autocommit(_DDL_EXTENSION,           "extension timescaledb")
        self._autocommit(_DDL_FEEDS,               "table feeds")
        self._autocommit(_DDL_ARTICLES,            "table articles")
        self._autocommit(_DDL_ARTICLES_HYPERTABLE, "hypertable articles")
        self._autocommit(_DDL_ENTITIES,            "table entities")
        self._autocommit(_DDL_ENTITY_MENTIONS,     "table entity_mentions")
        self._autocommit(_DDL_ENTITY_RELATIONS,    "table entity_relations")
        self._autocommit(_DDL_CHUNKS,              "table chunks")
        for stmt in _DDL_INDEXES:
            label = stmt.split("idx_")[1].split(" ")[0] if "idx_" in stmt else stmt[:50]
            self._autocommit(stmt, f"index {label}")
        log.info("Schema deployment complete.")

    def reset_schema(self) -> None:
        """DROP all tables then redeploy. DESTRUCTIVE — dev/test only."""
        log.warning("Resetting schema — all data will be lost.")
        self._autocommit(_DDL_DROP_ALL, "drop all tables")
        self.deploy_schema()

    # ------------------------------------------------------------------
    # feeds
    # ------------------------------------------------------------------

    def insert_feed(self, feed: FeedRow) -> int:
        """
        Insert a feed. On URL conflict, update name/category/is_active and return existing id.
        Returns feed id.
        """
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO feeds (url, name, category, is_active)
                VALUES (%(url)s, %(name)s, %(category)s, %(is_active)s)
                ON CONFLICT (url) DO UPDATE
                    SET name      = EXCLUDED.name,
                        category  = EXCLUDED.category,
                        is_active = EXCLUDED.is_active
                RETURNING id
                """,
                {
                    "url":       feed.url,
                    "name":      feed.name,
                    "category":  feed.category,
                    "is_active": feed.is_active,
                },
            )
            row = cur.fetchone()
            feed_id = row["id"]
            log.debug("feed id=%d  url=%s", feed_id, feed.url)
            return feed_id

    def mark_feed_fetched(self, feed_id: int) -> None:
        """Update last_fetched to NOW() and reset fetch_errors."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE feeds SET last_fetched = NOW(), fetch_errors = 0 WHERE id = %s",
                (feed_id,),
            )

    def increment_feed_error(self, feed_id: int) -> int:
        """Bump fetch_errors by 1. Returns new count (use for circuit-breaker logic)."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE feeds SET fetch_errors = fetch_errors + 1 WHERE id = %s RETURNING fetch_errors",
                (feed_id,),
            )
            return cur.fetchone()["fetch_errors"]

    def get_active_feeds(self) -> list[dict]:
        """Return all feeds where is_active = true."""
        with self._tx() as cur:
            cur.execute("SELECT * FROM feeds WHERE is_active = TRUE ORDER BY id")
            return cur.fetchall()

    # ------------------------------------------------------------------
    # articles
    # ------------------------------------------------------------------

    @staticmethod
    def _url_hash(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    def article_exists(self, url: str) -> bool:
        """
        Return True if url_hash already present — fast dedup check.
        Uses idx_articles_url_hash_only (plain index on url_hash, no fetched_at needed).
        Always call this before insert_article() to avoid silent skips.
        """
        with self._tx() as cur:
            cur.execute(
                "SELECT 1 FROM articles WHERE url_hash = %s LIMIT 1",
                (self._url_hash(url),),
            )
            return cur.fetchone() is not None

    def insert_article(self, article: ArticleRow) -> Optional[int]:
        """
        Insert article. Returns new id, or None if URL already exists (skip).
        Automatically computes url_hash and content_chars.

        Dedup note: ON CONFLICT targets (url_hash, fetched_at) — the composite
        unique index required by TimescaleDB.  The article_exists() pre-check
        guards against the theoretical case where the same URL arrives within the
        same timestamp second (extremely unlikely but handled gracefully).
        """
        url_hash      = self._url_hash(article.url)
        content_chars = len(article.raw_content) if article.raw_content else 0

        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO articles
                    (url, url_hash, feed_id, title, author, published_at,
                     source_domain, language, raw_content, content_chars, extra)
                VALUES
                    (%(url)s, %(url_hash)s, %(feed_id)s, %(title)s, %(author)s,
                     %(published_at)s, %(source_domain)s, %(language)s,
                     %(raw_content)s, %(content_chars)s, %(extra)s)
                ON CONFLICT (url_hash, fetched_at) DO NOTHING
                RETURNING id
                """,
                {
                    "url":           article.url,
                    "url_hash":      url_hash,
                    "feed_id":       article.feed_id,
                    "title":         article.title,
                    "author":        article.author,
                    "published_at":  article.published_at,
                    "source_domain": article.source_domain,
                    "language":      article.language,
                    "raw_content":   article.raw_content,
                    "content_chars": content_chars,
                    "extra":         psycopg2.extras.Json(article.extra) if article.extra else None,
                },
            )
            row = cur.fetchone()
            if row is None:
                log.debug("article skipped (duplicate): %s", article.url)
                return None
            art_id = row["id"]
            log.debug("article inserted id=%d  url=%s", art_id, article.url)
            return art_id

    def set_article_status(self, article_id: int, status: str) -> None:
        """Update processing_status. Sets processed_at when status='processed'/'embedded'."""
        with self._tx() as cur:
            cur.execute(
                """
                UPDATE articles
                SET processing_status = %s,
                    processed_at = CASE WHEN %s IN ('processed','embedded') THEN NOW() ELSE processed_at END
                WHERE id = %s
                """,
                (status, status, article_id),
            )

    def get_pending_articles(self, limit: int = 100) -> list[dict]:
        """Fetch up to `limit` articles in 'pending' state for the Celery worker."""
        with self._tx() as cur:
            cur.execute(
                """
                SELECT id, url, raw_content, feed_id, source_domain
                FROM   articles
                WHERE  processing_status = 'pending'
                ORDER  BY fetched_at
                LIMIT  %s
                FOR UPDATE SKIP LOCKED
                """,
                (limit,),
            )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # entities
    # ------------------------------------------------------------------

    def upsert_entity(self, entity: EntityRow) -> int:
        """
        Insert or update entity by (name, entity_type).
        On conflict: merges aliases, bumps mention_count, updates last_seen.
        Returns entity id.
        """
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO entities (name, entity_type, aliases, first_seen, last_seen, mention_count, extra)
                VALUES (%(name)s, %(entity_type)s, %(aliases)s, NOW(), NOW(), 1, %(extra)s)
                ON CONFLICT (name, entity_type) DO UPDATE
                    SET aliases       = (
                            SELECT array_agg(DISTINCT elem)
                            FROM   unnest(entities.aliases || EXCLUDED.aliases) AS elem
                        ),
                        last_seen     = NOW(),
                        mention_count = entities.mention_count + 1
                RETURNING id
                """,
                {
                    "name":        entity.name,
                    "entity_type": entity.entity_type,
                    "aliases":     entity.aliases,
                    "extra":       psycopg2.extras.Json(entity.extra) if entity.extra else None,
                },
            )
            return cur.fetchone()["id"]

    def get_entity_by_name(self, name: str, entity_type: str) -> Optional[dict]:
        with self._tx() as cur:
            cur.execute(
                "SELECT * FROM entities WHERE name = %s AND entity_type = %s",
                (name, entity_type),
            )
            return cur.fetchone()

    # ------------------------------------------------------------------
    # entity_mentions
    # ------------------------------------------------------------------

    def insert_entity_mention(
        self,
        article_id: int,
        entity_id: int,
        mention_text: Optional[str] = None,
        mention_count: int = 1,
    ) -> int:
        """
        Link an entity to an article. On duplicate (same article+entity),
        increments mention_count instead of raising.
        Returns mention id.
        """
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO entity_mentions (article_id, entity_id, mention_text, mention_count)
                VALUES (%(article_id)s, %(entity_id)s, %(mention_text)s, %(mention_count)s)
                ON CONFLICT (article_id, entity_id) DO UPDATE
                    SET mention_count = entity_mentions.mention_count + EXCLUDED.mention_count
                RETURNING id
                """,
                {
                    "article_id":    article_id,
                    "entity_id":     entity_id,
                    "mention_text":  mention_text,
                    "mention_count": mention_count,
                },
            )
            return cur.fetchone()["id"]

    def bulk_insert_entity_mentions(self, rows: list[tuple[int, int, str, int]]) -> None:
        """
        High-throughput bulk insert for NLP worker output.
        Each tuple: (article_id, entity_id, mention_text, mention_count)
        """
        with self._tx() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO entity_mentions (article_id, entity_id, mention_text, mention_count)
                VALUES %s
                ON CONFLICT (article_id, entity_id) DO UPDATE
                    SET mention_count = entity_mentions.mention_count + EXCLUDED.mention_count
                """,
                rows,
            )

    # ------------------------------------------------------------------
    # entity_relations
    # ------------------------------------------------------------------

    def upsert_entity_relation(
        self,
        from_entity_id: int,
        relation_type: str,
        to_entity_id: int,
        article_id: int,
        confidence_delta: float = 0.05,
    ) -> int:
        """
        Insert or strengthen a relation edge.
        On conflict: increments weight, bumps confidence, appends article_id to evidence.
        Returns relation id.
        """
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO entity_relations
                    (from_entity_id, relation_type, to_entity_id,
                     confidence, weight, first_seen, last_seen, evidence_article_ids)
                VALUES
                    (%(from_id)s, %(rel_type)s, %(to_id)s,
                     0.5, 1, NOW(), NOW(), ARRAY[%(article_id)s::bigint])
                ON CONFLICT (from_entity_id, relation_type, to_entity_id) DO UPDATE
                    SET weight               = entity_relations.weight + 1,
                        confidence           = LEAST(entity_relations.confidence + %(delta)s, 1.0),
                        last_seen            = NOW(),
                        evidence_article_ids = array_append(
                            entity_relations.evidence_article_ids, %(article_id)s::bigint
                        )
                RETURNING id
                """,
                {
                    "from_id":    from_entity_id,
                    "rel_type":   relation_type,
                    "to_id":      to_entity_id,
                    "article_id": article_id,
                    "delta":      confidence_delta,
                },
            )
            return cur.fetchone()["id"]

    def get_relations_for_entity(self, entity_id: int) -> list[dict]:
        """Return all edges where entity is source or target, with entity names joined."""
        with self._tx() as cur:
            cur.execute(
                """
                SELECT
                    er.id, er.relation_type, er.confidence, er.weight,
                    er.first_seen, er.last_seen,
                    e1.name AS from_name, e1.entity_type AS from_type,
                    e2.name AS to_name,   e2.entity_type AS to_type
                FROM   entity_relations er
                JOIN   entities e1 ON e1.id = er.from_entity_id
                JOIN   entities e2 ON e2.id = er.to_entity_id
                WHERE  er.from_entity_id = %s OR er.to_entity_id = %s
                ORDER  BY er.weight DESC
                """,
                (entity_id, entity_id),
            )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # chunks
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: ChunkRow) -> Optional[int]:
        """Insert a single chunk. Returns chunk id."""
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO chunks (article_id, chunk_index, content, start_char, end_char)
                VALUES (%(article_id)s, %(chunk_index)s, %(content)s, %(start_char)s, %(end_char)s)
                ON CONFLICT (article_id, chunk_index) DO NOTHING
                RETURNING id
                """,
                {
                    "article_id":  chunk.article_id,
                    "chunk_index": chunk.chunk_index,
                    "content":     chunk.content,
                    "start_char":  chunk.start_char,
                    "end_char":    chunk.end_char,
                },
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def bulk_insert_chunks(self, chunks: list[ChunkRow]) -> None:
        """Insert all chunks for an article in a single round-trip."""
        rows = [
            (c.article_id, c.chunk_index, c.content, c.start_char, c.end_char)
            for c in chunks
        ]
        with self._tx() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks (article_id, chunk_index, content, start_char, end_char)
                VALUES %s
                ON CONFLICT (article_id, chunk_index) DO NOTHING
                """,
                rows,
            )

    def get_unembedded_chunks(self, limit: int = 500) -> list[dict]:
        """Fetch chunks not yet pushed to Qdrant. Used by the embedding worker."""
        with self._tx() as cur:
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

    def mark_chunks_embedded(self, chunk_ids: list[int]) -> None:
        """Set is_embedded=true for a batch of chunk ids after Qdrant upsert."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE chunks SET is_embedded = TRUE WHERE id = ANY(%s)",
                (chunk_ids,),
            )

    def all_chunks_embedded(self, article_id: int) -> bool:
        """Return True when every chunk for an article has been embedded."""
        with self._tx() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN is_embedded THEN 1 ELSE 0 END) AS done
                FROM   chunks
                WHERE  article_id = %s
                """,
                (article_id,),
            )
            row = cur.fetchone()
            return row["total"] > 0 and row["total"] == row["done"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="FactLens DB schema tool")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables then redeploy (DESTRUCTIVE — dev only)",
    )
    args = parser.parse_args()

    with Database() as db:
        if args.reset:
            confirm = input("This will DELETE all data. Type 'yes' to continue: ")
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                sys.exit(0)
            db.reset_schema()
        else:
            db.deploy_schema()


if __name__ == "__main__":
    _cli()
