"""
FactLens — db/schema.py

All DDL strings and schema lifecycle methods (deploy / reset).

Design notes
------------
TimescaleDB requires every UNIQUE / PRIMARY KEY on a hypertable to include the
partition column (fetched_at).  To avoid polluting every FK with that column we:

  • Strip inline PRIMARY KEY / UNIQUE from the articles DDL.
  • Add UNIQUE indexes on (id, fetched_at) and (url_hash, fetched_at) instead.
  • Keep a plain index on url_hash alone for fast O(log n) existence checks that
    don't need fetched_at (see ArticleRepository.article_exists).

Dedup strategy recap
--------------------
  1. exists(url)         → fast pre-check on idx_articles_guid_only.
  2. insert_article()    → ON CONFLICT (guid, fetched_at) DO NOTHING as
                           safety-net for same-second races (extremely rare).
"""

from __future__ import annotations

import logging

import psycopg2

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL strings
# ---------------------------------------------------------------------------

_EXTENSION = "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# _FEEDS = """
# CREATE TABLE IF NOT EXISTS feeds (
#     id            SERIAL PRIMARY KEY,
#     url           TEXT        NOT NULL UNIQUE,
#     name          TEXT,
#     category      TEXT,
#     is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
#     last_fetched  TIMESTAMPTZ,
#     fetch_errors  INTEGER     NOT NULL DEFAULT 0,
#     created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
# );
# """

_ARTICLES = """
CREATE TABLE IF NOT EXISTS articles (
    id                BIGSERIAL NOT NULL,

    -- identity / dedup
    url               TEXT        NOT NULL,

    title             TEXT,
    category          TEXT        NOT NULL,
    published_at      TIMESTAMPTZ,
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT NOW() ,
    source_domain     TEXT,
    language          VARCHAR(10) NOT NULL DEFAULT 'en',
    state             TEXT        NOT NULL DEFAULT 'pending',

    -- content
    raw_content       TEXT,

    -- named entities
    entity_person       TEXT[],
    entity_norp         TEXT[],
    entity_fac          TEXT[],
    entity_org          TEXT[],
    entity_gpe          TEXT[],
    entity_loc          TEXT[],
    entity_product      TEXT[],
    entity_event        TEXT[],
    entity_work_of_art  TEXT[],
    entity_law          TEXT[],
    entity_language     TEXT[],
    entity_date         TEXT[],
    entity_time         TEXT[],
    entity_percent      TEXT[],
    entity_money        TEXT[],
    entity_quantity     TEXT[],
    entity_ordinal      TEXT[],
    entity_cardinal     TEXT[],

    -- processing
    processing_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (
            processing_status IN (
                'pending',
                'processing',
                'processed',
                'embedded',
                'failed'
            )
        ),

    processed_at      TIMESTAMPTZ,

    -- arbitrary metadata
    extra             JSONB,

    -- TimescaleDB: UNIQUE must include the partition column (fetched_at).
    -- articles.py uses ON CONFLICT (guid, fetched_at) DO NOTHING.
    UNIQUE (url, fetched_at)
);
"""

_ARTICLES_HYPERTABLE = """
SELECT create_hypertable(
    'articles',
    'fetched_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists       => TRUE,
    migrate_data        => TRUE
);
"""

_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
    id             TEXT        PRIMARY KEY,
    entity_name           TEXT[]        NOT NULL,
    entity_type    TEXT        NOT NULL,
    UNIQUE (entity_name, entity_type)
);
"""

# _ENTITY_MENTIONS = """
# CREATE TABLE IF NOT EXISTS entity_mentions (
#     id            BIGSERIAL   PRIMARY KEY,
#     article_id    BIGINT      NOT NULL,
#     entity_id     INTEGER     NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
#     mention_text  TEXT,
#     mention_count INTEGER     NOT NULL DEFAULT 1,
#     created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
#     UNIQUE (article_id, entity_id)
# );
# """

# NOTE: entity_mentions.article_id has no FK to articles(id).
# PostgreSQL does not allow FKs referencing a hypertable column that is only
# covered by a UNIQUE INDEX (not a UNIQUE CONSTRAINT). Referential integrity is
# maintained at the application layer.

# _ENTITY_RELATIONS = """
# CREATE TABLE IF NOT EXISTS entity_relations (
#     id                   BIGSERIAL PRIMARY KEY,
#     from_entity_id       INTEGER   NOT NULL REFERENCES entities(id),
#     relation_type        TEXT      NOT NULL,
#     to_entity_id         INTEGER   NOT NULL REFERENCES entities(id),
#     confidence           REAL      NOT NULL DEFAULT 0.5,
#     weight               INTEGER   NOT NULL DEFAULT 1,
#     first_seen           TIMESTAMPTZ,
#     last_seen            TIMESTAMPTZ,
#     evidence_article_ids BIGINT[]  NOT NULL DEFAULT '{}',
#     extra                JSONB,
#     UNIQUE (from_entity_id, relation_type, to_entity_id)
# );
# """

_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL   PRIMARY KEY,
    article_id   BIGINT      NOT NULL,
    chunk_index  INTEGER     NOT NULL,
    content      TEXT        NOT NULL,
    is_embedded  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (article_id, chunk_index)
);
"""

# NOTE: chunks.article_id also has no FK for the same reason as entity_mentions.

_INDEXES: list[str] = [
    # TimescaleDB-compatible unique indexes (must include partition column)
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_id_fetched        ON articles (id, fetched_at);",
    # "CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_guid_fetched      ON articles (guid, fetched_at);",
    # Plain index for fast existence checks (no fetched_at needed)
    # "CREATE INDEX        IF NOT EXISTS idx_articles_guid_only         ON articles (guid);",
    # Standard lookup indexes
    "CREATE INDEX        IF NOT EXISTS idx_articles_fetched_at        ON articles (fetched_at);",
    "CREATE INDEX        IF NOT EXISTS idx_articles_published_at      ON articles (published_at);",
    # "CREATE INDEX        IF NOT EXISTS idx_articles_feed_id           ON articles (feed_id);",
    "CREATE INDEX        IF NOT EXISTS idx_articles_processing_status ON articles (processing_status);",
    "CREATE INDEX        IF NOT EXISTS idx_articles_source_domain     ON articles (source_domain);",
    # "CREATE INDEX        IF NOT EXISTS idx_entity_mentions_article_id ON entity_mentions (article_id);",
    # "CREATE INDEX        IF NOT EXISTS idx_entity_mentions_entity_id  ON entity_mentions (entity_id);",
    # "CREATE INDEX        IF NOT EXISTS idx_entity_relations_from      ON entity_relations (from_entity_id);",
    # "CREATE INDEX        IF NOT EXISTS idx_entity_relations_to        ON entity_relations (to_entity_id);",
    # "CREATE INDEX        IF NOT EXISTS idx_chunks_article_id          ON chunks (article_id);",
    # "CREATE INDEX        IF NOT EXISTS idx_chunks_is_embedded         ON chunks (is_embedded);",
]

_DROP_ALL = """
DROP TABLE IF EXISTS entities          CASCADE;
DROP TABLE IF EXISTS articles          CASCADE;

"""


# ---------------------------------------------------------------------------
# SchemaManager
# ---------------------------------------------------------------------------

class SchemaManager:
    """
    Owns DDL execution.  Accepts a raw psycopg2 connection so it stays
    independent of the higher-level connection wrapper.
    """

    def __init__(self, conn: psycopg2.extensions.connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_ddl(self, sql: str, label: str) -> None:
        """Execute a DDL statement in autocommit mode (required by some DDL)."""
        old_autocommit = self._conn.autocommit
        self._conn.autocommit = True
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql)
            log.info("DDL OK — %s", label)
        except psycopg2.Error as exc:
            log.error("DDL FAIL — %s — %s", label, exc.pgerror or exc)
            raise
        finally:
            self._conn.autocommit = old_autocommit

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def deploy(self) -> None:
        """Create all tables, hypertable, and indexes. Idempotent — safe on every startup."""
        log.info("Deploying schema...")
        self._run_ddl(_EXTENSION,           "extension timescaledb")
        # self._run_ddl(_FEEDS,               "table feeds")
        self._run_ddl(_ARTICLES,            "table articles")
        self._run_ddl(_ARTICLES_HYPERTABLE, "hypertable articles")
        self._run_ddl(_ENTITIES,            "table entities")
        # self._run_ddl(_ENTITY_MENTIONS,     "table entity_mentions")
        # self._run_ddl(_ENTITY_RELATIONS,    "table entity_relations")
        self._run_ddl(_CHUNKS,              "table chunks")
        for stmt in _INDEXES:
            label = stmt.split("idx_")[1].split(" ")[0] if "idx_" in stmt else stmt[:50]
            self._run_ddl(stmt, f"index {label}")
        log.info("Schema deployment complete.")

    def reset(self) -> None:
        """DROP all tables then redeploy. DESTRUCTIVE — dev/test only."""
        log.warning("Resetting schema — all data will be lost.")
        self._run_ddl(_DROP_ALL, "drop all tables")
        self.deploy()