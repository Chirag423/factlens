"""
FactLens — db/database.py

Database facade.

Composes Connection, SchemaManager, and all Repositories into a single
entry point.  Callers that previously imported `Database` from `db.py`
can import it from here with no API changes.

Usage
-----
    from db.database import Database

    with Database() as db:
        db.deploy_schema()

        feed_id  = db.feeds.insert(FeedRow(url="...", name="NDTV"))
        art_id   = db.articles.insert(ArticleRow(url="...", raw_content="..."))
        ent_id   = db.entities.upsert(EntityRow(name="Modi", entity_type="PERSON"))
        db.mentions.insert(article_id=art_id, entity_id=ent_id, mention_text="PM Modi")
        db.relations.upsert(from_entity_id=1, relation_type="MEMBER_OF",
                            to_entity_id=2, article_id=art_id)
        chunk_id = db.chunks.insert(ChunkRow(article_id=art_id, chunk_index=0, content="..."))
        db.chunks.mark_embedded([chunk_id])

Design
------
The facade owns no SQL.  It delegates every operation to the appropriate
repository.  This keeps the class thin and makes individual repositories
unit-testable in isolation by injecting a mock Connection.
"""

from __future__ import annotations

import logging
from typing import Optional

from db.connection import Connection
from db.repositories import (
    ArticleRepository,
    EntityRepository,
)
from db.repositories.chunks import ChunkRepository
from db.schema import SchemaManager

log = logging.getLogger(__name__)


class Database:
    """
    Single entry point for all FactLens DB operations.

    Attributes (public repositories)
    ---------------------------------
    feeds     — FeedRepository
    articles  — ArticleRepository
    entities  — EntityRepository
    mentions  — EntityMentionRepository
    relations — EntityRelationRepository
    chunks    — ChunkRepository
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._conn    = Connection(database_url)
        self._schema  = SchemaManager(self._conn.raw)

        # Public repository attributes — callers use these directly
        # self.feeds     = FeedRepository(self._conn)
        self.articles  = ArticleRepository(self._conn)
        self.entities  = EntityRepository(self._conn)
        # self.mentions  = EntityMentionRepository(self._conn)
        # self.relations = EntityRelationRepository(self._conn)
        self.chunks    = ChunkRepository(self._conn)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Schema lifecycle (thin delegation to SchemaManager)
    # ------------------------------------------------------------------

    def deploy_schema(self) -> None:
        """Create all tables, hypertable, and indexes. Idempotent — safe on every startup."""
        self._schema.deploy()

    def reset_schema(self) -> None:
        """DROP all tables then redeploy. DESTRUCTIVE — dev/test only."""
        self._schema.reset()
