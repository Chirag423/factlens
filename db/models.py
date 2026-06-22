"""
FactLens — db/models.py

Plain data-transfer objects (dataclasses) for every DB table.
No database imports — importable anywhere without a DB connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ArticleRow:
    url: str
    raw_content: str
    category: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    title: Optional[str] = None
    published_at: Optional[datetime] = None
    source_domain: Optional[str] = None
    language: str = "en"
    state: str = "pending"

    entity_person:       Optional[list[str]] = None
    entity_norp:         Optional[list[str]] = None
    entity_fac:          Optional[list[str]] = None
    entity_org:          Optional[list[str]] = None
    entity_gpe:          Optional[list[str]] = None
    entity_loc:          Optional[list[str]] = None
    entity_product:      Optional[list[str]] = None
    entity_event:        Optional[list[str]] = None
    entity_work_of_art:  Optional[list[str]] = None
    entity_law:          Optional[list[str]] = None
    entity_language:     Optional[list[str]] = None
    entity_date:         Optional[list[str]] = None
    entity_time:         Optional[list[str]] = None
    entity_percent:      Optional[list[str]] = None
    entity_money:        Optional[list[str]] = None
    entity_quantity:     Optional[list[str]] = None
    entity_ordinal:      Optional[list[str]] = None
    entity_cardinal:     Optional[list[str]] = None

    extra: Optional[dict] = None


@dataclass
class EntityRow:
    id: str
    entity_name: list[str]
    entity_type: str                # PERSON | ORG | GPE | LAW | EVENT


@dataclass
class ChunkRow:
    """
    Child-level (sentence) chunk — stored in Qdrant, not in PostgreSQL.
    This dataclass is retained for in-memory bookkeeping during the pipeline.
    """
    parent_id:  str
    chunk_index: int
    content:     str


@dataclass
class ParentChunkRow:
    """
    Parent-level (paragraph) chunk — persisted in the parent_chunks PG table.
    parent_id is a deterministic UUID-v5 string; it is also stored as the
    ``parent_id`` field in every Qdrant child-chunk payload for retrieval.
    """
    parent_id:   str   # uuid5(article_id, chunk_index) — never changes between runs
    article_id:  int
    chunk_index: int
    content:     str
