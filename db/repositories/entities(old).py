"""
FactLens — db/repositories/entities.py

DB operations for the `entities` and `entity_mentions` tables.
Split into two focused repository classes following Single Responsibility.
"""

from __future__ import annotations

import logging
from typing import Optional

import psycopg2.extras

from db.connection import Connection
from db.models import EntityRow

log = logging.getLogger(__name__)


class EntityRepository:
    """Insert / upsert / lookup for the entities table."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn


    def append_entity_name(
    self,
    entity_id: str,
    new_name: str
    ) -> None:
        with self._conn._tx() as cur:
            cur.execute(
                """
                UPDATE entities
                SET entity_name = (
                    SELECT ARRAY(
                        SELECT DISTINCT unnest(
                            entity_name || ARRAY[%s]
                        )
                    )
                )
                WHERE id = %s
                """,
                (new_name, entity_id),
            )


    def insert(
    self,
    entity_id: str,
    entity_name: str,
    entity_type: str
    ) -> None:
        with self._conn._tx() as cur:
            cur.execute(
                """
                INSERT INTO entities (
                    id,
                    entity_name,
                    entity_type
                )
                VALUES (
                    %s,
                    ARRAY[%s],
                    %s
                )
                """,
                (
                    entity_id,
                    entity_name,
                    entity_type,
                ),
        )


    def upsert(self, entity: EntityRow) -> str:
        """
        Insert or update an entity keyed on (id,entity_name, entity_type).

        On conflict:
          • Merges aliases arrays (deduplicates with array_agg + DISTINCT).
          • Bumps mention_count by 1.
          • Updates last_seen to NOW().

        Returns entity id.
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                INSERT INTO entities
                    (id, name, entity_type)
                VALUES
                    (%(id)s, %(entity_name)s, %(entity_type)s)
                ON CONFLICT (entity_namename, entity_type)
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        entity_type = EXCLUDED.entity_type,
                """,
                {   
                    "id": entity.id,
                    "entity_name":        entity.entity_name,
                    "entity_type": entity.entity_type,
                },
            )
            return cur.fetchone()["id"]

    def get_by_name(self, name: str, entity_type: str) -> Optional[dict]:
        """Lookup a single entity by its canonical name and type."""
        with self._conn._tx() as cur:
            cur.execute(
                "SELECT * FROM entities WHERE name = %s AND entity_type = %s",
                (name, entity_type),
            )
            return cur.fetchone()





