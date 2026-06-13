"""
FactLens — db/repositories/entities.py

DB operations for the `entities` and `entity_mentions` tables.

Schema assumptions
------------------
    CREATE TABLE entities (
        id           TEXT PRIMARY KEY,          -- e.g. "ORG_11"
        entity_name  TEXT[]  NOT NULL,          -- array of known aliases
        entity_type  TEXT    NOT NULL           -- spaCy label: ORG, PERSON, …
    );

The ``entity_name`` column is a Postgres TEXT array so that every alias
for a canonical entity is stored in one row. All read paths that need to
do fuzzy matching should call ``get_by_type()``, which unnests the array
and returns one row per alias — making the match loop trivial.
"""

from __future__ import annotations

import logging
from typing import Optional

from db.connection import Connection
from db.models import EntityRow

log = logging.getLogger(__name__)

# All valid spaCy NER labels. Used to validate entity_type on insert/upsert
# so nothing outside the 18-label OntoNotes schema can enter the table.
VALID_ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
    "ORDINAL", "CARDINAL",
})


class EntityRepository:
    """Insert / upsert / lookup for the entities table."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_type(entity_type: str) -> None:
        """Raise ValueError if entity_type is not one of the 18 NER labels."""
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type '{entity_type}'. "
                f"Must be one of: {sorted(VALID_ENTITY_TYPES)}"
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_by_type(self, entity_type: str) -> list[dict]:
        """
        Fetch all entities of ``entity_type`` with their alias arrays
        **unnested** — one dict per alias, all sharing the same ``id``.

        This is the primary read path for the entity-resolution fuzzy
        blocking step: by returning one row per alias, the matching loop
        can call ``normalise_entity_text(row["entity_name"])`` directly
        without unpacking arrays.

        Parameters
        ----------
        entity_type : str
            A valid spaCy NER label, e.g. ``"ORG"``.

        Returns
        -------
        list[dict]
            Each dict has keys ``id``, ``entity_name``, ``entity_type``.
            Returns an empty list when no entities of that type exist.
        """
        self._validate_type(entity_type)

        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    unnest(entity_name) AS entity_name,
                    entity_type
                FROM entities
                WHERE entity_type = %s
                ORDER BY id
                """,
                (entity_type,),
            )
            rows = cur.fetchall()

        result = []
        for row in rows:
            try:
                result.append({
                    "id": row["id"],
                    "entity_name": row["entity_name"],
                    "entity_type": row["entity_type"],
                })
            except (TypeError, KeyError):
                result.append({
                    "id": row[0],
                    "entity_name": row[1],
                    "entity_type": row[2],
                })
        return result

    def get_by_id(self, entity_id: str) -> Optional[dict]:
        """Return a single entity row by its canonical ID, or None."""
        with self._conn._tx() as cur:
            cur.execute(
                "SELECT id, entity_name, entity_type FROM entities WHERE id = %s",
                (entity_id,),
            )
            return cur.fetchone()

    def get_by_name(self, name: str, entity_type: str) -> Optional[dict]:
        """
        Lookup a single entity whose alias array contains ``name``,
        scoped to ``entity_type``.
        """
        self._validate_type(entity_type)
        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT id, entity_name, entity_type
                FROM entities
                WHERE %s = ANY(entity_name)
                  AND entity_type = %s
                """,
                (name, entity_type),
            )
            return cur.fetchone()

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def get_next_canonical_id(self, entity_type: str) -> str:
        """
        Generate the next sequential canonical ID for ``entity_type``.

        Format: ``{TYPE}_{N}`` — e.g. ``ORG_1``, ``ORG_12``, ``PERSON_3``.

        The sequence number is derived from the MAX integer suffix across
        all existing IDs of that type.  ``COALESCE`` to 0 handles an
        empty table (so the very first entity gets ``{TYPE}_1``).

        Parameters
        ----------
        entity_type : str
            A valid spaCy NER label.

        Returns
        -------
        str
            Next available canonical ID string.
        """
        self._validate_type(entity_type)

        with self._conn._tx() as cur:
            cur.execute(
                """
                SELECT COALESCE(
                    MAX(CAST(SPLIT_PART(id, '_', 2) AS INTEGER)),
                    0
                ) AS max_seq
                FROM entities
                WHERE entity_type = %s
                """,
                (entity_type,),
            )
            row = cur.fetchone()

        try:
            max_seq: int = row["max_seq"]
        except (TypeError, KeyError):
            max_seq = row[0] if row else 0

        return f"{entity_type}_{max_seq + 1}"

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert(self, entity_id: str, entity_name: str, entity_type: str) -> None:
        """
        Insert a brand-new entity row.

        ``entity_name`` is stored as a single-element TEXT array; additional
        aliases are added later via ``append_entity_name()``.

        Parameters
        ----------
        entity_id : str
            Pre-generated canonical ID, e.g. ``"ORG_11"``.
        entity_name : str
            The initial (display) name for the entity.
        entity_type : str
            A valid spaCy NER label.
        """
        self._validate_type(entity_type)

        with self._conn._tx() as cur:
            cur.execute(
                """
                INSERT INTO entities (id, entity_name, entity_type)
                VALUES (%s, ARRAY[%s], %s)
                """,
                (entity_id, entity_name, entity_type),
            )

        log.debug("insert: new entity id=%s name=%s type=%s", entity_id, entity_name, entity_type)

    def upsert(self, entity: EntityRow) -> str:
        """
        Insert a new entity row, or do nothing if the canonical ID already
        exists (race-condition safety net for concurrent workers).

        On conflict the existing row is left untouched; the caller should
        use ``append_entity_name()`` separately if it wants to add a new
        alias to an existing entity.

        Parameters
        ----------
        entity : EntityRow
            Value object with ``id``, ``entity_name``, and ``entity_type``.

        Returns
        -------
        str
            The canonical ID that is live in the DB after this call.
        """
        self._validate_type(entity.entity_type)

        with self._conn._tx() as cur:
            cur.execute(
                """
                INSERT INTO entities (id, entity_name, entity_type)
                VALUES (%s, ARRAY[%s], %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                (entity.id, entity.entity_name, entity.entity_type),
            )
            row = cur.fetchone()

        if row is None:
            # Row already existed — conflict fired, RETURNING returned nothing.
            log.debug("upsert: entity id=%s already exists — skipped.", entity.id)
            return entity.id

        try:
            canonical_id: str = row["id"]
        except (TypeError, KeyError):
            canonical_id = row[0]

        log.debug("upsert: inserted entity id=%s type=%s", canonical_id, entity.entity_type)
        return canonical_id

    def append_entity_name(self, entity_id: str, new_name: str) -> None:
        """
        Add ``new_name`` to the ``entity_name`` array for the given entity,
        deduplicating via ``DISTINCT unnest``.

        This is a no-op (no error, no duplicate) if ``new_name`` is already
        present in the array.

        Parameters
        ----------
        entity_id : str
            Canonical ID of the entity to update, e.g. ``"ORG_11"``.
        new_name : str
            The new alias to append, e.g. ``"Apple Inc."``.
        """
        with self._conn._tx() as cur:
            cur.execute(
                """
                UPDATE entities
                SET entity_name = (
                    SELECT ARRAY(
                        SELECT DISTINCT unnest(entity_name || ARRAY[%s])
                    )
                )
                WHERE id = %s
                """,
                (new_name, entity_id),
            )

        log.debug("append_entity_name: entity_id=%s ← '%s'", entity_id, new_name)

    def update(
        self,
        entity_id: str,
        entity_name: Optional[list[str]] = None,
        entity_type: Optional[str] = None,
    ) -> None:
        """
        Overwrite fields of an existing entity row.

        Only non-None arguments are written; passing both as None is a no-op.

        Parameters
        ----------
        entity_id : str
            Canonical ID of the entity to update.
        entity_name : list[str], optional
            New full alias array (replaces the existing array entirely).
            Use ``append_entity_name()`` to add a single alias instead.
        entity_type : str, optional
            New entity type (must be one of the 18 valid labels).
        """
        if entity_type is not None:
            self._validate_type(entity_type)

        updates: list[str] = []
        params: list = []

        if entity_name is not None:
            updates.append("entity_name = %s")
            params.append(entity_name)   # Postgres will accept a Python list as TEXT[]

        if entity_type is not None:
            updates.append("entity_type = %s")
            params.append(entity_type)

        if not updates:
            log.debug("update: nothing to update for entity_id=%s", entity_id)
            return

        params.append(entity_id)

        with self._conn._tx() as cur:
            cur.execute(
                f"UPDATE entities SET {', '.join(updates)} WHERE id = %s",
                params,
            )

        log.debug("update: entity_id=%s fields=%s", entity_id, updates)
