# """
# FactLens — db/repositories/relations.py

# DB operations for the `entity_relations` table.
# """

# from __future__ import annotations

# import logging

# from db.connection import Connection

# log = logging.getLogger(__name__)


# class EntityRelationRepository:
#     """Upsert and query for the entity_relations (graph edge) table."""

#     def __init__(self, conn: Connection) -> None:
#         self._conn = conn

#     def upsert(
#         self,
#         from_entity_id: int,
#         relation_type: str,
#         to_entity_id: int,
#         article_id: int,
#         confidence_delta: float = 0.05,
#     ) -> int:
#         """
#         Insert or strengthen a relation edge keyed on
#         (from_entity_id, relation_type, to_entity_id).

#         On conflict:
#           • Increments edge weight by 1.
#           • Bumps confidence by confidence_delta, capped at 1.0.
#           • Appends article_id to evidence_article_ids.
#           • Updates last_seen.

#         Returns relation id.
#         """
#         with self._conn._tx() as cur:
#             cur.execute(
#                 """
#                 INSERT INTO entity_relations
#                     (from_entity_id, relation_type, to_entity_id,
#                      confidence, weight, first_seen, last_seen, evidence_article_ids)
#                 VALUES
#                     (%(from_id)s, %(rel_type)s, %(to_id)s,
#                      0.5, 1, NOW(), NOW(), ARRAY[%(article_id)s::bigint])
#                 ON CONFLICT (from_entity_id, relation_type, to_entity_id) DO UPDATE
#                     SET weight               = entity_relations.weight + 1,
#                         confidence           = LEAST(
#                             entity_relations.confidence + %(delta)s, 1.0
#                         ),
#                         last_seen            = NOW(),
#                         evidence_article_ids = array_append(
#                             entity_relations.evidence_article_ids,
#                             %(article_id)s::bigint
#                         )
#                 RETURNING id
#                 """,
#                 {
#                     "from_id":    from_entity_id,
#                     "rel_type":   relation_type,
#                     "to_id":      to_entity_id,
#                     "article_id": article_id,
#                     "delta":      confidence_delta,
#                 },
#             )
#             return cur.fetchone()["id"]

#     def get_for_entity(self, entity_id: int) -> list[dict]:
#         """
#         Return all edges where entity is the source or target.
#         Joins entity names so callers get human-readable from_name / to_name.
#         Ordered by descending weight.
#         """
#         with self._conn._tx() as cur:
#             cur.execute(
#                 """
#                 SELECT
#                     er.id,
#                     er.relation_type,
#                     er.confidence,
#                     er.weight,
#                     er.first_seen,
#                     er.last_seen,
#                     e1.name        AS from_name,
#                     e1.entity_type AS from_type,
#                     e2.name        AS to_name,
#                     e2.entity_type AS to_type
#                 FROM   entity_relations er
#                 JOIN   entities e1 ON e1.id = er.from_entity_id
#                 JOIN   entities e2 ON e2.id = er.to_entity_id
#                 WHERE  er.from_entity_id = %s
#                     OR er.to_entity_id   = %s
#                 ORDER  BY er.weight DESC
#                 """,
#                 (entity_id, entity_id),
#             )
#             return cur.fetchall()
