# """
# FactLens — db/repositories/feeds.py

# All DB operations for the `feeds` table.
# """

# from __future__ import annotations

# import logging
# from typing import Optional

# from db.connection import Connection
# from db.models import FeedRow

# log = logging.getLogger(__name__)


# class FeedRepository:
#     """CRUD + helpers for the feeds table."""

#     def __init__(self, conn: Connection) -> None:
#         self._conn = conn

#     def insert(self, feed: FeedRow) -> int:
#         """
#         Upsert a feed by URL.
#         On conflict: refreshes name, category, and is_active.
#         Returns the feed id (existing or new).
#         """
#         with self._conn._tx() as cur:
#             cur.execute(
#                 """
#                 INSERT INTO feeds (url, name, category, is_active)
#                 VALUES (%(url)s, %(name)s, %(category)s, %(is_active)s)
#                 ON CONFLICT (url) DO UPDATE
#                     SET name      = EXCLUDED.name,
#                         category  = EXCLUDED.category,
#                         is_active = EXCLUDED.is_active
#                 RETURNING id
#                 """,
#                 {
#                     "url":       feed.url,
#                     "name":      feed.name,
#                     "category":  feed.category,
#                     "is_active": feed.is_active,
#                 },
#             )
#             feed_id: int = cur.fetchone()["id"]
#             log.debug("feed id=%d url=%s", feed_id, feed.url)
#             return feed_id

#     def mark_fetched(self, feed_id: int) -> None:
#         """Set last_fetched = NOW() and reset fetch_errors to 0."""
#         with self._conn._tx() as cur:
#             cur.execute(
#                 "UPDATE feeds SET last_fetched = NOW(), fetch_errors = 0 WHERE id = %s",
#                 (feed_id,),
#             )

#     def increment_error(self, feed_id: int) -> int:
#         """Bump fetch_errors by 1. Returns the new count (useful for circuit-breaker logic)."""
#         with self._conn._tx() as cur:
#             cur.execute(
#                 """
#                 UPDATE feeds
#                 SET fetch_errors = fetch_errors + 1
#                 WHERE id = %s
#                 RETURNING fetch_errors
#                 """,
#                 (feed_id,),
#             )
#             return cur.fetchone()["fetch_errors"]

#     def get_active(self) -> list[dict]:
#         """Return all feeds where is_active = TRUE, ordered by id."""
#         with self._conn._tx() as cur:
#             cur.execute("SELECT * FROM feeds WHERE is_active = TRUE ORDER BY id")
#             return cur.fetchall()
