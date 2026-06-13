"""
FactLens — db/repositories/__init__.py

Re-export all repository classes for convenient import:

    from db.repositories import FeedRepository, ArticleRepository, ...
"""

from db.repositories.articles import ArticleRepository
from db.repositories.chunks import ChunkRepository
from db.repositories.entities import EntityRepository
# from db.repositories.feeds import FeedRepository
# from db.repositories.relations import EntityRelationRepository

__all__ = [
    "ArticleRepository",
    "EntityRepository",
    # "EntityMentionRepository",
    # "EntityRelationRepository",
    # "ChunkRepository",
]