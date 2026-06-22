"""
FactLens — rag/relation_extractor.py

Relation Extraction worker using the Sarvam AI SDK.

Execution model
---------------
* Async-first — uses ``AsyncSarvamAI`` so it fits naturally inside the
  existing asyncio / Celery pipeline.
* A sync wrapper ``extract_relations_sync`` is provided for Celery tasks
  that run in a regular thread pool.

Algorithm
---------
1. Call ``prompts.relation_extraction.build_messages()`` to build the
   system + user message pair.
2. Send to ``AsyncSarvamAI.chat.completions()`` with ``sarvam-m`` (fast,
   cost-efficient) or ``sarvam-105b`` (higher accuracy) depending on the
   ``model`` kwarg.
3. Parse the JSON graph via ``parse_graph_response()``.
4. Run a post-processing pass (``_reconcile_nodes``) that fuzzy-matches
   every node's ``name`` field against your ``EntityRepository`` and
   replaces the model-assigned IDs (PERSON_1, ORG_2 …) with your stable
   canonical IDs so downstream Neo4j / PostgreSQL writes stay consistent.
5. Return a ``RelationGraph`` dataclass — typed, Neo4j-ready.

Dependencies
------------
    pip install sarvamai rapidfuzz
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from sarvamai import AsyncSarvamAI, SarvamAI
from sarvamai.core.api_error import ApiError

from db.repositories.entities import EntityRepository
from entity_resolution import normalise_entity_text, _find_best_match, MATCH_THRESHOLD
from prompts.relation_extraction import build_messages, parse_graph_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default model — sarvam-m is fast and cheap; swap to sarvam-105b for
#: higher accuracy on complex political articles.
DEFAULT_MODEL: str = os.getenv("SARVAM_RELATION_MODEL", "sarvam-m")

#: Max output tokens.  A dense article with 30 entities rarely needs more.
MAX_TOKENS: int = int(os.getenv("SARVAM_RELATION_MAX_TOKENS", "4096"))

#: Temperature — keep low for structured JSON extraction.
TEMPERATURE: float = float(os.getenv("SARVAM_RELATION_TEMPERATURE", "0.1"))

#: wiki_grounding — Sarvam-specific feature; helps anchor Indian entity names.
WIKI_GROUNDING: bool = os.getenv("SARVAM_WIKI_GROUNDING", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A single entity node, ready to MERGE into Neo4j."""
    model_id:     str            # original PERSON_1 / ORG_2 … from the model
    canonical_id: str            # stable ID from EntityRepository (e.g. ORG_11)
    label:        str            # spaCy OntoNotes type  (e.g. "ORG")
    name:         str            # canonical surface form
    aliases:      list[str] = field(default_factory=list)

    def to_neo4j_props(self) -> dict[str, Any]:
        return {
            "id":      self.canonical_id,
            "label":   self.label,
            "name":    self.name,
            "aliases": self.aliases,
        }


@dataclass
class GraphRelationship:
    """A directed relationship between two graph nodes."""
    rel_id:    str            # R1, R2 … (article-scoped)
    rel_type:  str            # SCREAMING_SNAKE_CASE relationship type
    source_id: str            # canonical_id of source node
    target_id: str            # canonical_id of target node
    role:      Optional[str]  # e.g. "Finance Minister", "defendant"
    since:     Optional[str]  # ISO-8601 or None
    until:     Optional[str]  # ISO-8601 or None
    inferred:  bool           # True → implicit / contextually deduced
    evidence:  Optional[str]  # supporting sentence (required when inferred)
    custom:    bool           # True → relationship type not in predefined list

    def to_neo4j_props(self) -> dict[str, Any]:
        return {
            "id":       self.rel_id,
            "role":     self.role,
            "since":    self.since,
            "until":    self.until,
            "inferred": self.inferred,
            "evidence": self.evidence,
            "custom":   self.custom,
        }


@dataclass
class RelationGraph:
    """
    Complete graph for one article — nodes + relationships, both reconciled
    against the canonical entity store.
    """
    nodes:           list[GraphNode]
    relationships:   list[GraphRelationship]
    primary_entity_ids: list[str]         # canonical IDs of top 3–5 entities
    dominant_domain: Optional[str]        # political | financial | …
    article_id:      Optional[str] = None # set by caller if known

    # ── Convenience helpers ──────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    def cypher_merge_nodes(self) -> list[str]:
        """
        Generate one ``MERGE`` statement per node.

        Usage::

            driver = GraphDatabase.driver(URI, auth=AUTH)
            with driver.session() as session:
                for stmt in graph.cypher_merge_nodes():
                    session.run(stmt)
        """
        stmts: list[str] = []
        for node in self.nodes:
            props = node.to_neo4j_props()
            stmts.append(
                f"MERGE (n:{node.label} {{id: $id}}) "
                f"SET n += $props",
            )
        return stmts

    def cypher_merge_relationships(self) -> list[str]:
        """
        Generate one ``MERGE`` statement per relationship.

        This requires the nodes to already exist in the graph.
        """
        stmts: list[str] = []
        for rel in self.relationships:
            stmts.append(
                f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
                f"MERGE (a)-[r:{rel.rel_type} {{id: $id}}]->(b) "
                f"SET r += $props"
            )
        return stmts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reconcile_nodes(
    raw_nodes: list[dict[str, Any]],
    repo: EntityRepository,
) -> dict[str, GraphNode]:
    """
    Replace model-assigned IDs (PERSON_1, ORG_2 …) with stable canonical IDs
    from ``EntityRepository``.

    Strategy (mirrors entity_resolution.py):
    * Fetch the in-memory alias block for each label via ``get_by_type()``.
    * Run ``_find_best_match()`` (token_sort_ratio ≥ 85) against the block.
    * On hit  → use the existing canonical ID, append new alias.
    * On miss → upsert a new entity, use the returned canonical ID.

    Parameters
    ----------
    raw_nodes : list[dict]
        The ``nodes`` list straight from ``parse_graph_response()``.
    repo : EntityRepository
        Live repository instance.

    Returns
    -------
    dict[str, GraphNode]
        Mapping of model_id → GraphNode (with canonical_id filled in).
    """
    # Cache per-label blocks so we only hit the DB once per label.
    block_cache: dict[str, list[dict[str, str]]] = {}
    reconciled: dict[str, GraphNode] = {}

    for raw in raw_nodes:
        model_id: str = raw["id"]
        label: str    = raw["label"]
        name: str     = raw["name"]
        aliases: list[str] = raw.get("aliases", [])

        # Load block for this label (once per label per article).
        if label not in block_cache:
            block_cache[label] = repo.get_by_type(label)

        block = block_cache[label]
        normalised = normalise_entity_text(name)

        match = _find_best_match(normalised, block) if normalised else None

        if match is not None:
            canonical_id = match["id"]
            # Register new surface forms as aliases so future articles match.
            repo.append_entity_name(canonical_id, name)
            for alias in aliases:
                repo.append_entity_name(canonical_id, alias)
            # Extend the in-memory block with the new surface forms.
            block.append({"id": canonical_id, "entity_name": name, "entity_type": label})
            logger.debug(
                "_reconcile_nodes: '%s' → existing canonical_id=%s", name, canonical_id
            )
        else:
            # Import here to avoid circular — only needed on miss path.
            from db.models import EntityRow
            from entity_resolution import MATCH_THRESHOLD as _T  # noqa: F401

            canonical_id = repo.get_next_canonical_id(label)
            new_entity = EntityRow(
                id=canonical_id,
                entity_name=name,
                entity_type=label,
            )
            canonical_id = repo.upsert(new_entity)
            block.append({"id": canonical_id, "entity_name": name, "entity_type": label})
            logger.info(
                "_reconcile_nodes: '%s' inserted as new entity canonical_id=%s",
                name, canonical_id,
            )

        reconciled[model_id] = GraphNode(
            model_id=model_id,
            canonical_id=canonical_id,
            label=label,
            name=name,
            aliases=aliases,
        )

    return reconciled


def _build_graph(
    raw: dict[str, Any],
    node_map: dict[str, GraphNode],
) -> RelationGraph:
    """
    Convert parsed JSON + reconciled node map into a typed ``RelationGraph``.

    Relationships whose source or target node is missing from ``node_map``
    are skipped with a warning (should not happen on well-formed model output).
    """
    relationships: list[GraphRelationship] = []

    for r in raw.get("relationships", []):
        source_model_id = r.get("source", "")
        target_model_id = r.get("target", "")

        source_node = node_map.get(source_model_id)
        target_node = node_map.get(target_model_id)

        if source_node is None or target_node is None:
            logger.warning(
                "_build_graph: relationship %s references unknown node "
                "(source=%s, target=%s) — skipping.",
                r.get("id"), source_model_id, target_model_id,
            )
            continue

        relationships.append(GraphRelationship(
            rel_id=r.get("id", ""),
            rel_type=r.get("type", "MENTIONED_WITH"),
            source_id=source_node.canonical_id,
            target_id=target_node.canonical_id,
            role=r.get("role") or None,
            since=r.get("since") or None,
            until=r.get("until") or None,
            inferred=bool(r.get("inferred", False)),
            evidence=r.get("evidence") or None,
            custom=bool(r.get("custom", False)),
        ))

    meta = raw.get("article_meta", {})
    primary_model_ids: list[str] = meta.get("primary_entities", [])
    primary_canonical_ids: list[str] = [
        node_map[mid].canonical_id
        for mid in primary_model_ids
        if mid in node_map
    ]

    return RelationGraph(
        nodes=list(node_map.values()),
        relationships=relationships,
        primary_entity_ids=primary_canonical_ids,
        dominant_domain=meta.get("dominant_domain") or None,
    )


# ---------------------------------------------------------------------------
# Async extractor
# ---------------------------------------------------------------------------

async def extract_relations_async(
    article_text: str,
    repo: EntityRepository,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    wiki_grounding: bool = WIKI_GROUNDING,
    api_key: Optional[str] = None,
) -> RelationGraph:
    """
    Extract entities and relationships from ``article_text`` using Sarvam AI.

    This is the primary entry point for async callers (e.g. the asyncio
    ingestion pipeline or an async Celery task).

    Parameters
    ----------
    article_text : str
        Plain-text body of the article (not HTML).
    repo : EntityRepository
        Live repository used for canonical ID reconciliation.
    model : str
        Sarvam model ID.  ``"sarvam-m"`` (fast/cheap) or
        ``"sarvam-105b"`` (higher accuracy for complex articles).
    max_tokens : int
        Max completion tokens.
    temperature : float
        Sampling temperature — keep ≤ 0.2 for structured JSON output.
    wiki_grounding : bool
        Enable Sarvam's wiki grounding for Indian entity names.
    api_key : str, optional
        Sarvam API subscription key.  Defaults to the
        ``SARVAM_API_KEY`` environment variable.

    Returns
    -------
    RelationGraph
        Typed graph with stable canonical IDs, ready for Neo4j ingestion.

    Raises
    ------
    ApiError
        Propagated from the Sarvam SDK on HTTP / auth failures.
    ValueError
        If the model returns empty or structurally invalid JSON.
    json.JSONDecodeError
        If the model returns non-parseable content.
    """
    key = api_key or os.environ.get("SARVAM_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Sarvam AI API key not found.  Set the SARVAM_API_KEY "
            "environment variable or pass api_key= explicitly."
        )

    messages = build_messages(article_text)

    client = AsyncSarvamAI(api_subscription_key=key)

    logger.info(
        "extract_relations_async: calling Sarvam AI (model=%s, "
        "len(article)=%d chars).",
        model, len(article_text),
    )

    try:
        response = await client.chat.completions(
            messages=messages,       # type: ignore[arg-type]
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            wiki_grounding=wiki_grounding,
            stream=False,
        )
    except ApiError as exc:
        logger.error(
            "Sarvam AI API error: status=%s body=%s", exc.status_code, exc.body
        )
        raise

    raw_content: str = response.choices[0].message.content or ""
    logger.debug("Sarvam AI raw response (first 500 chars): %s", raw_content[:500])

    raw_graph = parse_graph_response(raw_content)

    # Reconcile model-assigned node IDs → canonical entity IDs.
    node_map = _reconcile_nodes(raw_graph["nodes"], repo)

    graph = _build_graph(raw_graph, node_map)
    graph.article_id = None  # caller should set this after the call

    logger.info(
        "extract_relations_async complete: %d nodes, %d relationships, "
        "domain=%s.",
        graph.node_count,
        graph.relationship_count,
        graph.dominant_domain,
    )
    return graph


# ---------------------------------------------------------------------------
# Sync wrapper (for Celery tasks / non-async callers)
# ---------------------------------------------------------------------------

def extract_relations_sync(
    article_text: str,
    repo: EntityRepository,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    wiki_grounding: bool = WIKI_GROUNDING,
    api_key: Optional[str] = None,
) -> RelationGraph:
    """
    Synchronous wrapper around ``extract_relations_async``.

    Suitable for Celery tasks that do not run inside an asyncio event loop.
    If called from within a running loop, use ``extract_relations_async``
    directly via ``await``.

    Parameters and return value are identical to ``extract_relations_async``.
    """
    return asyncio.run(
        extract_relations_async(
            article_text,
            repo,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            wiki_grounding=wiki_grounding,
            api_key=api_key,
        )
    )