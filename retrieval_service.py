"""
FactLens — retrieval/service.py

RetrievalService
================
Accepts a natural-language query, performs a Hybrid RRF search over
Qdrant's ``factlens_child_chunks`` collection (dense BGE-M3 semantic
search + sparse BGE-M3 lexical/syntactic search), then fetches the full
parent-paragraph texts from PostgreSQL and assembles a ready-to-inject
LLM context block with citations.

Pipeline
────────
  query : str
    │
    ▼
  BGE-M3.encode()
    ├── dense_vec      (1024-dim cosine)           — semantic meaning
    └── lexical_weights (dict[token_id, float])    — syntactic/lexical signal
    │
    ▼
  Qdrant client.query_points()  ← modern API (not .search())
    prefetch[0]: dense   vector key ""              top-PREFETCH_LIMIT
    prefetch[1]: sparse  vector key "hybrid-sparse" top-PREFETCH_LIMIT
    fusion     : RRF  →  final top-FINAL_LIMIT child chunks
    │
    ▼
  Extract + deduplicate parent_ids from Qdrant payload
    (preserves RRF rank order — best child chunk wins when two children
     share the same parent paragraph)
    │
    ├──▶  ParentChunkRepository.get_by_parent_ids()  →  parent texts (PG)
    └──▶  ArticleRepository.get_urls_by_ids()         →  source URLs   (PG)
    │
    ▼
  dict
    ├── assembled_context : str    (paragraphs joined by ---)
    └── citations         : list[{parent_id, article_id, source_url}]

Integration
───────────
  Uses the project's shared modules:
    db.connection                 → Connection (psycopg2 wrapper)
    db.repositories.articles      → ArticleRepository
    db.repositories.chunks        → ParentChunkRepository
  External:
    FlagEmbedding                 → BGEM3FlagModel
    qdrant_client                 → QdrantClient, models

Environment variables
─────────────────────
  BGE_MODEL    BGE-M3 HuggingFace path/ID  (default: BAAI/bge-m3)
  QDRANT_HOST  Qdrant host                 (default: localhost)
  QDRANT_PORT  Qdrant port                 (default: 6333)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from db.connection import Connection
from db.repositories.articles import ArticleRepository
from db.repositories.chunks import ParentChunkRepository


# ═══════════════════════════════════════════════════════════════════════════
# Module logger
# ═══════════════════════════════════════════════════════════════════════════

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME: str = "factlens_child_chunks"

# ── Search knobs ─────────────────────────────────────────────────────────────
# Candidates fetched per branch before RRF fusion.
# Higher = better recall at the cost of a slightly larger fusion window.
PREFETCH_LIMIT: int = 20

# Final number of child chunks returned by RRF.
# Each unique parent_id among these becomes one context paragraph.
FINAL_LIMIT: int = 5

# ── Model ────────────────────────────────────────────────────────────────────
BGE_MODEL_NAME: str = os.getenv("BGE_MODEL", "BAAI/bge-m3")

# ── Output formatting ─────────────────────────────────────────────────────────
# Separator injected between parent paragraphs in assembled_context.
# The triple-dash is a lightweight Markdown horizontal rule that most LLMs
# recognise as a section boundary without adding noise to plain-text prompts.
_CONTEXT_SEPARATOR: str = "\n\n---\n\n"


# ═══════════════════════════════════════════════════════════════════════════
# RetrievalService
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalService:
    """
    Stateless-query, stateful-init service for hybrid RAG retrieval.

    The BGE-M3 model and the Qdrant client are both expensive to initialise
    (~2–4 s on first load).  Inject them at construction time so they are
    created once (at application startup) and shared across all retrieve()
    calls.

    Parameters
    ──────────
    embed_model    : Loaded BGEM3FlagModel (use_fp16=True recommended).
    qdrant_client  : Connected QdrantClient.
    conn           : db.connection.Connection — shared PostgreSQL connection.

    Example
    ───────
    ::

        from FlagEmbedding import BGEM3FlagModel
        from qdrant_client import QdrantClient

        from db.connection import Connection
        from retrieval.service import RetrievalService, init_retrieval_service

        svc = init_retrieval_service()          # convenience factory
        result = svc.retrieve("Who funded the 2024 disinformation campaign?")

        print(result["assembled_context"])
        for cite in result["citations"]:
            print(cite["source_url"])
    """

    def __init__(
        self,
        embed_model:   BGEM3FlagModel,
        qdrant_client: QdrantClient,
        conn:          Connection,
    ) -> None:
        self._model        = embed_model
        self._qdrant       = qdrant_client
        self._article_repo = ArticleRepository(conn)
        self._chunk_repo   = ParentChunkRepository(conn)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Run the full hybrid retrieval pipeline for *query*.

        The method is intentionally side-effect-free (no writes) so it is
        safe to call concurrently from multiple threads sharing the same
        RetrievalService instance, as long as the underlying psycopg2
        Connection is thread-safe (which it is when each call enters its own
        ``_tx()`` context manager).

        Parameters
        ──────────
        query : str
            Raw natural-language query from the user interface.

        Returns
        ───────
        dict with two keys::

            {
                "assembled_context": str,
                    # Parent paragraph texts joined by '---' separators.
                    # Ready to be injected verbatim into the Sarvam LLM prompt.

                "citations": [
                    {
                        "parent_id":  str,   # UUID-v5 of the parent chunk
                        "article_id": int,   # PG primary key of the source article
                        "source_url": str,   # original URL of the article
                    },
                    ...
                ],
                    # One entry per unique parent paragraph, in RRF rank order.
                    # Display these as footnote / source links in the UI.
            }

        An empty-context result (assembled_context="", citations=[]) is
        returned — never an exception — when Qdrant finds nothing or none of
        the returned parent_ids exist in PostgreSQL.
        """
        log.info("retrieve() — query: '%s'", query[:120])

        # ── Step 1 · Embed the query with BGE-M3 ─────────────────────────────
        dense_vec, sparse_weights = self._embed_query(query)

        # ── Step 2 · Hybrid RRF search in Qdrant ─────────────────────────────
        child_points = self._hybrid_search(dense_vec, sparse_weights)

        if not child_points:
            log.warning("Qdrant returned 0 results — returning empty context.")
            return _empty_result()

        log.debug("Qdrant: %d fused child chunk(s) returned.", len(child_points))

        # ── Step 3 · Extract & deduplicate parent_ids (preserve rank order) ───
        #
        # Multiple top child chunks may share the same parent paragraph.
        # We keep only the first occurrence so the parent is not repeated in
        # the assembled context while still honouring RRF ordering.
        parent_ids_ordered:   list[str]       = []
        article_id_by_parent: dict[str, int]  = {}
        seen_parents:         set[str]        = set()

        for point in child_points:
            payload    = point.payload or {}
            parent_id  = payload.get("parent_id")
            article_id = payload.get("article_id")

            if not parent_id:
                log.warning(
                    "Qdrant point %s has no 'parent_id' in payload — skipping.",
                    point.id,
                )
                continue

            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_ids_ordered.append(parent_id)
                # article_id may legitimately be None for malformed payloads;
                # handled gracefully in Step 5.
                article_id_by_parent[parent_id] = article_id  # type: ignore[assignment]

        if not parent_ids_ordered:
            log.warning("No valid parent_ids found after payload extraction.")
            return _empty_result()

        log.debug(
            "Unique parent_ids to fetch: %d — %s",
            len(parent_ids_ordered),
            parent_ids_ordered,
        )

        # ── Step 4 · Fetch full parent texts from PostgreSQL ──────────────────
        parent_texts: dict[str, str] = self._chunk_repo.get_by_parent_ids(
            parent_ids_ordered
        )

        if not parent_texts:
            log.warning(
                "ParentChunkRepository returned no rows for parent_ids=%s",
                parent_ids_ordered,
            )
            return _empty_result()

        # ── Step 5 · Fetch source URLs from the articles table ────────────────
        unique_article_ids: list[int] = list(
            {aid for aid in article_id_by_parent.values() if aid is not None}
        )
        url_by_article_id: dict[int, str] = self._article_repo.get_urls_by_ids(
            unique_article_ids
        )

        # ── Step 6 · Assemble context string + citations list ─────────────────
        context_parts: list[str]  = []
        citations:     list[dict] = []

        for pid in parent_ids_ordered:
            text = parent_texts.get(pid)

            if not text:
                # Parent_id came from Qdrant but is absent in PG.
                # This can happen transiently during a re-chunking run.
                log.warning(
                    "parent_id '%s' missing from parent_chunks table — skipping.",
                    pid,
                )
                continue

            aid: int | None = article_id_by_parent.get(pid)
            url: str        = url_by_article_id.get(aid, "") if aid is not None else ""

            context_parts.append(text)
            citations.append(
                {
                    "parent_id":  pid,
                    "article_id": aid,
                    "source_url": url,
                }
            )

        assembled_context: str = _CONTEXT_SEPARATOR.join(context_parts)

        log.info(
            "Context assembled — %d paragraph(s) from %d unique article(s).  "
            "Total context length: %d chars.",
            len(context_parts),
            len({c["article_id"] for c in citations if c["article_id"] is not None}),
            len(assembled_context),
        )

        return {
            "assembled_context": assembled_context,
            "citations":         citations,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_query(
        self,
        query: str,
    ) -> tuple[list[float], dict[int, float]]:
        """
        Encode *query* with BGE-M3 in hybrid mode (dense + sparse).

        BGE-M3 is multi-lingual and multi-granularity; passing a single
        sentence (the query) is correct — the model does not require batching.

        Returns
        ───────
        dense_vec      : list[float]
            1024-dimensional cosine-space representation of the query.
            Used for semantic / conceptual search.

        sparse_weights : dict[int, float]
            Token vocabulary ID → IDF-weighted lexical score.
            Used for syntactic / keyword search (mirrors BM25-style retrieval).
        """
        log.debug("BGE-M3: encoding query…")

        output = self._model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vec: list[float]       = output["dense_vecs"][0].tolist()
        sparse_weights: dict[int, float] = output["lexical_weights"][0]

        log.debug(
            "BGE-M3 encoded — dense dim: %d, sparse tokens: %d.",
            len(dense_vec),
            len(sparse_weights),
        )
        return dense_vec, sparse_weights

    def _hybrid_search(
        self,
        dense_vec:      list[float],
        dense_weights:  dict[int, float],   # named for clarity in call-site
    ) -> list[qmodels.ScoredPoint]:
        """
        Issue a two-branch prefetch query to Qdrant with RRF fusion.

        Branch 1 — Semantic (dense):
            Queries the default ``""`` vector (1024-dim cosine BGE-M3).
            Captures conceptual / paraphrase similarity.

        Branch 2 — Syntactic (sparse):
            Queries the ``"hybrid-sparse"`` sparse vector (BGE-M3 lexical
            weights).  Captures exact-term / keyword overlap — critical for
            proper nouns, legislation names, and numeric claims that dense
            embeddings can mishandle.

        Fusion — Reciprocal Rank Fusion (RRF):
            Both ranked lists are merged via Qdrant's native RRF
            implementation.  RRF is rank-position based, so it is robust to
            the different score scales of cosine similarity vs. sparse dot
            product — no normalisation needed.

        Returns
        ───────
        list[ScoredPoint] — up to FINAL_LIMIT points, with full payload.
        """
        # Build the sparse vector from the lexical_weights dict.
        # lex_weights key   = token vocabulary index (int)
        # lex_weights value = IDF-weighted term score (float)
        sparse_vector = qmodels.SparseVector(
            indices=list(dense_weights.keys()),
            values=[float(v) for v in dense_weights.values()],
        )

        log.debug(
            "Qdrant query_points — prefetch_limit=%d, final_limit=%d.",
            PREFETCH_LIMIT,
            FINAL_LIMIT,
        )

        response = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                # ── Branch 1: Dense semantic search ─────────────────────────
                qmodels.Prefetch(
                    query=dense_vec,
                    using="",               # empty-string key = the named dense
                                            # vector in vectors_config
                    limit=PREFETCH_LIMIT,
                ),
                # ── Branch 2: Sparse syntactic/lexical search ────────────────
                qmodels.Prefetch(
                    query=sparse_vector,
                    using="hybrid-sparse",  # sparse vector key in collection
                    limit=PREFETCH_LIMIT,
                ),
            ],
            # RRF merges the two ranked prefetch lists into one final ranking.
            # No score thresholding — RRF handles the fusion purely by rank.
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=FINAL_LIMIT,
            with_payload=True,
        )

        return response.points


# ═══════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════

def _empty_result() -> dict[str, Any]:
    """Canonical empty-context return value — used in all early-exit paths."""
    return {"assembled_context": "", "citations": []}


# ═══════════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════════

def init_retrieval_service(conn: Connection) -> RetrievalService:
    """
    Build a fully initialised RetrievalService from environment variables.

    Reads QDRANT_HOST, QDRANT_PORT, and BGE_MODEL from the environment
    (with the same defaults as factlens_chunker.py).

    Call once at application startup; pass the returned instance everywhere.

    Parameters
    ──────────
    conn : Connection
        An open db.connection.Connection.  Injected so the caller controls
        connection lifecycle (close on shutdown, reuse across services, etc.).

    Returns
    ───────
    RetrievalService
    """
    log.info("Initialising RetrievalService…")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    log.info("Connecting to Qdrant at %s:%d…", QDRANT_HOST, QDRANT_PORT)
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    qdrant_client.get_collections()   # lightweight liveness check
    log.info("Qdrant: connection established.")

    # ── BGE-M3 ────────────────────────────────────────────────────────────────
    log.info(
        "Loading BGE-M3 model '%s' (use_fp16=True)…  "
        "First run may download ~2 GB of weights.",
        BGE_MODEL_NAME,
    )
    embed_model = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=True)
    log.info("BGE-M3: model loaded.")

    return RetrievalService(
        embed_model=embed_model,
        qdrant_client=qdrant_client,
        conn=conn,
    )
