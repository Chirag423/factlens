"""
FactLens — factlens_chunker.py

Parent-Child Chunking & Hybrid Embedding Pipeline
==================================================
Reads articles in state 'NER Done' from PostgreSQL, produces:

  • Parent chunks  (paragraph-level, ~300–500 words)
        → stored in PostgreSQL via ParentChunkRepository

  • Child chunks   (sentence-level)
        → encoded with BGE-M3 (dense 1024-dim + sparse lexical weights)
        → upserted to Qdrant as hybrid vectors

Idempotency guarantees
──────────────────────
  • parent_id  : uuid5(article_id || chunk_index)
        → ON CONFLICT (parent_id) DO NOTHING in parent_chunks
  • child_id   : uuid5(parent_id  || sentence_index)
        → Qdrant upsert overwrites any existing point with the same ID
  • State machine: articles advance
        'NER Done' → 'Chunking Done'   (success)
        'NER Done' → 'Chunking Error'  (persistent per-article failure)
    Re-running the script never reprocesses already-committed articles.

Integration
───────────
  Uses the project's shared modules:
    db.connection              → Connection (psycopg2 wrapper)
    db.repositories.articles   → ArticleRepository
    db.repositories.chunks     → ParentChunkRepository
    db.models                  → ParentChunkRow

─────────────────────────────────────────────────────────────────────────────
requirements.txt additions
─────────────────────────────────────────────────────────────────────────────
  qdrant-client>=1.9.0
  FlagEmbedding>=1.2.9
  spacy>=3.7.4
  torch>=2.2.0

  # After pip install, download the spaCy model:
  #   python -m spacy download en_core_web_sm
─────────────────────────────────────────────────────────────────────────────

Environment variables (all have sensible defaults)
───────────────────────────────────────────────────
  DATABASE_URL    PostgreSQL DSN  (required — read by db.connection.Connection)
  QDRANT_HOST     Qdrant host     (default: localhost)
  QDRANT_PORT     Qdrant port     (default: 6333)
  SPACY_MODEL     spaCy model     (default: en_core_web_sm)
  BGE_MODEL       BGE-M3 path    (default: BAAI/bge-m3)
"""

from __future__ import annotations

# ── Standard library ────────────────────────────────────────────────────────
import logging
import os
import re
import sys
import uuid
from typing import NamedTuple

# ── Third-party ─────────────────────────────────────────────────────────────
import spacy
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models as qmodels

# ── Project modules ──────────────────────────────────────────────────────────
from db.connection import Connection
from db.models import ParentChunkRow
from db.repositories.articles import ArticleRepository
from db.repositories.chunks import ParentChunkRepository


# ═══════════════════════════════════════════════════════════════════════════
# Logging  (uses project-level logging; this module just gets its own logger)
# ═══════════════════════════════════════════════════════════════════════════

log = logging.getLogger("FactLens.Chunker")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration  (all tuneable via environment variables)
# ═══════════════════════════════════════════════════════════════════════════

# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME: str = "factlens_child_chunks"
VECTOR_SIZE: int     = 1024          # BGE-M3 dense output dimension

# ── Model names ─────────────────────────────────────────────────────────────
SPACY_MODEL: str    = os.getenv("SPACY_MODEL", "en_core_web_trf")
BGE_MODEL_NAME: str = os.getenv("BGE_MODEL", "BAAI/bge-m3")

# ── Pipeline knobs ──────────────────────────────────────────────────────────
BATCH_SIZE: int       = 100   # articles fetched per DB round-trip
PARENT_MIN_WORDS: int = 300   # soft lower bound for a parent chunk (informational)
PARENT_MAX_WORDS: int = 500   # hard upper bound (enforced by chunker)
EMBED_BATCH_SIZE: int = 32    # sentences per BGE-M3 forward pass

# ── Processing states ────────────────────────────────────────────────────────
SOURCE_STATE:  str = "ner_done"
SUCCESS_STATE: str = "Chunking Done"
FAILURE_STATE: str = "Chunking Error"

# ── Safety valve ─────────────────────────────────────────────────────────────
# Halt after this many consecutive batch-level failures (e.g. Qdrant down).
# Per-article failures do NOT count toward this limit.
MAX_CONSECUTIVE_BATCH_FAILURES: int = 3

# ── Deterministic UUID namespaces (must never change between runs) ───────────
_PARENT_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
_CHILD_NS  = uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")

# ── Entity columns to flatten into the Qdrant payload ───────────────────────
_ENTITY_COLUMNS: tuple[str, ...] = (
    "entity_person", "entity_norp",   "entity_fac",      "entity_org",
    "entity_gpe",    "entity_loc",    "entity_product",  "entity_event",
    "entity_work_of_art", "entity_law", "entity_language", "entity_date",
    "entity_time",   "entity_percent", "entity_money",   "entity_quantity",
    "entity_ordinal", "entity_cardinal",
)


# ═══════════════════════════════════════════════════════════════════════════
# Internal data container
# ═══════════════════════════════════════════════════════════════════════════

class _ArticleData(NamedTuple):
    """Lightweight view of an article row as needed by the chunker."""
    article_id:         int
    content:            str
    source_url:         str
    canonical_entities: list[str]   # flattened from all 18 entity_* arrays


# ═══════════════════════════════════════════════════════════════════════════
# ID helpers — deterministic UUIDs guarantee idempotency on re-runs
# ═══════════════════════════════════════════════════════════════════════════

def make_parent_id(article_id: int, chunk_index: int) -> str:
    """
    Stable UUID-v5 for a parent chunk.
    Same article_id + chunk_index always produces the same UUID.
    """
    return str(uuid.uuid5(_PARENT_NS, f"parent::{article_id}::{chunk_index}"))


def make_child_id(parent_id: str, sentence_index: int) -> str:
    """
    Stable UUID-v5 for a child (sentence) chunk.
    Used as the Qdrant point ID — upserts overwrite if it already exists.
    """
    return str(uuid.uuid5(_CHILD_NS, f"child::{parent_id}::{sentence_index}"))


# ═══════════════════════════════════════════════════════════════════════════
# Entity helper
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_entities(row: dict) -> list[str]:
    """
    Merge all 18 entity_* TEXT[] columns from an article row into one flat list.

    None values and empty arrays are silently skipped.  The result is stored
    in every child chunk's Qdrant payload under ``canonical_entities`` so that
    entity-based filtering is possible at query time.
    """
    entities: list[str] = []
    for col in _ENTITY_COLUMNS:
        val = row.get(col)
        if val:
            entities.extend(val)
    return entities


# ═══════════════════════════════════════════════════════════════════════════
# Parent chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_into_parents(content: str) -> list[str]:
    """
    Split article content into paragraph-level parent chunks of approximately
    PARENT_MIN_WORDS – PARENT_MAX_WORDS words each.

    Algorithm
    ─────────
    1. Split on blank lines (natural paragraph boundaries).
    2. Accumulate paragraphs into a buffer until adding the next would
       exceed PARENT_MAX_WORDS → flush buffer as one chunk.
    3. Paragraphs longer than PARENT_MAX_WORDS are hard-split by word count
       (word boundaries are preserved).

    Edge cases
    ──────────
    • Articles with no blank lines → treated as a single paragraph and
      hard-split if necessary.
    • Trailing whitespace and empty segments are stripped before processing.
    """
    raw_paras: list[str] = [
        p.strip()
        for p in re.split(r"\n\s*\n", content.strip())
        if p.strip()
    ]
    if not raw_paras:
        raw_paras = [content.strip()]

    chunks: list[str] = []
    buf:    list[str] = []
    buf_wc: int       = 0

    for para in raw_paras:
        words   = para.split()
        para_wc = len(words)

        if para_wc > PARENT_MAX_WORDS:
            # ── Oversized paragraph — flush pending buffer, then hard-split ─
            if buf:
                chunks.append("\n\n".join(buf))
                buf.clear()
                buf_wc = 0
            for start in range(0, para_wc, PARENT_MAX_WORDS):
                chunks.append(" ".join(words[start : start + PARENT_MAX_WORDS]))

        elif buf_wc + para_wc > PARENT_MAX_WORDS:
            # ── Adding this paragraph would overflow the buffer ────────────
            if buf:
                chunks.append("\n\n".join(buf))
            buf    = [para]
            buf_wc = para_wc

        else:
            # ── Fits in current buffer ─────────────────────────────────────
            buf.append(para)
            buf_wc += para_wc

    if buf:
        chunks.append("\n\n".join(buf))

    return [c for c in chunks if c.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# Child chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_into_children(parent_id: str, parent_text: str, nlp) -> list[dict]:
    """
    Split a parent chunk into sentence-level child chunks using spaCy.

    Returns
    ───────
    list of dicts, each containing:
      child_id        : deterministic UUID string (used as the Qdrant point ID)
      sentence_index  : 0-based position within the parent
      text            : cleaned sentence string
    """
    doc      = nlp(parent_text)
    children: list[dict] = []

    for idx, sent in enumerate(doc.sents):
        text = sent.text.strip()
        if not text:
            continue
        children.append(
            {
                "child_id":       make_child_id(parent_id, idx),
                "sentence_index": idx,
                "text":           text,
            }
        )

    return children


# ═══════════════════════════════════════════════════════════════════════════
# Embedding — BGE-M3 hybrid (dense + sparse)
# ═══════════════════════════════════════════════════════════════════════════

def generate_embeddings(
    texts: list[str],
    model: BGEM3FlagModel,
) -> tuple[list, list[dict]]:
    """
    Encode a batch of texts with BGE-M3 in hybrid mode, returning both the
    1024-dim dense vectors and the sparse lexical weight dicts.

    Parameters
    ──────────
    texts  : list of sentence strings
    model  : initialised BGEM3FlagModel instance

    Returns
    ───────
    dense_vecs     : numpy array of shape (N, 1024)
    lexical_weights: list of dicts mapping token_id (int) → weight (float),
                     one dict per input text.
    """
    result = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        max_length=512,
        return_dense=True,
        return_sparse=True,        # ← enables lexical_weights output
        return_colbert_vecs=False,
    )
    # result["dense_vecs"]      : numpy ndarray (N, 1024)
    # result["lexical_weights"] : list[dict[int, float]], length N
    return result["dense_vecs"], result["lexical_weights"]


# ═══════════════════════════════════════════════════════════════════════════
# Qdrant operations
# ═══════════════════════════════════════════════════════════════════════════

def ensure_qdrant_collection(client: QdrantClient) -> None:
    """
    Create the Qdrant collection with hybrid vector support if it does not
    already exist.  Safe to call on every startup (idempotent).

    Collection layout
    ─────────────────
      ""              → dense cosine vectors  (1024-dim, BGE-M3)
      "hybrid-sparse" → sparse lexical vectors (BGE-M3 lexical_weights)
    """
    if client.collection_exists(COLLECTION_NAME):
        log.info(
            "Qdrant: collection '%s' already exists — skipping creation.",
            COLLECTION_NAME,
        )
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
        sparse_vectors_config={
            "hybrid-sparse": qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=True)
            )
        },
    )
    log.info(
        "Qdrant: created collection '%s' (dense=%d-dim Cosine + sparse hybrid-sparse).",
        COLLECTION_NAME,
        VECTOR_SIZE,
    )


def qdrant_upsert(client: QdrantClient, points: list[qmodels.PointStruct]) -> None:
    """
    Upsert a list of PointStructs into the collection.

    ``wait=True`` ensures the operation is fully persisted before returning,
    so callers can safely update PostgreSQL state afterwards without risking
    a Qdrant-side gap.

    Existing points with the same ID are silently overwritten — this is the
    Qdrant-side idempotency guarantee that makes re-runs safe.
    """
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    log.info("Qdrant: upserted %d point(s).", len(points))


# ═══════════════════════════════════════════════════════════════════════════
# Batch fetch
# ═══════════════════════════════════════════════════════════════════════════

def fetch_batch(article_repo: ArticleRepository) -> list[_ArticleData]:
    """
    Fetch up to BATCH_SIZE articles in state SOURCE_STATE via ArticleRepository.

    Column mapping (DB → _ArticleData)
    ───────────────────────────────────
      id          → article_id
      raw_content → content
      url         → source_url
      entity_*[]  → canonical_entities  (all 18 arrays flattened into one list)
    """
    rows = article_repo.get_by_state(SOURCE_STATE, limit=BATCH_SIZE)
    if not rows:
        return []

    result: list[_ArticleData] = []
    for r in rows:
        result.append(
            _ArticleData(
                article_id=r["id"],
                content=r.get("raw_content") or "",
                source_url=r.get("url") or "",
                canonical_entities=_flatten_entities(r),
            )
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Core batch processor
# ═══════════════════════════════════════════════════════════════════════════

def process_batch(
    batch:        list[_ArticleData],
    parent_repo:  ParentChunkRepository,
    qdrant:       QdrantClient,
    nlp,
    embed_model:  BGEM3FlagModel,
) -> tuple[list[int], list[int]]:
    """
    Process one batch of articles through the full pipeline:
      article → parent chunks (PG) → child chunks → hybrid embeddings → Qdrant

    Per-article failures are caught, logged, and added to failed_ids so the
    rest of the batch continues uninterrupted.

    Batch-level write failures (ParentChunkRepository.bulk_insert or
    qdrant_upsert) are intentionally NOT caught here — they propagate to
    main() which decides whether to retry or halt.

    Transaction model
    ─────────────────
    Because db.connection.Connection._tx() auto-commits on context exit,
    parent chunk inserts and state updates are in separate PG transactions.
    If Qdrant upsert fails after parents are inserted:
      • Parent rows persist (idempotent: DO NOTHING on re-run).
      • Article state stays SOURCE_STATE → reprocessed next run.
      • Qdrant points are re-upserted (same deterministic IDs, safe overwrite).
    This guarantees eventual consistency with no manual rollback needed.

    Returns
    ───────
    (successful_ids, failed_ids) — both are lists of article_id ints.
    """
    all_parent_rows:   list[ParentChunkRow]     = []
    all_qdrant_points: list[qmodels.PointStruct] = []
    successful_ids:    list[int]                 = []
    failed_ids:        list[int]                 = []

    for article in batch:
        if not article.content.strip():
            log.warning(
                "Article %d: empty raw_content — marking as error.",
                article.article_id,
            )
            failed_ids.append(article.article_id)
            continue

        try:
            # ── 1. Parent chunking ──────────────────────────────────────────
            parent_texts = chunk_into_parents(article.content)
            if not parent_texts:
                raise ValueError("chunk_into_parents() returned no chunks.")

            article_parent_rows:  list[ParentChunkRow]     = []
            article_qdrant_points: list[qmodels.PointStruct] = []

            for p_idx, p_text in enumerate(parent_texts):
                parent_id = make_parent_id(article.article_id, p_idx)

                # ── 2. Collect parent row (bulk-inserted after all articles) ─
                article_parent_rows.append(
                    ParentChunkRow(
                        parent_id=parent_id,
                        article_id=article.article_id,
                        chunk_index=p_idx,
                        content=p_text,
                    )
                )

                # ── 3. Child chunking (sentence-level via spaCy) ────────────
                children = chunk_into_children(parent_id, p_text, nlp)
                if not children:
                    log.warning(
                        "Article %d / parent %d: no sentences extracted — skipping.",
                        article.article_id,
                        p_idx,
                    )
                    continue

                child_texts = [c["text"] for c in children]

                # ── 4. Hybrid embedding with BGE-M3 ────────────────────────
                dense_vecs, lexical_weights = generate_embeddings(child_texts, embed_model)

                # ── 5. Build Qdrant PointStructs (hybrid: dense + sparse) ───
                for child, dense_vec, lex_weights in zip(children, dense_vecs, lexical_weights):
                    article_qdrant_points.append(
                        qmodels.PointStruct(
                            id=child["child_id"],
                            vector={
                                # Empty-string key = the named dense vector
                                # configured in vectors_config for the collection.
                                "": dense_vec.tolist(),
                                # Sparse lexical weights from BGE-M3.
                                # lex_weights is dict[int, float]:
                                #   key   = token vocabulary index
                                #   value = IDF-weighted term score
                                "hybrid-sparse": qmodels.SparseVector(
                                    indices=list(lex_weights.keys()),
                                    values=[float(v) for v in lex_weights.values()],
                                ),
                            },
                            payload={
                                # Primary retrieval fields
                                "text":               child["text"],
                                "parent_id":          parent_id,
                                "article_id":         article.article_id,
                                "source_url":         article.source_url,
                                # Entity filter support
                                "canonical_entities": article.canonical_entities,
                                # Debugging / re-ranking metadata
                                "sentence_index":     child["sentence_index"],
                                "parent_chunk_index": p_idx,
                            },
                        )
                    )

            all_parent_rows.extend(article_parent_rows)
            all_qdrant_points.extend(article_qdrant_points)
            successful_ids.append(article.article_id)

            log.debug(
                "  ✓ Article %d → %d parent(s), %d child chunk(s).",
                article.article_id,
                len(article_parent_rows),
                len(article_qdrant_points),
            )

        except Exception as exc:
            log.error(
                "  ✗ Article %d failed during chunking/embedding: %s",
                article.article_id,
                exc,
                exc_info=True,
            )
            failed_ids.append(article.article_id)

    # ── Batch writes (exceptions propagate intentionally to main()) ──────────
    #
    # Order matters:
    #   1. Insert parents into PG first (idempotent via DO NOTHING on re-runs).
    #   2. Upsert child embeddings to Qdrant (idempotent via deterministic IDs).
    #   If step 2 fails, step 1 is already committed but state stays SOURCE_STATE,
    #   so the article will be fully reprocessed on the next run — safe.

    if all_parent_rows:
        parent_repo.bulk_insert(all_parent_rows)

    if all_qdrant_points:
        qdrant_upsert(qdrant, all_qdrant_points)

    return successful_ids, failed_ids


# ═══════════════════════════════════════════════════════════════════════════
# Initialisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def init_qdrant() -> QdrantClient:
    """Connect to the Qdrant instance and verify the connection is alive."""
    log.info("Connecting to Qdrant at %s:%d…", QDRANT_HOST, QDRANT_PORT)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.get_collections()   # lightweight liveness check
    log.info("Qdrant: connection established.")
    return client


def init_spacy(model_name: str = SPACY_MODEL):
    """
    Load the spaCy model and disable all pipeline components not needed for
    sentence segmentation, improving throughput.

    Kept components: tok2vec (shared encoder), parser (provides doc.sents),
    senter (lightweight alternative segmenter).
    """
    log.info("Loading spaCy model '%s'…", model_name)
    try:
        nlp = spacy.load(model_name)
    except OSError:
        log.critical(
            "spaCy model '%s' not found. Run:  python -m spacy download %s",
            model_name,
            model_name,
        )
        sys.exit(1)

    keep_pipes    = {"tok2vec", "parser", "senter"}
    pipes_to_drop = [p for p in nlp.pipe_names if p not in keep_pipes]
    for pipe_name in pipes_to_drop:
        try:
            nlp.disable_pipe(pipe_name)
        except Exception:
            pass

    nlp.max_length = 2_000_000   # guard against very long articles
    log.info("spaCy: loaded (active pipes: %s).", list(nlp.pipe_names))
    return nlp


def init_embed_model(model_name: str = BGE_MODEL_NAME) -> BGEM3FlagModel:
    """
    Load the BGE-M3 model.

    First run downloads ~2 GB of weights to the HuggingFace cache.
    use_fp16=True halves VRAM usage and speeds up GPU inference; on CPU-only
    machines the flag is silently ignored by the library.
    """
    log.info("Loading BGE-M3 model '%s' (first run downloads ~2 GB)…", model_name)
    model = BGEM3FlagModel(model_name, use_fp16=True)
    log.info("BGE-M3: model loaded.")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log.info("=" * 65)
    log.info("FactLens Chunking Pipeline starting.")
    log.info("=" * 65)

    # ── 1. Initialise DB (uses DATABASE_URL from environment) ────────────────
    try:
        conn         = Connection()
        article_repo = ArticleRepository(conn)
        parent_repo  = ParentChunkRepository(conn)
    except RuntimeError as exc:
        log.critical("Cannot initialise database connection: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.critical("Unexpected error initialising database: %s", exc, exc_info=True)
        sys.exit(1)

    # ── 2. Initialise Qdrant ─────────────────────────────────────────────────
    try:
        qdrant = init_qdrant()
        ensure_qdrant_collection(qdrant)
    except Exception as exc:
        log.critical("Cannot connect to / initialise Qdrant: %s", exc)
        conn.close()
        sys.exit(1)

    # ── 3. Initialise ML models ──────────────────────────────────────────────
    try:
        nlp         = init_spacy()
        embed_model = init_embed_model()
    except SystemExit:
        conn.close()
        raise   # propagate sys.exit() cleanly

    # ── 4. Main processing loop ──────────────────────────────────────────────
    total_ok:   int = 0
    total_fail: int = 0
    batch_num:  int = 0
    consecutive_batch_failures: int = 0

    try:
        while True:
            # Fetch from the logical top — processed rows have transitioned out
            # of SOURCE_STATE and will not reappear.
            batch = fetch_batch(article_repo)

            if not batch:
                log.info(
                    "No more articles in state '%s'. Pipeline complete.",
                    SOURCE_STATE,
                )
                break

            batch_num += 1
            log.info(
                "══ Batch %d: %d article(s) fetched (state='%s') ══",
                batch_num,
                len(batch),
                SOURCE_STATE,
            )

            try:
                # Core pipeline — per-article errors are swallowed internally;
                # batch-level write errors propagate here.
                ok_ids, fail_ids = process_batch(
                    batch, parent_repo, qdrant, nlp, embed_model
                )

                # Advance article states — each call is its own _tx() commit.
                article_repo.bulk_update_state(ok_ids,   SUCCESS_STATE)
                article_repo.bulk_update_state(fail_ids, FAILURE_STATE)

                total_ok   += len(ok_ids)
                total_fail += len(fail_ids)
                consecutive_batch_failures = 0

                log.info(
                    "Batch %d done — ✓ %d succeeded  ✗ %d failed  "
                    "(running totals: %d ok / %d fail).",
                    batch_num,
                    len(ok_ids),
                    len(fail_ids),
                    total_ok,
                    total_fail,
                )

            except Exception as exc:
                # A batch-level failure (e.g. Qdrant unreachable, PG write error).
                # Articles retain SOURCE_STATE and will be retried on the next run.
                # Any parent_chunks already inserted are idempotent on re-run.
                log.error(
                    "Batch %d: batch-level failure — articles will be retried. Error: %s",
                    batch_num,
                    exc,
                    exc_info=True,
                )

                consecutive_batch_failures += 1
                if consecutive_batch_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                    log.critical(
                        "%d consecutive batch failures. Halting to avoid an "
                        "infinite retry loop. Fix the underlying issue and rerun.",
                        consecutive_batch_failures,
                    )
                    break

                log.info(
                    "Will retry next batch (%d/%d consecutive failures so far).",
                    consecutive_batch_failures,
                    MAX_CONSECUTIVE_BATCH_FAILURES,
                )
                continue

    except KeyboardInterrupt:
        log.warning("Interrupted by user.")

    except Exception as exc:
        log.critical("Unhandled exception in main loop: %s", exc, exc_info=True)

    finally:
        conn.close()
        log.info(
            "Pipeline finished. Total: %d article(s) processed successfully, "
            "%d failed.",
            total_ok,
            total_fail,
        )


if __name__ == "__main__":
    main()
