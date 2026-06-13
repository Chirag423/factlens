"""
FactLens — Parent-Child Chunking & Embedding Pipeline
======================================================
Reads articles in state 'NER Done' from PostgreSQL, produces:
  • Parent chunks  (paragraph-level, 300-500 words) → stored in PostgreSQL
  • Child chunks   (sentence-level)                 → embedded with BGE-M3
                                                       and upserted to Qdrant

Idempotency guarantees
──────────────────────
  • parent_id  : uuid5(article_id + chunk_index)  → PG uses ON CONFLICT DO NOTHING
  • child_id   : uuid5(parent_id  + sent_index)   → Qdrant uses upsert (overwrite)
  • State machine: articles advance NER Done → Chunking Done (success)
                                            → Chunking Error (persistent failure)
    so re-running never reprocesses already-committed articles.

─────────────────────────────────────────────────────────────────────────────
requirements.txt
─────────────────────────────────────────────────────────────────────────────
  psycopg2-binary>=2.9.9
  qdrant-client>=1.9.0
  FlagEmbedding>=1.2.9
  spacy>=3.7.4
  torch>=2.2.0
  transformers>=4.40.0
  numpy>=1.26.0

  # After pip install, download the spaCy model:
  #   python -m spacy download en_core_web_sm
─────────────────────────────────────────────────────────────────────────────

Environment variables
─────────────────────
  PG_DSN          PostgreSQL DSN (default: postgresql://user:password@localhost:5432/factlens)
  QDRANT_HOST     Qdrant host    (default: localhost)
  QDRANT_PORT     Qdrant port    (default: 6333)
  SPACY_MODEL     spaCy model    (default: en_core_web_sm)
  BGE_MODEL       BGE-M3 model   (default: BAAI/bge-m3)
"""

# ── Standard library ────────────────────────────────────────────────────────
import json
import logging
import os
import re
import sys
import uuid
from typing import NamedTuple

# ── Third-party ─────────────────────────────────────────────────────────────
import psycopg2
import psycopg2.extras
import spacy
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("FactLens.Chunker")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration  (all tuneable via environment variables)
# ═══════════════════════════════════════════════════════════════════════════

# ── Connections ─────────────────────────────────────────────────────────────
PG_DSN: str       = os.getenv("DATABASE_URL")
QDRANT_HOST: str  = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int  = int(os.getenv("QDRANT_PORT", "6333"))

# ── Qdrant collection ───────────────────────────────────────────────────────
COLLECTION_NAME: str = "factlens_child_chunks"
VECTOR_SIZE: int     = 1024          # BGE-M3 dense output dimension

# ── Model names ─────────────────────────────────────────────────────────────
SPACY_MODEL: str    = os.getenv("SPACY_MODEL", "en_core_web_sm")
BGE_MODEL_NAME: str = os.getenv("BGE_MODEL", "BAAI/bge-m3")

# ── Pipeline knobs ──────────────────────────────────────────────────────────
BATCH_SIZE: int        = 100    # articles fetched per DB round-trip
PARENT_MIN_WORDS: int  = 300    # soft lower bound for a parent chunk
PARENT_MAX_WORDS: int  = 500    # hard upper bound  (enforced)
EMBED_BATCH_SIZE: int  = 32     # sentences per BGE-M3 forward pass

# ── Processing states ────────────────────────────────────────────────────────
SOURCE_STATE:  str = "NER Done"
SUCCESS_STATE: str = "Chunking Done"
FAILURE_STATE: str = "Chunking Error"

# ── Safety valve ─────────────────────────────────────────────────────────────
# Stop after this many consecutive batch-level failures (e.g. Qdrant is down)
# to avoid an infinite retry loop. Per-article failures do NOT count here.
MAX_CONSECUTIVE_BATCH_FAILURES: int = 3

# ── Deterministic UUID namespaces (must never change between runs) ───────────
_PARENT_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
_CHILD_NS  = uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")


# ═══════════════════════════════════════════════════════════════════════════
# Typed data container
# ═══════════════════════════════════════════════════════════════════════════

class ArticleRow(NamedTuple):
    article_id:           str
    content:              str
    source_url:           str
    canonical_entity_ids: list   # list[str] from entities_json


# ═══════════════════════════════════════════════════════════════════════════
# ID helpers — deterministic UUIDs guarantee idempotency on re-runs
# ═══════════════════════════════════════════════════════════════════════════

def make_parent_id(article_id: str, chunk_index: int) -> str:
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
# Parent chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_into_parents(content: str) -> list[str]:
    """
    Split article content into logical parent chunks of approximately
    PARENT_MIN_WORDS – PARENT_MAX_WORDS words each.

    Algorithm
    ─────────
    1. Split on blank lines (natural paragraph boundaries).
    2. Accumulate paragraphs into a buffer until adding the next would
       exceed PARENT_MAX_WORDS → flush buffer as one chunk.
    3. Paragraphs that are themselves longer than PARENT_MAX_WORDS are
       hard-split by word count (preserving word boundaries).

    Edge cases
    ──────────
    • Articles with no blank lines are treated as a single long paragraph
      and hard-split if needed.
    • Trailing whitespace and empty segments are stripped.
    """
    # Step 1 — split on blank lines
    raw_paras: list[str] = [
        p.strip()
        for p in re.split(r"\n\s*\n", content.strip())
        if p.strip()
    ]
    if not raw_paras:
        # Whole article is one block; treat it as a single paragraph
        raw_paras = [content.strip()]

    chunks: list[str] = []
    buf: list[str]    = []   # paragraphs accumulated in the current chunk
    buf_wc: int       = 0    # current buffer word count

    for para in raw_paras:
        words  = para.split()
        para_wc = len(words)

        if para_wc > PARENT_MAX_WORDS:
            # ── Oversized paragraph ───────────────────────────────────────
            # Flush any pending buffer first
            if buf:
                chunks.append("\n\n".join(buf))
                buf.clear()
                buf_wc = 0
            # Hard-split by word count
            for start in range(0, para_wc, PARENT_MAX_WORDS):
                sub_chunk = " ".join(words[start : start + PARENT_MAX_WORDS])
                chunks.append(sub_chunk)

        elif buf_wc + para_wc > PARENT_MAX_WORDS:
            # ── Adding this paragraph would overflow the buffer ───────────
            if buf:
                chunks.append("\n\n".join(buf))
            buf    = [para]
            buf_wc = para_wc

        else:
            # ── Fits in current buffer ────────────────────────────────────
            buf.append(para)
            buf_wc += para_wc

    # Final flush of any remaining buffer content
    if buf:
        chunks.append("\n\n".join(buf))

    return [c for c in chunks if c.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# Child chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_into_children(parent_id: str, parent_text: str, nlp) -> list[dict]:
    """
    Split a parent chunk into sentence-level child chunks using spaCy's
    dependency parser / senter component.

    Returns
    ───────
    List of dicts, each containing:
      child_id        : deterministic UUID string
      sentence_index  : integer position within the parent (0-based; gaps
                        are possible if spaCy produced empty sentences)
      text            : cleaned sentence string
    """
    doc = nlp(parent_text)
    children: list[dict] = []

    for idx, sent in enumerate(doc.sents):
        text = sent.text.strip()
        if not text:
            # Skip blank sentences (e.g. stray newlines parsed as segments)
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
# Embedding
# ═══════════════════════════════════════════════════════════════════════════

def generate_embeddings(texts: list[str], model: BGEM3FlagModel) -> list[list[float]]:
    """
    Encode a batch of texts with BGE-M3, returning only the 1024-dim
    dense vectors (sparse and ColBERT outputs are disabled for speed).

    Parameters
    ──────────
    texts  : list of sentence strings
    model  : initialised BGEM3FlagModel instance

    Returns
    ───────
    List of float lists, one per input text, each of length VECTOR_SIZE.
    """
    result = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        max_length=512,           # sufficient for sentence-level chunks
        return_dense=True,
        return_sparse=False,      # not needed for this pipeline
        return_colbert_vecs=False,
    )
    # result["dense_vecs"] is a numpy array of shape (len(texts), 1024)
    return result["dense_vecs"].tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Qdrant operations
# ═══════════════════════════════════════════════════════════════════════════

def ensure_qdrant_collection(client: QdrantClient) -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Safe to call on every startup (idempotent).
    """
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        log.info("Qdrant: created collection '%s' (size=%d, Cosine).", COLLECTION_NAME, VECTOR_SIZE)
    else:
        log.info("Qdrant: collection '%s' already exists — skipping creation.", COLLECTION_NAME)


def qdrant_upsert(client: QdrantClient, points: list[PointStruct]) -> None:
    """
    Upsert a list of PointStructs into the collection.
    Existing points with the same ID are silently overwritten — this is
    the Qdrant-side idempotency guarantee.

    `wait=True` ensures the operation is fully persisted before returning.
    """
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    log.info("Qdrant: upserted %d point(s).", len(points))


# ═══════════════════════════════════════════════════════════════════════════
# PostgreSQL helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_entities(raw) -> list:
    """
    Safely convert the entities_json column to a Python list.

    Handles three cases that can occur depending on the psycopg2 version
    and column type (TEXT vs JSONB):
      • Already a list  (psycopg2 auto-decoded JSONB)
      • A JSON string   (TEXT column or auto-decode disabled)
      • None / null     (article has no linked entities)
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            log.warning("entities_json could not be decoded: %.60r", raw)
            return []
    return []


def fetch_batch(conn) -> list[ArticleRow]:
    """
    Fetch up to BATCH_SIZE articles whose processing_state is SOURCE_STATE.
    Results are ordered by article_id for stable pagination.

    Because we update the state of each processed article before the next
    fetch, we always query from the logical top (no OFFSET needed).
    """
    sql = """
        SELECT article_id::text                    AS article_id,
               content,
               COALESCE(source_url, '')            AS source_url,
               entities_json
        FROM   articles
        WHERE  processing_state = %s
        ORDER  BY article_id
        LIMIT  %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (SOURCE_STATE, BATCH_SIZE))
        rows = cur.fetchall()

    return [
        ArticleRow(
            article_id=str(r["article_id"]),
            content=r["content"] or "",
            source_url=r["source_url"] or "",
            canonical_entity_ids=_parse_entities(r["entities_json"]),
        )
        for r in rows
    ]


def pg_insert_parents(conn, records: list[dict]) -> None:
    """
    Bulk-insert parent chunk records into the parent_chunks table.

    ON CONFLICT (parent_id) DO NOTHING ensures that re-runs are safe:
    a deterministic parent_id that already exists is silently skipped.
    Requires a UNIQUE / PRIMARY KEY constraint on parent_chunks.parent_id.
    """
    sql = """
        INSERT INTO parent_chunks (parent_id, article_id, parent_text)
        VALUES (%s, %s, %s)
        ON CONFLICT (parent_id) DO NOTHING
    """
    rows = [(r["parent_id"], r["article_id"], r["parent_text"]) for r in records]
    with conn.cursor() as cur:
        # execute_batch is significantly faster than executemany for bulk inserts
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    log.debug("PG: inserted/skipped %d parent chunk row(s).", len(rows))


def pg_update_states(conn, article_ids: list[str], new_state: str) -> None:
    """
    Bulk-update the processing_state column for a list of article IDs.
    Uses ANY(%s) for an efficient single-query update regardless of list size.
    """
    if not article_ids:
        return
    sql = """
        UPDATE articles
        SET    processing_state = %s
        WHERE  article_id::text = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (new_state, article_ids))
    log.debug("PG: set %d article(s) → '%s'.", len(article_ids), new_state)


# ═══════════════════════════════════════════════════════════════════════════
# Core batch processor
# ═══════════════════════════════════════════════════════════════════════════

def process_batch(
    batch:       list[ArticleRow],
    conn,
    qdrant:      QdrantClient,
    nlp,
    embed_model: BGEM3FlagModel,
) -> tuple[list[str], list[str]]:
    """
    Process one batch of articles through the full pipeline:
      article → parent chunks → child chunks → embeddings → Qdrant points

    Per-article failures are caught and logged; the article is added to
    failed_ids so it can be marked accordingly. The rest of the batch
    continues uninterrupted.

    Batch-level write failures (pg_insert_parents, qdrant_upsert) are
    intentionally NOT caught here — they propagate to main() which rolls
    back the open transaction and decides whether to retry or halt.

    Returns
    ───────
    (successful_ids, failed_ids) — both are lists of article_id strings.
    """
    all_parent_records: list[dict]        = []
    all_qdrant_points:  list[PointStruct] = []
    successful_ids:     list[str]         = []
    failed_ids:         list[str]         = []

    # ── Per-article processing ───────────────────────────────────────────────
    for article in batch:
        if not article.content.strip():
            log.warning("Article %s: empty content — marking as error.", article.article_id)
            failed_ids.append(article.article_id)
            continue

        try:
            # ── 1. Parent chunking ──────────────────────────────────────────
            parent_texts = chunk_into_parents(article.content)
            if not parent_texts:
                raise ValueError("chunk_into_parents() returned no chunks.")

            article_parents: list[dict]        = []
            article_points:  list[PointStruct] = []

            for p_idx, p_text in enumerate(parent_texts):
                parent_id = make_parent_id(article.article_id, p_idx)

                # ── 2. Store parent metadata (committed later as a batch) ───
                article_parents.append(
                    {
                        "parent_id":   parent_id,
                        "article_id":  article.article_id,
                        "parent_text": p_text,
                    }
                )

                # ── 3. Child chunking (sentence-level) ─────────────────────
                children = chunk_into_children(parent_id, p_text, nlp)
                if not children:
                    log.warning(
                        "Article %s / parent %d: no sentences extracted — skipping.",
                        article.article_id, p_idx,
                    )
                    continue

                child_texts = [c["text"] for c in children]

                # ── 4. Embed child chunks with BGE-M3 ───────────────────────
                vectors = generate_embeddings(child_texts, embed_model)

                # ── 5. Build Qdrant PointStructs ────────────────────────────
                for child, vector in zip(children, vectors):
                    article_points.append(
                        PointStruct(
                            id=child["child_id"],
                            vector=vector,
                            payload={
                                # Required fields per spec
                                "text":                 child["text"],
                                "parent_id":            parent_id,
                                "article_id":           article.article_id,
                                "source_url":           article.source_url,
                                "canonical_entity_ids": article.canonical_entity_ids,
                                # Bonus context fields (useful for debugging / re-ranking)
                                "sentence_index":       child["sentence_index"],
                                "parent_chunk_index":   p_idx,
                            },
                        )
                    )

            # Accumulate for batch writes below
            all_parent_records.extend(article_parents)
            all_qdrant_points.extend(article_points)
            successful_ids.append(article.article_id)

            log.debug(
                "  ✓ Article %s → %d parent(s), %d child chunk(s).",
                article.article_id, len(article_parents), len(article_points),
            )

        except Exception as exc:
            log.error(
                "  ✗ Article %s failed during chunking/embedding: %s",
                article.article_id, exc, exc_info=True,
            )
            failed_ids.append(article.article_id)

    # ── Batch writes (exceptions propagate intentionally) ────────────────────
    #
    # NOTE on Qdrant / PostgreSQL consistency:
    #   These two systems are not in the same transaction. If pg_insert_parents
    #   succeeds but qdrant_upsert fails, the PG rows will be rolled back in
    #   main() (since autocommit=False), but any Qdrant points written before
    #   the error are permanent. On the next run those Qdrant points will be
    #   overwritten (upsert, same deterministic IDs) — a safe no-op.
    #   Parent PG rows will be re-inserted with ON CONFLICT DO NOTHING.
    # ─────────────────────────────────────────────────────────────────────────

    if all_parent_records:
        pg_insert_parents(conn, all_parent_records)

    if all_qdrant_points:
        qdrant_upsert(qdrant, all_qdrant_points)

    return successful_ids, failed_ids


# ═══════════════════════════════════════════════════════════════════════════
# Initialisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def init_postgres():
    """Open a PostgreSQL connection with manual transaction control."""
    log.info("Connecting to PostgreSQL…")
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False          # We commit explicitly after each batch
    log.info("PostgreSQL: connection established.")
    return conn


def init_qdrant() -> QdrantClient:
    """Connect to the local Qdrant instance."""
    log.info("Connecting to Qdrant at %s:%d…", QDRANT_HOST, QDRANT_PORT)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # Trigger a lightweight call to verify the connection is alive
    client.get_collections()
    log.info("Qdrant: connection established.")
    return client


def init_spacy(model_name: str = SPACY_MODEL):
    """
    Load the spaCy model and disable all pipeline components that are not
    required for sentence segmentation, improving throughput.

    We keep:
      tok2vec  — shared feature encoder (required by parser / senter)
      parser   — dependency parser (provides doc.sents in most models)
      senter   — lightweight sentence segmenter (fallback / alternative)
    """
    log.info("Loading spaCy model '%s'…", model_name)
    try:
        nlp = spacy.load(model_name)
    except OSError:
        log.critical(
            "spaCy model '%s' not found. Run:  python -m spacy download %s",
            model_name, model_name,
        )
        sys.exit(1)

    # Disable pipes that are not needed for sentence boundary detection
    keep_pipes = {"tok2vec", "parser", "senter"}
    pipes_to_disable = [p for p in nlp.pipe_names if p not in keep_pipes]
    for pipe_name in pipes_to_disable:
        try:
            nlp.disable_pipe(pipe_name)
        except Exception:
            pass  # Non-critical; some models may not have all named pipes

    # Increase max_length to handle very long news articles without errors
    nlp.max_length = 2_000_000

    log.info("spaCy: loaded (active pipes: %s).", list(nlp.pipe_names))
    return nlp


def init_embed_model(model_name: str = BGE_MODEL_NAME) -> BGEM3FlagModel:
    """
    Load the BGE-M3 model.
    First run will download ~2 GB of weights to the HuggingFace cache.
    use_fp16=True halves VRAM usage and speeds up GPU inference.
    On CPU-only machines the flag is silently ignored by the library.
    """
    log.info("Loading BGE-M3 model '%s' (first run downloads ~2 GB)…", model_name)
    model = BGEM3FlagModel(model_name, use_fp16=True)
    log.info("BGE-M3: model loaded.")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 65)
    log.info("FactLens Chunking Pipeline starting.")
    log.info("=" * 65)

    # ── 1. Initialise all resources ──────────────────────────────────────────
    try:
        conn = init_postgres()
    except psycopg2.OperationalError as exc:
        log.critical("Cannot connect to PostgreSQL: %s", exc)
        sys.exit(1)

    try:
        qdrant = init_qdrant()
        ensure_qdrant_collection(qdrant)
    except Exception as exc:
        log.critical("Cannot connect to / initialise Qdrant: %s", exc)
        conn.close()
        sys.exit(1)

    try:
        nlp         = init_spacy()
        embed_model = init_embed_model()
    except SystemExit:
        conn.close()
        raise  # propagate sys.exit() cleanly

    # ── 2. Main processing loop ──────────────────────────────────────────────
    total_ok:   int = 0
    total_fail: int = 0
    batch_num:  int = 0
    consecutive_batch_failures: int = 0

    try:
        while True:
            # Always fetch from the logical top — processed rows have
            # transitioned to SUCCESS_STATE or FAILURE_STATE and won't reappear.
            batch = fetch_batch(conn)

            if not batch:
                log.info(
                    "No more articles in state '%s'. Pipeline complete.", SOURCE_STATE
                )
                break

            batch_num += 1
            log.info(
                "══ Batch %d: %d article(s) fetched (state='%s') ══",
                batch_num, len(batch), SOURCE_STATE,
            )

            try:
                # Core pipeline — per-article errors are swallowed internally;
                # batch-level write errors propagate here.
                ok_ids, fail_ids = process_batch(batch, conn, qdrant, nlp, embed_model)

                # Update article states and commit the full batch atomically
                pg_update_states(conn, ok_ids,   SUCCESS_STATE)
                pg_update_states(conn, fail_ids,  FAILURE_STATE)
                conn.commit()

                total_ok   += len(ok_ids)
                total_fail += len(fail_ids)
                consecutive_batch_failures = 0   # reset on any successful commit

                log.info(
                    "Batch %d committed — ✓ %d succeeded  ✗ %d failed  "
                    "(running totals: %d ok / %d fail).",
                    batch_num, len(ok_ids), len(fail_ids), total_ok, total_fail,
                )

            except Exception as exc:
                # A batch-level failure (e.g. Qdrant unreachable, PG write error).
                # Roll back so no article state changes are committed in PG.
                # Articles retain SOURCE_STATE and will be retried next run.
                log.error(
                    "Batch %d: batch-level failure — rolling back. Error: %s",
                    batch_num, exc, exc_info=True,
                )
                try:
                    conn.rollback()
                except Exception as rb_exc:
                    log.warning("Rollback itself failed: %s", rb_exc)

                consecutive_batch_failures += 1
                if consecutive_batch_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                    log.critical(
                        "%d consecutive batch failures. Halting to avoid an "
                        "infinite retry loop. Fix the underlying issue and rerun.",
                        consecutive_batch_failures,
                    )
                    break

                log.info(
                    "Will retry next batch "
                    "(%d/%d consecutive failures so far).",
                    consecutive_batch_failures, MAX_CONSECUTIVE_BATCH_FAILURES,
                )
                continue

    except KeyboardInterrupt:
        log.warning("Interrupted by user — rolling back any open transaction.")
        try:
            conn.rollback()
        except Exception:
            pass

    except Exception as exc:
        log.critical("Unhandled exception in main loop: %s", exc, exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass

    finally:
        conn.close()
        log.info("PostgreSQL connection closed.")
        log.info(
            "Pipeline finished. Total: %d article(s) processed successfully, "
            "%d failed.",
            total_ok, total_fail,
        )


if __name__ == "__main__":
    main()
