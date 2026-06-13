"""
entity_resolution.py
====================
Production-grade Entity Resolution (ER) worker for a media and legal NLP pipeline.

Execution model
---------------
* Synchronous — designed to run as a worker step after the async ingest layer
  has already committed the article and the spaCy NER pipeline has returned its
  output dict.
* One article at a time — called once per article from whatever job-queue
  consumer drives the processing loop.
* psycopg2 — standard synchronous driver; caller supplies an open connection
  (or one drawn from a pool context manager).

Responsibilities
----------------
1. Normalise incoming entity text (lowercase, strip punctuation, strip common
   Indian corporate/institutional suffixes).
2. For each entity label (ORG, PERSON, …), fetch the existing entities of that
   type from the DB into an in-memory block (blocking optimisation).
3. Run rapidfuzz token_sort_ratio against the normalised block.
4. Resolve to an existing canonical ID (score ≥ 85) **or** generate a new
   sequential ID and insert the entity (score < 85).
5. Return a mapping of {label: [canonical_id, …]} for the caller to use when
   updating the articles row.

Out of scope
------------
* Updating the articles table — the caller owns that responsibility.
* Any async/threading logic.

Dependencies
------------
    pip install rapidfuzz psycopg2-binary
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import psycopg2
from rapidfuzz import fuzz

if TYPE_CHECKING:
    # psycopg2 connection/cursor types are only used for annotations.
    from psycopg2.extensions import connection as PgConnection

# ---------------------------------------------------------------------------
# Module-level logger — caller configures handlers / level.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fuzzy match threshold (inclusive).  Scores ≥ this value are treated as the
#: same canonical entity.
MATCH_THRESHOLD: int = 85

#: Ordered list of Indian corporate / institutional suffixes to strip during
#: normalisation.  Longer / more-specific patterns are listed first so they
#: are matched before their subsets (e.g. "private limited" before "limited").
_SUFFIX_PATTERNS: List[str] = [
    r"\bprivate limited\b",
    r"\bpvt\.?\s*ltd\.?\b",
    r"\bpvt\.?\s*limited\b",
    r"\bprivate ltd\.?\b",
    r"\bllp\b",
    r"\blimited\b",
    r"\bltd\.?\b",
    r"\binc\.?\b",
    r"\bcorp\.?\b",
    r"\bcorporation\b",
    r"\bministry of\b",
    r"\bdept\.?\s*of\b",
    r"\bdepartment of\b",
    r"\benterprises\b",
    r"\benterprise\b",
    r"\bfoundation\b",
    r"\bindustries\b",
    r"\bgroup\b",
    r"\bassociates\b",
    r"\bservices\b",
]

# Pre-compile all suffix patterns into a single alternation regex for speed.
_SUFFIX_RE: re.Pattern = re.compile(
    "|".join(_SUFFIX_PATTERNS),
    flags=re.IGNORECASE,
)

# Strip punctuation using str.translate — faster than regex for this step.
_PUNCT_TABLE: dict = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# EntityRow dataclass — mirrors the entities table columns used for upsert.
# ---------------------------------------------------------------------------

@dataclass
class EntityRow:
    """
    Lightweight value object representing a row in the ``entities`` table.

    Attributes
    ----------
    id : str
        Canonical ID in ``{LABEL}_{COUNT}`` format, e.g. ``ORG_11``.
    entity_name : str
        The canonical (un-normalised) display name for the entity.
    entity_type : str
        The spaCy NER label, e.g. ``ORG``, ``PERSON``.
    """

    id: str
    entity_name: str
    entity_type: str


# ---------------------------------------------------------------------------
# EntityRepository — external dependency stub.
# The real implementation lives in the repository layer; we define only the
# interface here so this module can import and type-check cleanly.
# ---------------------------------------------------------------------------

class EntityRepository:
    """
    Insert / upsert / lookup interface for the ``entities`` table.

    This class is treated as an **external dependency** — only the interface
    contract is defined here.  The production body is provided by the
    repository layer (injected at call-site).

    Parameters
    ----------
    conn : PgConnection
        An open psycopg2 connection (or one drawn from a connection pool).
    """

    def __init__(self, conn: "PgConnection") -> None:
        self._conn = conn

    def upsert(self, entity: EntityRow) -> str:
        """
        Insert a new entity row or return the existing canonical ID on conflict.

        The underlying SQL performs an INSERT … ON CONFLICT (entity_name,
        entity_type) DO NOTHING, then SELECTs back the id so the caller always
        receives a valid canonical ID regardless of whether the row was freshly
        inserted or already present.

        Parameters
        ----------
        entity : EntityRow
            Fully populated entity value object.  ``entity.id`` must be the
            pre-generated canonical ID string (e.g. ``ORG_11``).

        Returns
        -------
        str
            The canonical ID string for the resolved entity, e.g. ``ORG_11``.

        Raises
        ------
        psycopg2.DatabaseError
            Propagated as-is; the caller (``resolve_entities``) owns rollback.
        """
        raise NotImplementedError(
            "EntityRepository.upsert() must be implemented by the repository layer."
        )


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalise_entity_text(text: str) -> str:
    """
    Normalise an entity string for fuzzy comparison.

    Pipeline (applied in order):
    1. Lowercase.
    2. Strip common Indian corporate / institutional suffixes
       (e.g. ``pvt ltd``, ``private limited``, ``ministry of``).
    3. Remove all punctuation characters.
    4. Collapse multiple whitespace runs to a single space and strip edges.

    Parameters
    ----------
    text : str
        Raw entity string as returned by spaCy, e.g. ``"Reliance Industries Ltd."``.

    Returns
    -------
    str
        Normalised string, e.g. ``"reliance"``.

    Examples
    --------
    >>> normalise_entity_text("Reliance Industries Ltd.")
    'reliance'
    >>> normalise_entity_text("Ministry of Finance")
    'finance'
    >>> normalise_entity_text("  Apple Inc.  ")
    'apple'
    """
    text = text.lower()
    # Iteratively remove suffixes until the string stabilises — handles
    # cases like "Reliance Industries Pvt. Ltd." where two suffixes appear.
    previous = None
    while previous != text:
        previous = text
        text = _SUFFIX_RE.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _fetch_block_for_type(
    conn: "PgConnection",
    entity_type: str,
) -> List[Dict[str, str]]:
    """
    Fetch all existing entities of ``entity_type`` from the ``entities`` table.

    This is the *blocking optimisation* step: by scoping the lookup to a single
    label (e.g. ``ORG``), we avoid comparing an incoming organisation name
    against thousands of person names or GPEs.

    Parameters
    ----------
    conn : PgConnection
        Open psycopg2 connection.
    entity_type : str
        The spaCy label to filter by, e.g. ``"ORG"``.

    Returns
    -------
    list of dict
        Each dict has keys ``id``, ``entity_name``, ``entity_type``.
        Returns an empty list if the table is empty for that type.

    Raises
    ------
    psycopg2.DatabaseError
        Propagated to the caller for centralised error handling.
    """
    query = """
        SELECT id, entity_name, entity_type
        FROM   entities
        WHERE  entity_type = %s
        ORDER  BY id;
    """
    with conn.cursor() as cur:
        cur.execute(query, (entity_type,))
        rows = cur.fetchall()

    return [
        {"id": row[0], "entity_name": row[1], "entity_type": row[2]}
        for row in rows
    ]


def _generate_next_canonical_id(
    conn: "PgConnection",
    entity_type: str,
) -> str:
    """
    Generate the next sequential canonical ID for ``entity_type``.

    Strategy
    --------
    The canonical ID format is ``{LABEL}_{COUNT}`` (e.g. ``ORG_11``).
    We extract the integer suffix from every existing ID of the given type,
    take the maximum, add 1, and format the result.

    The ``SPLIT_PART`` / ``CAST`` approach is done entirely in SQL to avoid
    Python-side parsing of potentially malformed IDs; the ``COALESCE`` guards
    against an empty table (returns 0, so the first ID becomes ``{TYPE}_1``).

    Parameters
    ----------
    conn : PgConnection
        Open psycopg2 connection.
    entity_type : str
        The spaCy label, e.g. ``"PERSON"``.

    Returns
    -------
    str
        Next canonical ID string, e.g. ``"PERSON_46"``.

    Raises
    ------
    psycopg2.DatabaseError
        Propagated to the caller.
    """
    query = """
        SELECT COALESCE(
            MAX(
                CAST(
                    SPLIT_PART(id, '_', 2) AS INTEGER
                )
            ),
            0
        ) AS max_seq
        FROM entities
        WHERE entity_type = %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (entity_type,))
        result = cur.fetchone()

    max_seq: int = result[0] if result else 0
    next_seq: int = max_seq + 1
    return f"{entity_type}_{next_seq}"


# ---------------------------------------------------------------------------
# Core fuzzy-matching logic
# ---------------------------------------------------------------------------

def _find_best_match(
    normalised_candidate: str,
    block: List[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """
    Run ``rapidfuzz.fuzz.token_sort_ratio`` against every entity in ``block``
    and return the best-matching row if its score meets ``MATCH_THRESHOLD``.

    ``token_sort_ratio`` is chosen because it is robust to token-order
    variations (e.g. "Adani Gautam" vs "Gautam Adani") and performs well on
    the short, suffix-stripped strings produced by ``normalise_entity_text``.

    Parameters
    ----------
    normalised_candidate : str
        The normalised incoming entity text.
    block : list of dict
        Existing entities of the same type fetched by ``_fetch_block_for_type``.

    Returns
    -------
    dict or None
        The best-matching entity dict (keys: ``id``, ``entity_name``,
        ``entity_type``) if score ≥ ``MATCH_THRESHOLD``, else ``None``.
    """
    best_score: float = 0.0
    best_match: Optional[Dict[str, str]] = None

    for existing in block:
        normalised_existing = normalise_entity_text(existing["entity_name"])
        score = fuzz.token_sort_ratio(normalised_candidate, normalised_existing)

        if score > best_score:
            best_score = score
            best_match = existing

    if best_score >= MATCH_THRESHOLD:
        logger.debug(
            "Fuzzy match: '%s' → '%s' (id=%s, score=%.1f)",
            normalised_candidate,
            best_match["entity_name"],  # type: ignore[index]
            best_match["id"],           # type: ignore[index]
            best_score,
        )
        return best_match

    logger.debug(
        "No match for '%s' (best_score=%.1f < threshold=%d)",
        normalised_candidate,
        best_score,
        MATCH_THRESHOLD,
    )
    return None


# ---------------------------------------------------------------------------
# Main resolution function
# ---------------------------------------------------------------------------

def resolve_entities(
    ner_output: Dict,
    conn: "PgConnection",
    repo: Optional[EntityRepository] = None,
) -> Dict[str, List[str]]:
    """
    Resolve all entities in a single article's NER output to canonical IDs.

    This is the primary entry point.  It is designed to be called **once per
    article** from the synchronous processing worker.

    Algorithm
    ---------
    For each entity label that has at least one detected mention:

    1. **Blocking** — fetch the existing entity block for that label from the
       DB (``_fetch_block_for_type``).  Only entities of the same type are
       compared, keeping the candidate set small.

    2. **Per-mention loop** — for each raw entity string under that label:

       a. Normalise the text (``normalise_entity_text``).
       b. Check the *within-run cache* first — if this exact normalised string
          was already resolved during this article's execution, reuse the
          canonical ID without another DB round-trip.
       c. Run fuzzy matching against the in-memory block
          (``_find_best_match``).
       d. **Scenario A (match found):** record the existing canonical ID.
       e. **Scenario B (no match):** generate the next sequential canonical ID
          (``_generate_next_canonical_id``), build an ``EntityRow``, call
          ``repo.upsert()``, append the new row to the in-memory block so
          subsequent mentions in the *same article* can match against it,
          and record the canonical ID.

    3. Return a deduplicated, label-keyed mapping of canonical ID lists.

    Parameters
    ----------
    ner_output : dict
        Direct output of ``perform_ner()`` — must contain keys ``"entities"``,
        ``"by_label"``, ``"label_counts"``, ``"total_entities"``.
    conn : PgConnection
        An open psycopg2 connection.  The caller is responsible for commit /
        rollback at the transaction boundary.  On ``psycopg2.DatabaseError``
        this function logs the error, rolls back the connection, and re-raises
        so the caller can handle the failure (e.g. mark the article as
        'failed').
    repo : EntityRepository, optional
        Injected repository instance.  If ``None``, a new instance is created
        from ``conn``.  Pass an explicit instance in tests to inject a mock.

    Returns
    -------
    dict
        Mapping of ``{entity_label: [canonical_id, …]}`` containing only
        labels that had at least one entity in the article.  Canonical IDs
        within each list are deduplicated and ordered by first occurrence.

        Example::

            {
                "ORG":    ["ORG_3", "ORG_11"],
                "PERSON": ["PERSON_7"],
                "GPE":    ["GPE_2", "GPE_14"],
            }

    Raises
    ------
    psycopg2.DatabaseError
        Re-raised after rollback when any DB operation fails.  The caller
        should catch this and mark the article's ``processing_status`` as
        ``'failed'``.
    ValueError
        If ``ner_output`` is missing required keys.
    """
    # ── Input validation ───────────────────────────────────────────────────
    required_keys = {"entities", "by_label", "label_counts", "total_entities"}
    missing = required_keys - set(ner_output.keys())
    if missing:
        raise ValueError(
            f"ner_output is missing required keys: {missing}. "
            "Ensure you pass the direct output of perform_ner()."
        )

    if ner_output["total_entities"] == 0:
        logger.info("resolve_entities: no entities in NER output — nothing to resolve.")
        return {}

    # ── Repository setup ───────────────────────────────────────────────────
    if repo is None:
        repo = EntityRepository(conn)

    # ── Resolution loop ────────────────────────────────────────────────────
    # resolved_ids: label → ordered-unique list of canonical IDs for this article.
    resolved_ids: Dict[str, List[str]] = {}

    # within_run_cache: normalised_text → canonical_id
    # Prevents duplicate DB lookups / inserts when the same entity appears
    # multiple times within a single article.
    within_run_cache: Dict[str, str] = {}

    try:
        for label, raw_mentions in ner_output["by_label"].items():
            if not raw_mentions:
                # Label present but no entities found for it — skip entirely.
                continue

            logger.info(
                "Processing label '%s' with %d mention(s).",
                label,
                len(raw_mentions),
            )

            # ── Step 1: Blocking — load existing entities for this type ────
            block: List[Dict[str, str]] = _fetch_block_for_type(conn, label)
            logger.debug(
                "Fetched %d existing '%s' entities from DB for blocking.",
                len(block),
                label,
            )

            # Accumulate canonical IDs for this label (deduplicated, ordered).
            label_ids: List[str] = []

            for raw_text in raw_mentions:
                # ── Step 2a: Normalise ─────────────────────────────────────
                normalised = normalise_entity_text(raw_text)

                if not normalised:
                    # After stripping suffixes, nothing remains (e.g. "Ltd.")
                    # — skip this mention rather than inserting a blank entity.
                    logger.warning(
                        "Entity '%s' (label=%s) reduced to empty string after "
                        "normalisation — skipping.",
                        raw_text,
                        label,
                    )
                    continue

                # ── Step 2b: Within-run cache check ───────────────────────
                if normalised in within_run_cache:
                    canonical_id = within_run_cache[normalised]
                    logger.debug(
                        "Cache hit for '%s' → %s", normalised, canonical_id
                    )
                    if canonical_id not in label_ids:
                        label_ids.append(canonical_id)
                    continue

                # ── Step 2c: Fuzzy match against in-memory block ───────────
                match = _find_best_match(normalised, block)

                if match is not None:
                    # ── Scenario A: Match found ────────────────────────────
                    canonical_id = match["id"]
                    logger.info(
                        "MATCHED  '%s' → '%s' (canonical_id=%s)",
                        raw_text,
                        match["entity_name"],
                        canonical_id,
                    )
                else:
                    # ── Scenario B: No match — create new canonical entity ─
                    canonical_id = _generate_next_canonical_id(conn, label)

                    new_entity = EntityRow(
                        id=canonical_id,
                        entity_name=raw_text,   # Store the original display name.
                        entity_type=label,
                    )

                    returned_id: str = repo.upsert(new_entity)

                    # Guard: if upsert resolved to an existing row on conflict
                    # (race-condition edge case in concurrent workers), use
                    # the ID the DB actually assigned.
                    canonical_id = returned_id

                    # Append the new entity to the in-memory block so it is
                    # visible to subsequent mentions within this same article
                    # without another DB fetch.
                    block.append(
                        {
                            "id": canonical_id,
                            "entity_name": raw_text,
                            "entity_type": label,
                        }
                    )

                    logger.info(
                        "INSERTED '%s' as new entity (canonical_id=%s)",
                        raw_text,
                        canonical_id,
                    )

                # ── Update cache and output accumulators ──────────────────
                within_run_cache[normalised] = canonical_id

                if canonical_id not in label_ids:
                    label_ids.append(canonical_id)

            if label_ids:
                resolved_ids[label] = label_ids

    except psycopg2.DatabaseError as exc:
        # ── Rollback on any DB failure ─────────────────────────────────────
        # We issue a rollback here because any partial inserts within this
        # function would leave the entities table in an inconsistent state
        # (e.g. a new canonical ID that no article row references yet).
        # The caller must catch this exception and mark the article as 'failed'.
        logger.error(
            "DatabaseError during entity resolution — rolling back. Error: %s",
            exc,
            exc_info=True,
        )
        try:
            conn.rollback()
        except psycopg2.Error as rb_exc:
            logger.error("Rollback itself failed: %s", rb_exc, exc_info=True)
        raise  # Re-raise so the calling layer can handle article status update.

    logger.info(
        "resolve_entities complete. Resolved %d label(s), %d total canonical IDs.",
        len(resolved_ids),
        sum(len(v) for v in resolved_ids.values()),
    )
    return resolved_ids


# ---------------------------------------------------------------------------
# Convenience wrapper — maps resolved IDs to articles table column names
# ---------------------------------------------------------------------------

# Maps spaCy label → articles table column name.
# Used by the caller when it needs to build the UPDATE statement for articles.
LABEL_TO_COLUMN: Dict[str, str] = {
    "PERSON":      "entity_person",
    "NORP":        "entity_norp",
    "FAC":         "entity_fac",
    "ORG":         "entity_org",
    "GPE":         "entity_gpe",
    "LOC":         "entity_loc",
    "PRODUCT":     "entity_product",
    "EVENT":       "entity_event",
    "WORK_OF_ART": "entity_work_of_art",
    "LAW":         "entity_law",
    "LANGUAGE":    "entity_language",
    "DATE":        "entity_date",
    "TIME":        "entity_time",
    "PERCENT":     "entity_percent",
    "MONEY":       "entity_money",
    "QUANTITY":    "entity_quantity",
    "ORDINAL":     "entity_ordinal",
    "CARDINAL":    "entity_cardinal",
}


def build_article_update_payload(
    resolved_ids: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Translate the ``resolve_entities`` return value into a column-keyed dict
    suitable for building the ``UPDATE articles SET …`` statement.

    This is a **pure helper** — no DB I/O.  The caller constructs and executes
    the actual SQL.

    Parameters
    ----------
    resolved_ids : dict
        Output of ``resolve_entities``, keyed by spaCy label.

    Returns
    -------
    dict
        Mapping of ``{articles_column_name: [canonical_id, …]}``.

    Example
    -------
    >>> build_article_update_payload({"ORG": ["ORG_3"], "PERSON": ["PERSON_7"]})
    {'entity_org': ['ORG_3'], 'entity_person': ['PERSON_7']}
    """
    return {
        LABEL_TO_COLUMN[label]: ids
        for label, ids in resolved_ids.items()
        if label in LABEL_TO_COLUMN
    }
