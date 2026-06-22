"""
FactLens — entity_resolution.py

Entity Resolution (ER) worker.

Execution model
---------------
* Synchronous — called once per article after the spaCy NER step.
* Receives a ``by_label`` dict (spaCy label → [raw entity strings])
  directly from ``ner_spacy.process_pending_articles``.
* Uses ``EntityRepository`` for all DB I/O — no raw psycopg2 cursors.

Algorithm (per label)
---------------------
1. **Blocking** — fetch every known alias for that label via
   ``EntityRepository.get_by_type()``.  Scoping by type keeps the
   candidate set small (no comparing org names against person names).
2. **Per-mention loop** — for each raw entity string:
   a. Normalise (lowercase → strip suffixes → strip punctuation).
   b. Check the *within-run cache* to avoid redundant work for repeated
      mentions in the same article.
   c. Run ``rapidfuzz.fuzz.token_sort_ratio`` against every row in the
      in-memory block.
   d. **Match (score ≥ 85):** record the existing canonical ID and
      **append the raw text as a new alias** via
      ``EntityRepository.append_entity_name()``, so future articles can
      also match against this surface form.
   e. **No match:** generate the next sequential canonical ID via
      ``EntityRepository.get_next_canonical_id()``, insert the new entity,
      and add it to the in-memory block for subsequent mentions.
3. Return a deduplicated, label-keyed mapping of canonical ID lists.

Dependencies
------------
    pip install rapidfuzz
"""

from __future__ import annotations

import logging
import re
import string
from typing import Dict, List, Optional

from rapidfuzz import fuzz

from db.repositories.entities import EntityRepository, VALID_ENTITY_TYPES
from db.models import EntityRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fuzzy match threshold (inclusive).  Scores ≥ this are treated as the
#: same canonical entity.
MATCH_THRESHOLD: int = 85

#: Ordered suffix patterns to strip during normalisation.  More-specific
#: patterns appear before their subsets (e.g. "private limited" → "limited").
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

_SUFFIX_RE: re.Pattern = re.compile(
    "|".join(_SUFFIX_PATTERNS),
    flags=re.IGNORECASE,
)

_PUNCT_TABLE: dict = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_entity_text(text: str) -> str:
    """
    Normalise an entity string for fuzzy comparison.

    Pipeline:
    1. Lowercase.
    2. Strip common Indian corporate / institutional suffixes iteratively
       until stable (handles "Reliance Industries Pvt. Ltd." → "reliance").
    3. Remove all punctuation.
    4. Collapse whitespace.

    Examples
    --------
    >>> normalise_entity_text("Reliance Industries Ltd.")
    'reliance'
    >>> normalise_entity_text("  Apple Inc.  ")
    'apple'
    >>> normalise_entity_text("Ministry of Finance")
    'finance'
    """
    text = text.lower()
    previous = None
    while previous != text:
        previous = text
        text = _SUFFIX_RE.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _find_best_match(
    normalised_candidate: str,
    block: List[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """
    Run ``token_sort_ratio`` against every row in ``block`` and return the
    best-matching row if its score meets ``MATCH_THRESHOLD``.

    ``block`` is produced by ``EntityRepository.get_by_type()`` which unnests
    the ``entity_name`` array, so each row already holds a single string.

    Parameters
    ----------
    normalised_candidate : str
        The already-normalised incoming entity text.
    block : list[dict]
        Existing entity rows with keys ``id``, ``entity_name``, ``entity_type``.

    Returns
    -------
    dict or None
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
    entities_by_label: Dict[str, List[str]],
    repo: EntityRepository,
) -> Dict[str, List[str]]:
    """
    Resolve raw entity strings to canonical IDs in the ``entities`` table.

    Called once per article — receives the ``label_keyed`` dict returned by
    ``ner_spacy.extract_entities()``.

    Parameters
    ----------
    entities_by_label : dict
        Mapping of ``{spacy_label: [raw_entity_string, …]}``.
        Only labels with at least one mention need to be included.
        Example::

            {
                "ORG":    ["Apple Inc.", "Apple"],
                "PERSON": ["Tim Cook"],
                "GPE":    ["Cupertino"],
            }

        All labels must be valid spaCy NER types (one of the 18 OntoNotes
        labels).  Unknown labels are skipped with a warning.

    repo : EntityRepository
        Live repository instance used for all DB reads and writes.

    Returns
    -------
    dict
        ``{spacy_label: [canonical_id, …]}`` — only labels that had at
        least one resolvable entity.  Canonical IDs within each list are
        deduplicated and ordered by first occurrence.
        Example::

            {
                "ORG":    ["ORG_3", "ORG_11"],
                "PERSON": ["PERSON_7"],
                "GPE":    ["GPE_2"],
            }
    """
    if not entities_by_label:
        logger.info("resolve_entities: empty input — nothing to resolve.")
        return {}

    # label → ordered-unique list of canonical IDs for this article.
    resolved_ids: Dict[str, List[str]] = {}

    # within_run_cache: normalised_text → canonical_id
    # Prevents duplicate DB round-trips for repeated mentions in one article.
    within_run_cache: Dict[str, str] = {}

    for label, raw_mentions in entities_by_label.items():

        # ── Validate label ─────────────────────────────────────────────
        if label not in VALID_ENTITY_TYPES:
            logger.warning(
                "resolve_entities: unknown entity_type '%s' — skipping.", label
            )
            continue

        if not raw_mentions:
            continue

        logger.info(
            "Processing label '%s' with %d mention(s).", label, len(raw_mentions)
        )

        # ── Step 1: Blocking — load existing entities for this type ────
        # get_by_type() unnests entity_name array → one row per alias.
        block: List[Dict[str, str]] = repo.get_by_type(label)
        logger.debug(
            "Fetched %d alias row(s) for label '%s' from DB.", len(block), label
        )

        label_ids: List[str] = []

        for raw_text in raw_mentions:

            # ── Step 2a: Normalise ─────────────────────────────────────
            normalised = normalise_entity_text(raw_text)

            if not normalised:
                logger.warning(
                    "Entity '%s' (label=%s) normalised to empty string — skipping.",
                    raw_text, label,
                )
                continue

            # ── Step 2b: Within-run cache ──────────────────────────────
            if normalised in within_run_cache:
                canonical_id = within_run_cache[normalised]
                logger.debug("Cache hit '%s' → %s", normalised, canonical_id)
                if canonical_id not in label_ids:
                    label_ids.append(canonical_id)
                continue

            # ── Step 2c: Fuzzy match against in-memory block ───────────
            match = _find_best_match(normalised, block)

            if match is not None:
                # ── Scenario A: Match found ────────────────────────────
                canonical_id = match["id"]

                # Append the new surface form as an alias so future
                # articles can also match against this exact string.
                repo.append_entity_name(canonical_id, raw_text)

                # Add the new alias to the in-memory block so later
                # mentions within *this* article can also match it.
                block.append({
                    "id": canonical_id,
                    "entity_name": raw_text,
                    "entity_type": label,
                })

                logger.info(
                    "MATCHED  '%s' → existing entity id=%s (alias added).",
                    raw_text, canonical_id,
                )

            else:
                # ── Scenario B: No match — create new canonical entity ─
                canonical_id = repo.get_next_canonical_id(label)

                new_entity = EntityRow(
                    id=canonical_id,
                    entity_name=raw_text,
                    entity_type=label,
                )

                # upsert() returns the live canonical_id (handles the
                # rare race where two workers insert the same entity
                # concurrently — the loser gets the winner's ID back).
                canonical_id = repo.upsert(new_entity)

                # Expand the in-memory block so subsequent mentions in
                # this same article can match the new entity.
                block.append({
                    "id": canonical_id,
                    "entity_name": raw_text,
                    "entity_type": label,
                })

                logger.info(
                    "INSERTED '%s' as new entity id=%s.", raw_text, canonical_id
                )

            # ── Update cache and output accumulators ───────────────────
            within_run_cache[normalised] = canonical_id
            if canonical_id not in label_ids:
                label_ids.append(canonical_id)

        if label_ids:
            resolved_ids[label] = label_ids

    logger.info(
        "resolve_entities complete: %d label(s), %d total canonical ID(s).",
        len(resolved_ids),
        sum(len(v) for v in resolved_ids.values()),
    )
    return resolved_ids


# ---------------------------------------------------------------------------
# Convenience helper — maps resolved IDs to articles table column names
# ---------------------------------------------------------------------------

LABEL_TO_COLUMN: Dict[str, str] = {
    "PERSON":       "entity_person",
    "NORP":         "entity_norp",
    "FAC":          "entity_fac",
    "ORG":          "entity_org",
    "GPE":          "entity_gpe",
    "LOC":          "entity_loc",
    "PRODUCT":      "entity_product",
    "EVENT":        "entity_event",
    "WORK_OF_ART":  "entity_work_of_art",
    "LAW":          "entity_law",
    "LANGUAGE":     "entity_language",
    "DATE":         "entity_date",
    "TIME":         "entity_time",
    "PERCENT":      "entity_percent",
    "MONEY":        "entity_money",
    "QUANTITY":     "entity_quantity",
    "ORDINAL":      "entity_ordinal",
    "CARDINAL":     "entity_cardinal",
}


def build_article_update_payload(
    resolved_ids: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Translate the ``resolve_entities`` return value into a column-keyed dict
    suitable for ``ArticleRepository.update_entities()``.

    Pure helper — no DB I/O.

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
