"""
FactLens — ner_spacy.py

Named Entity Recognition worker using spaCy.

Processing flow per article
---------------------------
1. Fetch all articles in state ``'pending'``.
2. Run spaCy NER on ``raw_content``.
3. Write extracted entity lists back to the article's entity_* columns
   via ``ArticleRepository.update_entities()``.
4. Advance the article's state to ``'ner_done'`` via
   ``ArticleRepository.update_state()``.
5. Hand the ``by_label`` dict (label → [entity strings]) off to
   ``resolve_entities()`` so canonical entity IDs are created / matched.
6. On any error: set article state to ``'ner_failed'`` and continue.
"""

import logging

import spacy

from db.connection import Connection
from db.repositories.articles import ArticleRepository
from db.repositories.entities import EntityRepository
from entity_resolution import resolve_entities

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label ↔ column mapping
# ---------------------------------------------------------------------------

# spaCy label  →  articles table column
NER_LABELS: dict[str, str] = {
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

# Reverse: articles column  →  spaCy label
_FIELD_TO_LABEL: dict[str, str] = {v: k for k, v in NER_LABELS.items()}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str = "en_core_web_trf") -> spacy.language.Language:
    """Load and return a spaCy model by name."""
    try:
        nlp = spacy.load(model_name)
        log.info("Loaded spaCy model: %s", model_name)
        return nlp
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' not installed.\n"
            f"Run:  python -m spacy download {model_name}"
        )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_entities(
    text: str,
    nlp: spacy.language.Language,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Run NER on ``text`` and return two parallel views of the results.

    Returns
    -------
    column_keyed : dict
        ``{articles_column_name: [entity_strings, …]}`` — ready to pass
        to ``ArticleRepository.update_entities()``.
        Example::

            {"entity_org": ["Apple", "Google"], "entity_person": ["Tim Cook"]}

    label_keyed : dict
        ``{spacy_label: [entity_strings, …]}`` — ready to pass to
        ``resolve_entities()``.
        Example::

            {"ORG": ["Apple", "Google"], "PERSON": ["Tim Cook"]}

    Both dicts contain **only labels / columns that had at least one
    entity**; empty labels are omitted entirely.  Values are
    deduplicated while preserving order of first occurrence.
    """
    doc = nlp(text)

    # Accumulate into column-keyed structure first (avoids double iteration).
    column_keyed: dict[str, list[str]] = {}

    for ent in doc.ents:
        field = NER_LABELS.get(ent.label_)
        if field is None:
            continue  # unknown label — ignore

        cleaned = ent.text.strip()
        if not cleaned:
            continue

        if field not in column_keyed:
            column_keyed[field] = []

        if cleaned not in column_keyed[field]:
            column_keyed[field].append(cleaned)

    # Build label-keyed view from the column-keyed one — O(n) single pass.
    label_keyed: dict[str, list[str]] = {
        _FIELD_TO_LABEL[field]: names
        for field, names in column_keyed.items()
    }

    return column_keyed, label_keyed


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def process_pending_articles(
    article_repo: ArticleRepository,
    entity_repo: EntityRepository,
    model_name: str = "en_core_web_trf",
) -> None:
    """
    Fetch all ``'pending'`` articles, run NER, persist results, and
    hand entities off to entity resolution.

    Parameters
    ----------
    article_repo : ArticleRepository
        Repository for reading and updating the articles table.
    entity_repo : EntityRepository
        Repository for reading and writing the entities table.
        Passed through to ``resolve_entities()``.
    model_name : str
        spaCy model to load (default: ``en_core_web_trf``).
    """
    log.info("Loading spaCy model '%s' …", model_name)
    nlp = load_model(model_name)

    log.info("Fetching pending articles …")
    articles = article_repo.get_by_state("pending")
    log.info("Found %d pending article(s).", len(articles))

    for idx, article in enumerate(articles, start=1):

        # ── Safely extract the article id ──────────────────────────────
        try:
            article_id: int = article["id"]
        except (TypeError, KeyError):
            article_id = article[0]

        url   = article.get("url", "<no url>")
        title = article.get("title", "<no title>")

        log.info("[%d/%d] Processing article id=%s url=%s", idx, len(articles), article_id, url)

        raw_content: str | None = article.get("raw_content")

        if not raw_content:
            log.warning("  Skipping article id=%s — empty raw_content.", article_id)
            article_repo.update_state(article_id, "ner_failed")
            continue

        try:
            # ── Step 1: Run NER ────────────────────────────────────────
            column_keyed, label_keyed = extract_entities(raw_content, nlp)

            total_entities = sum(len(v) for v in column_keyed.values())
            log.info("  Extracted %d entity mention(s) across %d label(s).",
                     total_entities, len(column_keyed))

            # ── Step 2: Persist entity lists to the article row ────────
            if column_keyed:
                article_repo.update_entities(article_id, column_keyed)
                log.debug("  Wrote entity columns for article_id=%s.", article_id)

            # ── Step 3: Advance state to 'ner_done' ────────────────────
            article_repo.update_state(article_id, "ner_done")
            log.info("  Article id=%s → state='ner_done'.", article_id)

            # ── Step 4: Entity resolution ──────────────────────────────
            if label_keyed:
                resolved = resolve_entities(label_keyed, entity_repo)
                log.info(
                    "  Entity resolution complete: %d label(s) resolved, "
                    "%d canonical ID(s) total.",
                    len(resolved),
                    sum(len(ids) for ids in resolved.values()),
                )
            else:
                log.info("  No entities to resolve for article id=%s.", article_id)

        except Exception as exc:  # noqa: BLE001
            log.error(
                "  Error processing article id=%s: %s — marking as 'ner_failed'.",
                article_id,
                exc,
                exc_info=True,
            )
            article_repo.update_state(article_id, "ner_failed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    conn         = Connection()
    article_repo = ArticleRepository(conn)
    entity_repo  = EntityRepository(conn)

    process_pending_articles(article_repo, entity_repo)
