# """ NER with spacy"""
# import spacy
# from typing import List, Dict


# # All 18 NER labels supported by en_core_web_trf (OntoNotes 5 schema)
# NER_LABELS = {
#     "PERSON":     "People, including fictional characters",
#     "NORP":       "Nationalities, religious or political groups",
#     "FAC":        "Buildings, airports, highways, bridges, etc.",
#     "ORG":        "Companies, agencies, institutions",
#     "GPE":        "Countries, cities, states (geo-political entities)",
#     "LOC":        "Non-GPE locations: mountain ranges, bodies of water",
#     "PRODUCT":    "Objects, vehicles, foods (not services)",
#     "EVENT":      "Named hurricanes, battles, wars, sports events",
#     "WORK_OF_ART":"Titles of books, songs, paintings, etc.",
#     "LAW":        "Named documents made into laws",
#     "LANGUAGE":   "Any named language",
#     "DATE":       "Absolute or relative dates or periods",
#     "TIME":       "Times smaller than a day",
#     "PERCENT":    "Percentage, including '%'",
#     "MONEY":      "Monetary values, including unit",
#     "QUANTITY":   "Measurements (weight, distance, etc.)",
#     "ORDINAL":    "'first', 'second', 'third', etc.",
#     "CARDINAL":   "Numerals that don't fall under another type",
# }


# def load_model(model_name: str = "en_core_web_trf") -> spacy.language.Language:
#     """Load the spaCy transformer model."""
#     try:
#         nlp = spacy.load(model_name)
#         print(f"✅ Model '{model_name}' loaded successfully.\n")
#         return nlp
#     except OSError:
#         raise OSError(
#             f"Model '{model_name}' not found.\n"
#             f"Run: python -m spacy download {model_name}"
#         )


# def perform_ner(text: str, nlp: spacy.language.Language = None) -> Dict:
#     """
#     Perform Named Entity Recognition on the input text.

#     Args:
#         text (str): Input text to analyse.
#         nlp:        Pre-loaded spaCy model. If None, loads en_core_web_trf.

#     Returns:
#         dict with keys:
#             - "entities"      : list of entity dicts (text, label, description,
#                                 start_char, end_char)
#             - "by_label"      : dict mapping each of the 18 labels to a list
#                                 of entity strings found under that label
#             - "label_counts"  : dict mapping each label to its entity count
#             - "total_entities": total number of entities found
#     """
#     if nlp is None:
#         nlp = load_model()

#     doc = nlp(text)

#     # Collect all detected entities
#     entities: List[Dict] = []
#     for ent in doc.ents:
#         entities.append({
#             "text":        ent.text,
#             "label":       ent.label_,
#             "description": NER_LABELS.get(ent.label_, "Unknown label"),
#             "start_char":  ent.start_char,
#             "end_char":    ent.end_char,
#         })

#     # Group by all 18 labels (empty list if none found)
#     by_label: Dict[str, List[str]] = {label: [] for label in NER_LABELS}
#     for ent in entities:
#         if ent["label"] in by_label:
#             by_label[ent["label"]].append(ent["text"])

#     label_counts = {label: len(items) for label, items in by_label.items()}

#     return {
#         "entities":       entities,
#         "by_label":       by_label,
#         "label_counts":   label_counts,
#         "total_entities": len(entities),
#     }


# def print_ner_results(result: Dict) -> None:
#     """Pretty-print the NER results returned by perform_ner()."""
#     print(f"{'='*60}")
#     print(f"  Total entities found: {result['total_entities']}")
#     print(f"{'='*60}\n")

#     # ── Flat list ──────────────────────────────────────────────
#     if result["entities"]:
#         print("📌 All Entities (in order of appearance):")
#         print(f"  {'Entity':<30} {'Label':<14} {'Description'}")
#         print(f"  {'-'*28} {'-'*12} {'-'*35}")
#         for ent in result["entities"]:
#             print(f"  {ent['text']:<30} {ent['label']:<14} {ent['description']}")
#     else:
#         print("  No entities detected.")

#     # ── Grouped by all 18 labels ───────────────────────────────
#     print(f"\n{'='*60}")
#     print("📂 Entities Grouped by All 18 NER Labels:")
#     print(f"{'='*60}")
#     for label, description in NER_LABELS.items():
#         found = result["by_label"][label]
#         count = result["label_counts"][label]
#         status = ", ".join(found) if found else "—"
#         print(f"\n  [{label}] ({count} found)  →  {description}")
#         print(f"    {status}")


# # ── Demo ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     sample_text = """Apple Inc. announced the release of the new iPhone 15 Pro on September 12, 2023."""
        
#     print("Input Text:")
#     print(f"  {sample_text}\n")

#     nlp = load_model("en_core_web_trf")
#     result = perform_ner(sample_text, nlp)
#     print_ner_results(result)










import spacy
from typing import Dict
from db.repositories.articles import ArticleRepository
from db.connection import Connection


NER_LABELS = {
    "PERSON": "entity_person",
    "NORP": "entity_norp",
    "FAC": "entity_fac",
    "ORG": "entity_org",
    "GPE": "entity_gpe",
    "LOC": "entity_loc",
    "PRODUCT": "entity_product",
    "EVENT": "entity_event",
    "WORK_OF_ART": "entity_work_of_art",
    "LAW": "entity_law",
    "LANGUAGE": "entity_language",
    "DATE": "entity_date",
    "TIME": "entity_time",
    "PERCENT": "entity_percent",
    "MONEY": "entity_money",
    "QUANTITY": "entity_quantity",
    "ORDINAL": "entity_ordinal",
    "CARDINAL": "entity_cardinal",
}


def load_model(model_name: str = "en_core_web_trf"):
    try:
        nlp = spacy.load(model_name)
        print(f"✅ Loaded model: {model_name}")
        return nlp
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' not installed.\n"
            f"Run:\n"
            f"python -m spacy download {model_name}"
        )


def extract_entities(text: str, nlp) -> Dict[str, list[str]]:
    """
    Extract entities from raw article content and return
    them mapped to ArticleRow entity fields.
    """

    doc = nlp(text)

    entities = {
        field_name: []
        for field_name in NER_LABELS.values()
    }

    for ent in doc.ents:
        field_name = NER_LABELS.get(ent.label_)

        if field_name:
            entities[field_name].append(ent.text.strip())

    # Optional: remove duplicates while preserving order
    for field_name, values in entities.items():
        entities[field_name] = list(dict.fromkeys(values))

    return entities


def process_pending_articles(article_repo):
    """
    Fetch all pending articles and perform NER on raw_content.
    """

    print("Loading spaCy model...")
    nlp = load_model()

    print("Fetching pending articles...")
    articles = article_repo.get_by_state("pending")

    print(f"Found {len(articles)} pending articles.\n")

    for idx, article in enumerate(articles, start=1):

        raw_content = article.get("raw_content")

        if not raw_content:
            print(f"[{idx}] Skipping article (empty raw_content)")
            continue

        print("=" * 100)
        print(f"ARTICLE #{idx}")
        print(f"URL: {article.get('url')}")
        print(f"TITLE: {article.get('title')}")
        print("=" * 100)

        entity_data = extract_entities(raw_content, nlp)

        total_entities = sum(
            len(values)
            for values in entity_data.values()
        )

        print(f"Total entities found: {total_entities}\n")

        for field_name, values in entity_data.items():
            if values:
                print(f"{field_name}:")
                for value in values:
                    print(f"  - {value}")
                print()

        print("\n")


if __name__ == "__main__":
    conn = Connection()
    article_repo = ArticleRepository(conn)

    process_pending_articles(article_repo)