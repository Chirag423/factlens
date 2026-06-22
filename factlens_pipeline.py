"""
FactLens — factlens_pipeline.py

Political Intelligence & Fact-Verification Pipeline
====================================================
Interactive terminal script that chains Retrieval → Augmentation →
Generation into a single loop.  Type a natural-language claim or question
at the prompt; FactLens retrieves supporting context from Qdrant + PostgreSQL,
builds a structured prompt, calls the Sarvam LLM, and prints a verdict.

Pipeline
────────
  [terminal input]
         │
         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  RETRIEVAL                                                   │
  │  BGE-M3 encode query (dense + sparse)                        │
  │  Qdrant hybrid search — RRF fusion (dense "" + "hybrid-      │
  │    sparse")  →  top-5 child chunks                           │
  │  Deduplicate parent_ids → fetch parent texts (PostgreSQL)    │
  │  Fetch source URLs via article_ids (PostgreSQL)              │
  └──────────────────────────────────────────────────────────────┘
         │
         ▼  assembled_context + citations
  ┌──────────────────────────────────────────────────────────────┐
  │  AUGMENTATION                                                │
  │  Build system prompt  (FactLens persona + JSON contract)     │
  │  Build user prompt    (injected context + query)             │
  └──────────────────────────────────────────────────────────────┘
         │
         ▼  [system_prompt, user_prompt]
  ┌──────────────────────────────────────────────────────────────┐
  │  GENERATION                                                  │
  │  SarvamAI.chat.completions (model=sarvam-30b)                │
  │  Parse JSON response → verdict / confidence / reasoning /    │
  │    explanation                                               │
  └──────────────────────────────────────────────────────────────┘
         │
         ▼
  [terminal output]

Output shape (Option C — structured + natural language)
───────────────────────────────────────────────────────
  {
      "verdict":     "TRUE" | "FALSE" | "UNVERIFIABLE",
      "confidence":  float  0.0 – 1.0,
      "reasoning":   str    (2–3 sentence logical chain citing context),
      "explanation": str    (1–2 paragraph reader-friendly summary),
  }

Environment variables
─────────────────────
  DATABASE_URL    PostgreSQL DSN                    (required)
  SARVAM_API_KEY  Sarvam AI subscription key        (required)
  QDRANT_HOST     Qdrant host        (default: localhost)
  QDRANT_PORT     Qdrant port        (default: 6333)
  BGE_MODEL       BGE-M3 path        (default: BAAI/bge-m3)
  SARVAM_MODEL    Sarvam model name  (default: sarvam-30b)

requirements.txt additions (beyond chunker deps)
────────────────────────────────────────────────
  sarvamai>=0.1.11
"""

from __future__ import annotations

# ── Standard library ─────────────────────────────────────────────────────────
import json
import logging
import os
import re
import sys
import textwrap
from typing import Any

# ── Third-party ───────────────────────────────────────────────────────────────
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from sarvamai import SarvamAI

# ── Project modules ───────────────────────────────────────────────────────────
from db.connection import Connection
from db.repositories.articles import ArticleRepository
from db.repositories.chunks import ParentChunkRepository


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration — all tuneable via environment variables
# ═══════════════════════════════════════════════════════════════════════════

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME: str = "factlens_child_chunks"
PREFETCH_LIMIT:  int = 20   # candidates per branch before RRF
FINAL_LIMIT:     int = 5    # child chunks after fusion → max 5 unique parents

# ── BGE-M3 ────────────────────────────────────────────────────────────────────
BGE_MODEL_NAME: str = os.getenv("BGE_MODEL", "BAAI/bge-m3")

# ── Sarvam ────────────────────────────────────────────────────────────────────
SARVAM_MODEL: str = os.getenv("SARVAM_MODEL", "sarvam-30b")

# ── Output ────────────────────────────────────────────────────────────────────
_CONTEXT_SEPARATOR: str = "\n\n---\n\n"   # between parent paragraphs in prompt

_VERDICT_ICONS: dict[str, str] = {
    "TRUE":          "✅  TRUE",
    "FALSE":         "❌  FALSE",
    "UNVERIFIABLE":  "⚠️   UNVERIFIABLE",
}


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def _embed_query(
    model: BGEM3FlagModel,
    query: str,
) -> tuple[list[float], dict[int, float]]:
    """
    Encode *query* with BGE-M3 in hybrid mode.

    Returns
    ───────
    dense_vec      : list[float]        — 1024-dim cosine vector (semantic)
    sparse_weights : dict[int, float]   — token_id → IDF weight (syntactic)
    """
    log.debug("BGE-M3: encoding query…")
    output = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vec:      list[float]       = output["dense_vecs"][0].tolist()
    sparse_weights: dict[int, float]  = output["lexical_weights"][0]

    log.debug(
        "BGE-M3 encoded — dense dim: %d, sparse tokens: %d.",
        len(dense_vec),
        len(sparse_weights),
    )
    return dense_vec, sparse_weights


def _hybrid_search(
    qdrant:         QdrantClient,
    dense_vec:      list[float],
    sparse_weights: dict[int, float],
) -> list[qmodels.ScoredPoint]:
    """
    Two-branch prefetch + RRF fusion query against Qdrant.

    Branch 1 — Semantic (dense):
        Vector key ``""`` (1024-dim cosine BGE-M3).
        Captures conceptual / paraphrase similarity.

    Branch 2 — Syntactic (sparse):
        Vector key ``"hybrid-sparse"`` (BGE-M3 lexical weights, BM25-style).
        Captures exact-term / keyword overlap — critical for proper nouns,
        legislation names, and numeric claims that dense embeddings can miss.

    Fusion — Reciprocal Rank Fusion (RRF):
        Rank-position-based merge; robust to the different score scales of
        cosine vs. sparse dot product — no normalisation needed.
    """
    sparse_vector = qmodels.SparseVector(
        indices=list(sparse_weights.keys()),
        values=[float(v) for v in sparse_weights.values()],
    )

    log.debug(
        "Qdrant hybrid search — prefetch_limit=%d, final_limit=%d.",
        PREFETCH_LIMIT,
        FINAL_LIMIT,
    )

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            # ── Branch 1: Dense semantic ─────────────────────────────────────
            qmodels.Prefetch(
                query=dense_vec,
                using="",               # empty-string key = default dense vector
                limit=PREFETCH_LIMIT,
            ),
            # ── Branch 2: Sparse syntactic/lexical ───────────────────────────
            qmodels.Prefetch(
                query=sparse_vector,
                using="hybrid-sparse",  # sparse vector key configured at index time
                limit=PREFETCH_LIMIT,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=FINAL_LIMIT,
        with_payload=True,
    )

    return response.points


def retrieve(
    query:        str,
    embed_model:  BGEM3FlagModel,
    qdrant:       QdrantClient,
    article_repo: ArticleRepository,
    chunk_repo:   ParentChunkRepository,
) -> dict[str, Any]:
    """
    Full retrieval stage: embed → search → fetch parent texts + URLs.

    Returns
    ───────
    {
        "assembled_context": str,   — parent paragraphs joined by ---
        "citations": [
            {"parent_id": str, "article_id": int, "source_url": str},
            ...
        ],
    }
    Empty dicts are returned (no exception) when nothing is found.
    """
    log.info("Retrieval: starting for query='%s'", query[:120])

    # ── 1. Embed ──────────────────────────────────────────────────────────────
    dense_vec, sparse_weights = _embed_query(embed_model, query)

    # ── 2. Hybrid RRF search ──────────────────────────────────────────────────
    child_points = _hybrid_search(qdrant, dense_vec, sparse_weights)

    if not child_points:
        log.warning("Qdrant returned 0 results.")
        return {"assembled_context": "", "citations": []}

    log.debug("Qdrant: %d fused child chunk(s).", len(child_points))

    # ── 3. Deduplicate parent_ids (preserve RRF rank order) ───────────────────
    parent_ids_ordered:   list[str]      = []
    article_id_by_parent: dict[str, int] = {}
    seen:                 set[str]       = set()

    for point in child_points:
        payload    = point.payload or {}
        parent_id  = payload.get("parent_id")
        article_id = payload.get("article_id")

        if not parent_id:
            log.warning("Point %s missing parent_id in payload — skipping.", point.id)
            continue

        if parent_id not in seen:
            seen.add(parent_id)
            parent_ids_ordered.append(parent_id)
            article_id_by_parent[parent_id] = article_id  # type: ignore[assignment]

    if not parent_ids_ordered:
        log.warning("No valid parent_ids found in Qdrant payload.")
        return {"assembled_context": "", "citations": []}

    # ── 4. Fetch full parent texts from PostgreSQL ────────────────────────────
    parent_texts: dict[str, str] = chunk_repo.get_by_parent_ids(parent_ids_ordered)

    if not parent_texts:
        log.warning("parent_chunks table returned nothing for %s", parent_ids_ordered)
        return {"assembled_context": "", "citations": []}

    # ── 5. Fetch source URLs from articles table ──────────────────────────────
    unique_article_ids: list[int] = list(
        {aid for aid in article_id_by_parent.values() if aid is not None}
    )
    url_by_article_id: dict[int, str] = article_repo.get_urls_by_ids(unique_article_ids)

    # ── 6. Assemble ───────────────────────────────────────────────────────────
    context_parts: list[str]  = []
    citations:     list[dict] = []

    for pid in parent_ids_ordered:
        text = parent_texts.get(pid)
        if not text:
            log.warning("parent_id %s missing from parent_chunks — skipping.", pid)
            continue

        aid: int | None = article_id_by_parent.get(pid)
        url: str        = url_by_article_id.get(aid, "") if aid is not None else ""

        context_parts.append(text)
        citations.append({"parent_id": pid, "article_id": aid, "source_url": url})

    log.info(
        "Retrieval complete — %d parent chunk(s) from %d unique article(s).",
        len(context_parts),
        len({c["article_id"] for c in citations if c["article_id"] is not None}),
    )

    return {
        "assembled_context": _CONTEXT_SEPARATOR.join(context_parts),
        "citations":         citations,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

# ── System prompt — FactLens agent persona ────────────────────────────────────
#
# Design notes:
#   • Establishes the agent as an expert, objective fact-checker.
#   • Makes the JSON-only contract unambiguous — LLMs sometimes add preamble;
#     repeating the constraint in both system and user prompts reduces that.
#   • Describes each verdict value so the model applies them consistently.
#   • Instructs the model to base its verdict exclusively on the retrieved
#     context — preventing hallucination from parametric memory.

_SYSTEM_PROMPT: str = """\
You are FactLens, an expert Political Intelligence & Fact-Verification Agent.
Your role is to evaluate political claims and news with rigorous objectivity,
cite only what is present in the provided source passages, and never
speculate beyond the evidence.

RESPONSE CONTRACT — CRITICAL
─────────────────────────────
You MUST respond with ONLY a single, valid JSON object.
No preamble. No explanation outside the JSON. No markdown code fences.
Raw JSON and nothing else.

Required JSON schema:
{
    "verdict":     "TRUE" | "FALSE" | "UNVERIFIABLE",
    "confidence":  <float between 0.0 and 1.0>,
    "reasoning":   "<2–3 sentences. Logical chain citing specific details from the context passages.>",
    "explanation": "<1–2 paragraphs. Plain-language summary for a non-expert reader.>"
}

Verdict definitions:
  TRUE          → The retrieved context directly supports the claim.
  FALSE         → The retrieved context directly contradicts the claim.
  UNVERIFIABLE  → The context is insufficient, absent, or ambiguous — you
                  cannot confirm or deny with the available evidence.

Confidence guidelines:
  1.0 — Multiple sources explicitly confirm / deny.
  0.8 — One clear source confirms / denies.
  0.6 — Partial or indirect evidence.
  0.4 — Tangential evidence only.
  0.2 — Minimal relevance, verdict is mostly UNVERIFIABLE.

Rules:
  • Base your verdict EXCLUSIVELY on the retrieved context passages provided.
  • Never use your parametric knowledge to fill gaps in the context.
  • If the context is empty, verdict MUST be UNVERIFIABLE with confidence ≤ 0.2.
  • Keep reasoning factual and concise.  Avoid political bias.\
"""


def build_prompt(
    query:             str,
    assembled_context: str,
) -> str:
    """
    Augmentation stage — construct the user turn of the prompt.

    Injects the retrieved context passages and the user's query into a
    structured template.  The template reminds the model of the JSON contract
    once more (redundant reminders improve LLM compliance significantly).

    Parameters
    ──────────
    query             : Raw query/claim string from the user.
    assembled_context : Parent paragraph texts joined by ``---`` separators,
                        produced by the retrieval stage.

    Returns
    ───────
    str — The full user-turn prompt string, ready for messages[].
    """
    if assembled_context.strip():
        context_block = assembled_context
    else:
        context_block = "[No relevant source passages were found in the database.]"

    prompt = f"""\
RETRIEVED SOURCE PASSAGES
══════════════════════════
{context_block}
══════════════════════════

CLAIM / QUERY TO VERIFY
───────────────────────
{query}

Respond with ONLY the JSON object described in your instructions.\
"""
    log.debug(
        "Augmentation: prompt built — context_chars=%d, query_chars=%d.",
        len(assembled_context),
        len(query),
    )
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate(
    sarvam_client: SarvamAI,
    user_prompt:   str,
) -> dict[str, Any]:
    """
    Generation stage — call Sarvam LLM and parse the JSON response.

    Uses ``sarvam-30b`` (or $SARVAM_MODEL override):
      • 64K context window — more than sufficient for ≤5 parent paragraphs.
      • Recommended for production workloads (balanced quality/cost).
      • Strong reasoning + Indic language support.

    Temperature is set to 0.1 for a fact-verification task — we want
    deterministic, evidence-grounded outputs, not creative variation.

    Parameters
    ──────────
    sarvam_client : Initialised SarvamAI client.
    user_prompt   : Augmented prompt string from build_prompt().

    Returns
    ───────
    dict with keys: verdict, confidence, reasoning, explanation.
    On parse failure, returns a fallback dict with verdict=UNVERIFIABLE
    and the raw LLM text in ``explanation`` so no information is lost.
    """
    log.info("Generation: calling Sarvam model '%s'…", SARVAM_MODEL)

    response = sarvam_client.chat.completions(
        model=SARVAM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,    # low temperature = focused, deterministic fact-checks
        max_tokens=1024,    # ample for the 4-field JSON; avoids runaway generation
    )

    raw: str = response.choices[0].message.content.strip()
    log.debug("Generation: raw response length=%d chars.", len(raw))

    return _parse_response(raw)


def _parse_response(raw: str) -> dict[str, Any]:
    """
    Extract and validate the JSON verdict from the LLM's raw output.

    Strategy:
      1. Try direct json.loads() — works when the model is well-behaved.
      2. Regex-extract the first {...} block — handles the case where the
         model adds a sentence before or after the JSON despite instructions.
      3. Graceful fallback — returns UNVERIFIABLE with the raw text preserved
         so the user still sees something meaningful.

    Parameters
    ──────────
    raw : Raw string content from response.choices[0].message.content.

    Returns
    ───────
    dict[str, Any] — always contains: verdict, confidence, reasoning, explanation.
    """
    # ── Attempt 1: Direct parse ───────────────────────────────────────────────
    try:
        parsed = json.loads(raw)
        return _normalise(parsed)
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: Regex extraction of first {...} block ─────────────────────
    # re.DOTALL so that newlines inside the JSON are matched by .
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            log.warning(
                "Generation: JSON was embedded in surrounding text — extracted via regex."
            )
            return _normalise(parsed)
        except json.JSONDecodeError:
            pass

    # ── Attempt 3: Graceful fallback ──────────────────────────────────────────
    log.error(
        "Generation: could not parse JSON from LLM response.  Raw:\n%s", raw
    )
    return {
        "verdict":     "UNVERIFIABLE",
        "confidence":  0.0,
        "reasoning":   "The model did not return a parseable JSON response.",
        "explanation": raw,   # preserve the full raw output for the user
    }


def _normalise(parsed: dict) -> dict[str, Any]:
    """
    Coerce and validate the parsed JSON into the expected schema.

    • Forces verdict to uppercase and validates it is one of the three values.
    • Clamps confidence to [0.0, 1.0].
    • Fills in any missing keys with safe defaults so callers never KeyError.
    """
    valid_verdicts = {"TRUE", "FALSE", "UNVERIFIABLE"}

    verdict = str(parsed.get("verdict", "UNVERIFIABLE")).upper().strip()
    if verdict not in valid_verdicts:
        log.warning("Unexpected verdict value '%s' — defaulting to UNVERIFIABLE.", verdict)
        verdict = "UNVERIFIABLE"

    try:
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "verdict":     verdict,
        "confidence":  confidence,
        "reasoning":   str(parsed.get("reasoning",   "No reasoning provided.")),
        "explanation": str(parsed.get("explanation", "No explanation provided.")),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Terminal display
# ═══════════════════════════════════════════════════════════════════════════

_WIDTH: int = 72   # terminal output width


def _hr(char: str = "═") -> str:
    return char * _WIDTH


def _wrap(text: str, indent: int = 0) -> str:
    """Word-wrap *text* to _WIDTH, optionally indenting continuation lines."""
    return textwrap.fill(
        text,
        width=_WIDTH,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )


def display_result(
    query:   str,
    result:  dict[str, Any],
    sources: list[dict],
) -> None:
    """
    Print the full FactLens verdict to the terminal in a readable layout.

    Parameters
    ──────────
    query   : The original user query string.
    result  : Parsed generation dict (verdict, confidence, reasoning, explanation).
    sources : Citations list from the retrieval stage.
    """
    verdict    = result["verdict"]
    confidence = result["confidence"]
    reasoning  = result["reasoning"]
    explanation= result["explanation"]

    verdict_display = _VERDICT_ICONS.get(verdict, f"?  {verdict}")
    confidence_pct  = f"{confidence * 100:.0f}%"

    # ── Confidence bar (10 blocks) ────────────────────────────────────────────
    filled = round(confidence * 10)
    bar    = "█" * filled + "░" * (10 - filled)

    print()
    print(_hr())
    print("  FactLens — Fact Verification Result")
    print(_hr())
    print()
    print(f"  QUERY      : {query}")
    print()
    print(f"  VERDICT    : {verdict_display}")
    print(f"  CONFIDENCE : [{bar}] {confidence_pct}")
    print()

    print(_hr("─"))
    print("  REASONING")
    print(_hr("─"))
    print(_wrap(reasoning, indent=2))
    print()

    print(_hr("─"))
    print("  EXPLANATION")
    print(_hr("─"))
    # explanation may have multiple paragraphs — wrap each separately
    for para in explanation.strip().split("\n\n"):
        print(_wrap(para.strip(), indent=2))
        print()

    if sources:
        print(_hr("─"))
        print("  SOURCES")
        print(_hr("─"))
        seen_urls: set[str] = set()
        citation_num = 1
        for src in sources:
            url = src.get("source_url", "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            aid = src.get("article_id", "?")
            print(f"  [{citation_num}] article_id={aid}")
            print(f"      {url}")
            citation_num += 1
        print()

    print(_hr())
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Service initialisation
# ═══════════════════════════════════════════════════════════════════════════

def _init_sarvam() -> SarvamAI:
    """
    Initialise the Sarvam AI client.

    Reads SarvamAI from the environment.  The SDK param is
    ``api_subscription_key`` (not ``api_key``) — this is a Sarvam-specific
    naming convention and differs from OpenAI's SDK.
    """
    api_key = os.getenv("SarvamAI", "").strip()
    if not api_key:
        log.critical(
            "SarvamAI environment variable is not set. "
            "Export it before running: export SarvamAI=sk_..."
        )
        sys.exit(1)

    client = SarvamAI(api_subscription_key=api_key)
    log.info("Sarvam AI client initialised (model=%s).", SARVAM_MODEL)
    return client


def _init_qdrant() -> QdrantClient:
    """Connect to Qdrant and verify the connection with a lightweight check."""
    log.info("Connecting to Qdrant at %s:%d…", QDRANT_HOST, QDRANT_PORT)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.get_collections()   # liveness check — raises if unreachable
    log.info("Qdrant: connection established.")
    return client


def _init_embed_model() -> BGEM3FlagModel:
    """Load BGE-M3. First run downloads ~2 GB of weights to HuggingFace cache."""
    log.info(
        "Loading BGE-M3 model '%s' (use_fp16=True)…  "
        "First run may download ~2 GB of weights.",
        BGE_MODEL_NAME,
    )
    model = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=True)
    log.info("BGE-M3: model loaded.")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Main interactive loop
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Interactive terminal loop.

    Initialises all services once, then enters a read-eval-print loop:
      • Prompts the user for a natural-language claim or question.
      • Runs the full Retrieval → Augmentation → Generation pipeline.
      • Prints the structured verdict to the terminal.
      • Repeats until the user types 'quit' or 'exit', or sends EOF (Ctrl-D).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print(_hr())
    print("  FactLens — Political Intelligence & Fact-Verification Agent")
    print("  Type a claim or question to verify.  Type 'quit' to exit.")
    print(_hr())
    print()

    # ── Initialise all services (expensive; done once at startup) ─────────────
    log.info("Initialising services…")

    # PostgreSQL
    try:
        conn         = Connection()      # reads DATABASE_URL from environment
        article_repo = ArticleRepository(conn)
        chunk_repo   = ParentChunkRepository(conn)
    except RuntimeError as exc:
        log.critical("Cannot initialise database connection: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.critical("Unexpected DB error: %s", exc, exc_info=True)
        sys.exit(1)

    # Qdrant
    try:
        qdrant = _init_qdrant()
    except Exception as exc:
        log.critical("Cannot connect to Qdrant: %s", exc)
        conn.close()
        sys.exit(1)

    # BGE-M3 embedding model
    try:
        embed_model = _init_embed_model()
    except Exception as exc:
        log.critical("Cannot load BGE-M3 model: %s", exc)
        conn.close()
        sys.exit(1)

    # Sarvam AI client
    sarvam_client = _init_sarvam()   # calls sys.exit(1) internally if key missing

    log.info("All services ready.  Entering query loop.")
    print()

    # ── Interactive loop ──────────────────────────────────────────────────────
    try:
        while True:
            # ── Read user input ───────────────────────────────────────────────
            try:
                query = input("  FactLens> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Exiting FactLens.  Goodbye.")
                break

            if not query:
                continue

            if query.lower() in {"quit", "exit", "q"}:
                print("\n  Exiting FactLens.  Goodbye.")
                break

            # ── Stage 1: Retrieval ────────────────────────────────────────────
            print()
            log.info("─" * 50)
            log.info("New query: '%s'", query[:120])

            try:
                retrieval_output = retrieve(
                    query        = query,
                    embed_model  = embed_model,
                    qdrant       = qdrant,
                    article_repo = article_repo,
                    chunk_repo   = chunk_repo,
                )
            except Exception as exc:
                log.error("Retrieval failed: %s", exc, exc_info=True)
                print("  [ERROR] Retrieval stage failed. Check logs.\n")
                continue

            assembled_context = retrieval_output["assembled_context"]
            citations         = retrieval_output["citations"]

            if not assembled_context.strip():
                log.warning("No context retrieved — proceeding with empty context.")

            # ── Stage 2: Augmentation ─────────────────────────────────────────
            user_prompt = build_prompt(query, assembled_context)

            # ── Stage 3: Generation ───────────────────────────────────────────
            try:
                generation_output = generate(sarvam_client, user_prompt)
            except Exception as exc:
                log.error("Generation failed: %s", exc, exc_info=True)
                print("  [ERROR] Generation stage failed. Check logs.\n")
                continue

            # ── Display ───────────────────────────────────────────────────────
            display_result(
                query   = query,
                result  = generation_output,
                sources = citations,
            )

    finally:
        conn.close()
        log.info("Database connection closed.")


if __name__ == "__main__":
    main()
