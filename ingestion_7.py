import feedparser
import trafilatura
import asyncio
import aiohttp
import time
import logging
import re
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict
from dateutil import parser as dateutil_parser
from dateutil.tz import gettz

from db.connection import Connection
from db.models import ArticleRow
from db.repositories.articles import ArticleRepository

# IST is ambiguous in dateutil (India / Israel / Ireland).
# Pinning it to Asia/Kolkata ensures Indian news dates parse correctly.
_IST_TZ   = gettz("Asia/Kolkata")
_TZINFOS  = {"IST": _IST_TZ}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("factlens.ingestion")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Bundle label + URL together so they never drift apart (fixes Bug 6)
FEEDS = [
    ("Business", "https://www.thehindu.com/business/feeder/default.rss"),
    ("Science",  "https://www.thehindu.com/sci-tech/science/feeder/default.rss"),
    ("Economy",  "https://www.thehindu.com/business/Economy/feeder/default.rss"),
    
    # Times of India Feeds
    ("TOI Feed (-2128672765)", "https://timesofindia.indiatimes.com/rssfeeds/-2128672765.cms"),
    ("TOI Top Stories",        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("TOI Most Recent",        "https://timesofindia.indiatimes.com/rssfeedmostrecent.cms"),
    ("TOI Feed (-2128936835)", "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms"),
    ("TOI Feed (296589292)",   "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"),
    ("TOI Feed (7098551)",     "https://timesofindia.indiatimes.com/rssfeeds/7098551.cms"),
    ("TOI Feed (1898055)",     "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms"),
    ("TOI US Feed (72258322)", "https://timesofindia.indiatimes.com/rssfeeds_us/72258322.cms"),
    ("TOI Feed (54829575)",    "https://timesofindia.indiatimes.com/rssfeeds/54829575.cms"),
    ("TOI Feed (4719148)",     "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms"),
    ("TOI Feed (2647163)",     "https://timesofindia.indiatimes.com/rssfeeds/2647163.cms"),
    ("TOI Feed (66949542)",    "https://timesofindia.indiatimes.com/rssfeeds/66949542.cms"),
    ("TOI Feed (913168846)",   "https://timesofindia.indiatimes.com/rssfeeds/913168846.cms"),
    ("TOI Feed (1081479906)",  "https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms"),
    ("TOI Feed (2886704)",     "https://timesofindia.indiatimes.com/rssfeeds/2886704.cms"),
    ("TOI Most Read",          "https://timesofindia.indiatimes.com/rssfeedmostread.cms"),
    ("TOI Most Shared",        "https://timesofindia.indiatimes.com/rssfeedmostshared.cms"),
    ("TOI Most Commented",     "https://timesofindia.indiatimes.com/rssfeedmostcommented.cms"),
    
    # The Hindu Feeds (Duplicates removed)
    ("Sport",                  "https://www.thehindu.com/sport/feeder/default.rss"),
    ("News",                   "https://www.thehindu.com/news/feeder/default.rss"),
    ("Entertainment",          "https://www.thehindu.com/entertainment/feeder/default.rss"),
    ("Life & Style",           "https://www.thehindu.com/life-and-style/feeder/default.rss"),
]
# A realistic browser User-Agent prevents CDN blocks (fixes Bug 4)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT   = aiohttp.ClientTimeout(total=15)   # per-request wall-clock limit (fixes Bug 2)
MAX_CONCURRENCY   = 5    # max simultaneous article fetches (fixes Bug 3)
MAX_RETRIES       = 2    # how many times to retry a timed-out URL
RETRY_DELAY       = 2.0  # seconds to wait before each retry attempt


# ─── DATA TYPES ───────────────────────────────────────────────────────────────

@dataclass
class FeedResult:
    """Holds the outcome of processing one RSS feed — preserves label alignment."""
    label:    str
    feed_url: str
    articles: list = field(default_factory=list)   # list of (url, rss_title) → then article dicts
    failed:   bool = False                          # True if the RSS fetch itself failed


# ─── LAYER 1: RSS FEED FETCHING ───────────────────────────────────────────────

# Matches "Published - June 05, 2026 04:42 am IST" anywhere in article body text.
# Used as Pass 3 fallback when the RSS entry carries no date metadata at all.
_BODY_DATE_RE = re.compile(
    r'(?:Published|Updated)\s*[-–:on]*\s*([A-Za-z]+ \d{1,2},?\s+\d{4}[^\n|]{0,30})',
    re.IGNORECASE,
)


def _parse_pub_date(entry, body_text: str = "") -> datetime | None:
    """
    Three-pass date extraction.

    Pass 1 — feedparser's published_parsed struct (UTC-normalised).
             Works for clean RFC-2822 / RFC-3339 feeds.

    Pass 2 — Raw entry.published / entry.updated string via dateutil fuzzy
             parse.  Handles non-standard Indian formats like
             "June 05, 2026 04:42 am IST" and +05:30 offsets that
             feedparser silently drops.

    Pass 3 — Scan the article body text for a "Published - <date>" line.
             The Hindu embeds the canonical publish date at the end of
             every article body.  RSS entries for these articles often
             carry no date metadata at all, so without this pass they
             would be treated as undated and slip through the filter.

    Returns a timezone-aware datetime, or None when all three passes fail.
    """
    # Pass 1: feedparser's pre-parsed struct
    pub_struct = entry.get("published_parsed")
    if pub_struct is not None:
        try:
            return datetime(*pub_struct[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass

    # Pass 2: raw RSS date string → dateutil fuzzy parse
    raw: str = entry.get("published", "") or entry.get("updated", "") or ""
    if raw:
        raw = re.sub(r"^Published\s*[-–]\s*", "", raw).strip()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = dateutil_parser.parse(raw, fuzzy=True, tzinfos=_TZINFOS)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_IST_TZ)
            return dt
        except (ValueError, OverflowError):
            pass

    # Pass 3: scan the article body for an embedded publish date
    if body_text:
        tail = body_text[-400:]   # date line is always near the end
        m = _BODY_DATE_RE.search(tail)
        if m:
            raw_body = m.group(1).strip()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dt = dateutil_parser.parse(raw_body, fuzzy=True, tzinfos=_TZINFOS)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_IST_TZ)
                return dt
            except (ValueError, OverflowError):
                pass

    return None


def _is_recent_or_undated(entry, body_text: str = "") -> bool:
    """
    Return True when the article was published today or yesterday (UTC),
    or when no publication date can be determined after all three passes.

    Pass the extracted article body as body_text so that articles whose
    RSS entries carry no date metadata can still be filtered via the
    "Published - <date>" line embedded in The Hindu's article bodies.
    """
    from datetime import timedelta
    pub_dt = _parse_pub_date(entry, body_text)
    if pub_dt is None:
        return True   # genuinely undated after all three passes → include
    pub_date  = pub_dt.astimezone(timezone.utc).date()
    today     = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    return pub_date >= yesterday


async def fetch_single_rss(session: aiohttp.ClientSession, label: str, url: str) -> FeedResult:
    """Fetch and parse one RSS feed.  Returns a FeedResult (never raises)."""
    log.info("Fetching RSS feed [%s]: %s", label, url)
    result = FeedResult(label=label, feed_url=url)

    try:
        async with session.get(url) as response:
            if response.status != 200:
                log.error("[%s] RSS fetch failed — HTTP %s", label, response.status)
                result.failed = True
                return result

            rss_metadata = feedparser.parse(await response.text())

            if rss_metadata.bozo:
                log.error("[%s] feedparser could not parse feed: %s", label, rss_metadata.bozo_exception)
                result.failed = True
                return result

            # Filter to today-or-undated entries and collect deduplicated URLs.
            # Each item stored as (url, rss_title) so the RSS <title> travels
            # alongside the URL and can be stamped onto the extracted article dict.
            seen: set[str] = set()
            for entry in rss_metadata.entries:
                if _is_recent_or_undated(entry):
                    url_candidate = entry.get("link", "").strip()
                    if url_candidate and url_candidate not in seen:
                        seen.add(url_candidate)
                        rss_title = entry.get("title", "").strip()
                        result.articles.append((url_candidate, rss_title))

            log.info("[%s] %d article URLs queued for extraction", label, len(result.articles))
            return result

    except asyncio.TimeoutError:
        log.warning("[%s] RSS fetch timed out: %s", label, url)
        result.failed = True
        return result
    except Exception as e:
        log.error("[%s] Unexpected error fetching RSS: %s", label, e)
        result.failed = True
        return result


# ─── LAYER 2: ARTICLE CONTENT EXTRACTION WITH RETRY QUEUE ────────────────────

async def fetch_article_with_retry(
    session: aiohttp.ClientSession,
    article_url: str,
    sem: asyncio.Semaphore,
) -> dict | None:
    """
    Fetch one article URL, with a retry queue for timeouts.

    Strategy (fixes Bug 2 + adds retry):
      - First attempt runs inside the shared semaphore.
      - On TimeoutError the URL is queued for a second attempt after a short
        delay, giving the server time to recover and letting other URLs finish
        first (queue behaviour the user requested).
      - Non-timeout failures (HTTP errors, trafilatura returning nothing) are
        not retried — they are genuine content problems, not transient ones.
    """
    for attempt in range(1, MAX_RETRIES + 2):   # attempts: 1, 2, 3 (first + 2 retries)
        if attempt > 1:
            log.info("  ↻ Retry %d/%d for: %s", attempt - 1, MAX_RETRIES, article_url)
            await asyncio.sleep(RETRY_DELAY)

        try:
            async with sem:
                async with session.get(article_url) as response:
                    if response.status != 200:
                        log.warning("  [!] HTTP %s — skipping: %s", response.status, article_url)
                        return None

                    # Decode manually to handle mis-reported charsets on
                    # Hindi / Urdu pages (fixes Bug 10)
                    raw_bytes = await response.read()
                    charset   = response.charset or "utf-8"
                    html      = raw_bytes.decode(charset, errors="replace")

                    # Walrus operator: assign + guard in one step
                    if raw_doc := trafilatura.bare_extraction(html, url=article_url):
                        extracted = raw_doc.as_dict()
                        # Guarantee the URL is always present regardless of
                        # whether trafilatura populated it from the url= param
                        extracted["url"] = article_url
                        return extracted
                    else:
                        log.warning("  trafilatura extracted nothing from: %s", article_url)
                        return None

        except asyncio.TimeoutError:
            if attempt <= MAX_RETRIES:
                log.warning("  Timeout on attempt %d — will retry: %s", attempt, article_url)
                continue   # loop back → sleep → retry
            else:
                log.error("  All %d attempts timed out — giving up: %s", MAX_RETRIES + 1, article_url)
                return None

        except Exception as e:
            log.error("  Unexpected error fetching article: %s — %s", article_url, e)
            return None

    return None   # unreachable, but satisfies type checkers


async def fetch_articles_for_feed(
    session: aiohttp.ClientSession,
    feed_result: FeedResult,
    sem: asyncio.Semaphore,
    global_seen: set[str],
) -> FeedResult:
    """
    Fetch all article bodies for one FeedResult, then post-filter by date.

    Post-filtering is the third and final date-filter pass: articles whose
    RSS entries carried no date metadata are checked against the
    "Published - <date>" line embedded in their body text before being
    kept.  This catches cases where the RSS feed omits publication dates
    entirely (common on The Hindu's economy/science feeds).
    """
    tasks = []
    urls_to_fetch = []   # list of (url, rss_title) — mirrors tasks index-for-index

    for url, rss_title in feed_result.articles:
        if url in global_seen:
            log.debug("  Skipping duplicate URL: %s", url)
            continue
        global_seen.add(url)
        urls_to_fetch.append((url, rss_title))
        tasks.append(fetch_article_with_retry(session, url, sem))

    if not tasks:
        feed_result.articles = []
        return feed_result

    extracted = await asyncio.gather(*tasks)

    # Post-filter: re-check each article using its body text as Pass 3.
    # We also stamp the RSS <title> onto each article dict here — it is the
    # authoritative title and overrides whatever trafilatura parsed from the HTML.
    kept = []
    empty_entry = {}   # no published_parsed, no published, no updated
    for (url, rss_title), article in zip(urls_to_fetch, extracted):
        if article is None:
            continue
        # RSS title is canonical; always prefer it over trafilatura's extraction
        if rss_title:
            article["title"] = rss_title
        body = article.get("text") or ""
        if _is_recent_or_undated(empty_entry, body):
            kept.append(article)
        else:
            log.info("  Post-filter blocked old article: %s", (article.get("title") or "N/A")[:60])

    feed_result.articles = kept
    return feed_result


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

async def fetch_all_feeds(feeds: list[tuple[str, str]]) -> list[FeedResult]:
    """
    Two-phase async pipeline:

    Phase 1 — Fetch all RSS feeds simultaneously.
              Each feed returns a FeedResult with today's article URLs.
    Phase 2 — Fetch all article bodies across all feeds simultaneously,
              throttled by semaphore, with per-URL retry queue on timeout.

    Returns a list of FeedResult (one per feed, in original order).
    Fixes Bug 5: failed feeds are preserved in the list as FeedResult(failed=True)
    so category labels never shift (fixes Bug 6 at the structural level too).
    """
    # Bug 4 fix: shared session with browser User-Agent and timeout
    async with aiohttp.ClientSession(headers=HEADERS, timeout=REQUEST_TIMEOUT) as session:

        # ── Phase 1 ──────────────────────────────────────────────────────────
        rss_tasks = [fetch_single_rss(session, label, url) for label, url in feeds]
        feed_results: list[FeedResult] = await asyncio.gather(*rss_tasks)

        # ── Phase 2 ──────────────────────────────────────────────────────────
        sem = asyncio.Semaphore(MAX_CONCURRENCY)   # fixes Bug 3
        global_seen: set[str] = set()              # cross-feed URL dedup (fixes Bug 7)

        extraction_tasks = [
            fetch_articles_for_feed(session, fr, sem, global_seen)
            for fr in feed_results
            if not fr.failed   # no point fetching articles for a feed that didn't load
        ]

        await asyncio.gather(*extraction_tasks)
        # feed_results is mutated in-place by fetch_articles_for_feed,
        # so failed feeds still sit at their original index → labels stay aligned

    return feed_results


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def ingest_news(feeds: list[tuple[str, str]] = FEEDS) -> list[FeedResult]:
    """
    Synchronous entry point.  Bridges sync → async via asyncio.run().

    Takes an optional feeds list so this function is testable and reusable
    without module-level globals (partial fix for Bug 8 — for Celery/FastAPI
    use, call fetch_all_feeds() directly from an async context instead).
    """
    start_time = time.time()

    # asyncio.run() starts the event loop, blocks until everything finishes,
    # then tears the loop down — safe to call from a sync Celery task worker.
    feed_results = asyncio.run(fetch_all_feeds(feeds))

    elapsed    = time.time() - start_time
    successful = [fr for fr in feed_results if not fr.failed]
    failed     = [fr for fr in feed_results if fr.failed]

    total_articles = sum(
        sum(1 for a in fr.articles if a is not None)
        for fr in successful
    )

    # ── Report ────────────────────────────────────────────────────────────────
    # Bug 9 fix: report shows both attempted and successful feed counts
    print("=" * 70)
    print("              FACTLENS FULL CONTENT INGESTION REPORT               ")
    print("=" * 70)
    print(f"Feeds Attempted:  {len(feeds)}")
    print(f"Feeds Successful: {len(successful)}  |  Failed: {len(failed)}")
    if failed:
        for fr in failed:
            print(f"  ✗  {fr.label} ({fr.feed_url})")
    print(f"Total Articles:   {total_articles}")
    print(f"Execution Time:   {elapsed:.2f}s")
    print(f"Date Filter:      {datetime.now(timezone.utc).date()} (UTC)")
    print("=" * 70)

    for feed_result in feed_results:
        if feed_result.failed:
            print(f"\n>>> [{feed_result.label.upper()}] FEED FAILED — NO ARTICLES <<<")
            continue

        non_null = [a for a in feed_result.articles if a is not None]
        print(f"\n>>> CATEGORY: {feed_result.label.upper()} ({len(non_null)} articles extracted) <<<\n")

        for idx, article in enumerate(feed_result.articles, start=1):
            if article is None:
                continue
            print(f"[{idx}] TITLE:  {article.get('title') or 'N/A'}")
            print(f"    URL:    {article.get('url') or 'N/A'}")
            print(f"    AUTHOR: {article.get('author') or 'N/A'}")
            print(f"    CONTENT:\n{article.get('text') or 'N/A'}")
            print("-" * 70)

    print("\n" + "=" * 70)
    print("                     END OF FULL CONTENT INGESTION                 ")
    print("=" * 70)

    print(total_articles)
    print(f"Execution Time:   {elapsed:.2f}s")

    # ── DB Insert ─────────────────────────────────────────────────────────────
    inserted_count  = 0
    skipped_count   = 0
    db_error_count  = 0

    conn = Connection()
    repo = ArticleRepository(conn)

    for feed_result in feed_results:
        if feed_result.failed:
            continue

        for article in feed_result.articles:
            if article is None:
                continue

            url = article.get("url") or ""
            if not url:
                log.warning("Skipping article with no URL")
                continue

            if repo.exists(url):
                log.debug("DB skip (already exists): %s", url)
                skipped_count += 1
                continue

            # Use the same three-pass date extractor used for filtering
            empty_entry = {}
            pub_dt = _parse_pub_date(empty_entry, article.get("text") or "")

            row = ArticleRow(
                url         = url,
                title       = article.get("title") or None,
                raw_content = article.get("text") or "",
                category    = feed_result.label,
                published_at= pub_dt,
                # fetched_at defaults to now() via the lambda in ArticleRow
            )

            try:
                article_id = repo.insert(row)
                if article_id is not None:
                    inserted_count += 1
                    log.debug("DB insert id=%d url=%s", article_id, url)
                else:
                    skipped_count += 1
            except Exception as exc:
                db_error_count += 1
                log.error("DB insert failed for %s — %s", url, exc)

    print("=" * 70)
    print("                        DB INSERT SUMMARY                          ")
    print("=" * 70)
    print(f"  Inserted: {inserted_count}")
    print(f"  Skipped (duplicates): {skipped_count}")
    if db_error_count:
        print(f"  Errors:   {db_error_count}")
    print("=" * 70)

    return feed_results


# ─── ASYNC ENTRY POINT (for Celery / FastAPI) ─────────────────────────────────
# Use this instead of ingest_news() when you are already inside an event loop.
#
# Example (FastAPI):
#   @app.get("/ingest")
#   async def trigger_ingest():
#       results = await fetch_all_feeds(FEEDS)
#       return {"articles": sum(len(fr.articles) for fr in results)}


if __name__ == "__main__":
    ingest_news()