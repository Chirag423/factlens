import asyncio
import aiohttp
import trafilatura
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field

from db.connection import Connection
from db.models import ArticleRow
from db.repositories.articles import ArticleRepository

# ─── LOGGING ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("factlens.ingestion")

# ─── CONFIG ──────────────────────────────────────────────────────────────

URLS_FILE = "urls.txt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
MAX_CONCURRENCY = 5
MAX_RETRIES = 2
RETRY_DELAY = 2


# ─── DATA TYPES ──────────────────────────────────────────────────────────

@dataclass
class ScrapeResult:
    category: str = "ethanol"
    articles: list = field(default_factory=list)


# ─── URL LOADER ──────────────────────────────────────────────────────────

def load_urls(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


# ─── ARTICLE FETCHING ────────────────────────────────────────────────────

async def fetch_article_with_retry(
    session: aiohttp.ClientSession,
    article_url: str,
    sem: asyncio.Semaphore,
):
    for attempt in range(MAX_RETRIES + 1):

        try:
            async with sem:
                async with session.get(article_url) as response:

                    if response.status != 200:
                        log.warning("HTTP %s : %s", response.status, article_url)
                        return None

                    raw_bytes = await response.read()

                    charset = response.charset or "utf-8"
                    html = raw_bytes.decode(charset, errors="replace")

                    extracted = trafilatura.bare_extraction(
                        html,
                        url=article_url
                    )

                    if not extracted:
                        return None

                    article = extracted.as_dict()

                    article["url"] = article_url

                    # FORCE CATEGORY
                    article["category"] = "ethanol"

                    return article

        except asyncio.TimeoutError:

            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
                continue

            log.error("Timeout: %s", article_url)
            return None

        except Exception as e:
            log.error("%s -> %s", article_url, e)
            return None


# ─── SCRAPER ─────────────────────────────────────────────────────────────

async def scrape_urls(urls: list[str]):

    async with aiohttp.ClientSession(
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    ) as session:

        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        tasks = [
            fetch_article_with_retry(session, url, sem)
            for url in urls
        ]

        results = await asyncio.gather(*tasks)

        return [r for r in results if r]


# ─── INGEST ──────────────────────────────────────────────────────────────

def ingest_news():

    start_time = time.time()

    urls = load_urls(URLS_FILE)

    articles = asyncio.run(scrape_urls(urls))

    print(f"URLs Loaded: {len(urls)}")
    print(f"Articles Extracted: {len(articles)}")

    conn = Connection()
    repo = ArticleRepository(conn)

    inserted = 0
    skipped = 0

    for article in articles:

        url = article.get("url")

        if not url:
            continue

        if repo.exists(url):
            skipped += 1
            continue

        row = ArticleRow(
            url=url,
            title=article.get("title"),     # extracted title
            raw_content=article.get("text", ""),
            category="ethanol",             # forced category
            published_at=None
        )

        try:
            article_id = repo.insert(row)

            if article_id:
                inserted += 1

        except Exception as exc:
            log.error("DB insert failed %s -> %s", url, exc)

    elapsed = time.time() - start_time

    print("=" * 60)
    print("INSERTED :", inserted)
    print("SKIPPED  :", skipped)
    print("TIME     :", f"{elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    ingest_news()