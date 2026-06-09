import feedparser
import trafilatura
import asyncio
import aiohttp
import time

# ─── LAYER 1: RSS FEED FETCHING ───────────────────────────────────────────────
# Each function handles ONE unit of work so asyncio.gather() can run
# many of them simultaneously instead of waiting for each to finish.

async def fetch_single_rss(session, url):

    """Fetch and parse one RSS feed. Returns feedparser metadata or None on failure."""

    print(f"Fetching RSS feed from: {url}")
    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"[!] Critical Error: Failed to fetch RSS feed from: {url} with status code {response.status}")
                return None                # Return None to indicate failure for this URL
            
            # feedparser accepts raw XML text directly, not just URLs
            rss_metadata = feedparser.parse(await response.text())
            
            if rss_metadata.bozo:           # Check if feedparser failed or found nothing
                print(f"[!] Critical Error: Unable to parse RSS feed from: {url}", rss_metadata.bozo_exception)
                return None     

            return rss_metadata            
    except Exception as e:
        print(f"[!] Critical Error: Exception occurred while fetching RSS feed from: {url} - {e}")
        return None 
    
# ─── LAYER 2: ARTICLE CONTENT EXTRACTION ─────────────────────────────────────

async def fetch_single_url_trafilatura(session, article_url):
    print(f"Fetching HTML Page from: {article_url}")
    try:
        async with session.get(article_url) as response:
            if response.status != 200:
                print(f"[!] Critical Error: Failed to fetch HTML Page from: {article_url} with status code {response.status}")
                return None 
            
            # Walrus operator (:=) assigns + checks in one step,
            # preventing .as_dict() being called on a None return value
            if (raw_doc := trafilatura.bare_extraction(await response.text())):
                extracted_data = raw_doc.as_dict()
                return extracted_data  # Or run your extra dictionary validations here
            else:
                print(f"Warning: Trafilatura couldn't extract anything from {article_url}")
                return None
    except Exception as e:
        print(f"Error processing article : {e}")
        return None

async def fetch_single_rss_trafilatura(session, article_urls: list):

    """Fetch all articles from one feed simultaneously."""
    # One gather() per feed — every article in this feed fetched at the same time

    trafilatura_tasks = [fetch_single_url_trafilatura(session, url) for url in article_urls]
    single_rss_content = await asyncio.gather(*trafilatura_tasks)
    return single_rss_content

def enlist_single_rss_urls(rss_metadata) -> list:
    """Pull all article URLs out of a parsed RSS feed."""
    return [entry.link for entry in rss_metadata.entries]

# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────
# One shared aiohttp session handles all concurrent requests across both phases.

async def fetch_all_feed(feed_urls: list):
    """
    Runs the full async ingestion pipeline in two phases:
      Phase 1 — Fetch all RSS feeds simultaneously → collect article URL lists
      Phase 2 — Fetch all article content simultaneously across all feeds
    Returns: [ [feed1_articles], [feed2_articles], ... ]
    """
    async with aiohttp.ClientSession() as session:


        # Phase 1: all RSS feeds fetched at the same time
        feedparser_tasks = [fetch_single_rss(session, url) for url in feed_urls]
        all_rss_metadata = await asyncio.gather(*feedparser_tasks)
        
        # Phase 2: queue article fetches for every successful feed
        article_tasks = []
        for rss_metadata in all_rss_metadata:
            if rss_metadata is not None:
                article_urls = enlist_single_rss_urls(rss_metadata)
                article_tasks.append(fetch_single_rss_trafilatura(session, article_urls))


        # All article fetches across all feeds run simultaneously
        return await asyncio.gather(*article_tasks)
    
# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
        

feed_urls = ['https://www.thehindu.com/business/feeder/default.rss', #Business
           'https://www.thehindu.com/sci-tech/science/feeder/default.rss', #Science
           'https://www.thehindu.com/business/Economy/feeder/default.rss'] #Economy        

def ingest_news():
    start_time = time.time()
    
    # asyncio.run() bridges sync → async: starts the event loop,
    # runs the full pipeline, returns results when everything is done

    # 1. Gather all the nested data: [ [feed1_articles], [feed2_articles], ... ]
    rss_results = asyncio.run(fetch_all_feed(feed_urls))
    
    total_articles = sum(
        sum(1 for a in feed if a is not None)  # skip failed article fetches
        for feed in rss_results if feed
    )

    print("=" * 70)
    print("              FACTLENS FULL CONTENT INGESTION REPORT               ")
    print("=" * 70)
    print(f"Feeds Processed:  {len(rss_results)} of {len(feed_urls)}")
    print(f"Total Articles:   {total_articles}")
    print(f"Execution Time:   {time.time() - start_time:.2f} seconds")
    print("=" * 70)
    
    # Optional category labels matching your feed_urls order for cleaner logs
    categories = ["Business Feed", "Science Feed", "Economy Feed"]
    
    # 2. Loop through each individual feed list
    for feed_idx, feed_list in enumerate(rss_results):
        if not feed_list:
            continue
            
        feed_label = categories[feed_idx] if feed_idx < len(categories) else f"Feed {feed_idx + 1}"
        print(f"\n>>> CATEGORY: {feed_label.upper()} ({len(feed_list)} Articles Extracted) <<<\n")
        
        # 3. Loop through every article inside this category
        for art_idx, article in enumerate(feed_list):
            if article is None:  # skip articles that failed to fetch
                continue

            print(f"[{art_idx + 1}] TITLE:  {article.get('title', 'N/A')}")
            print(f"    AUTHOR: {article.get('author') or 'N/A'}")
            print(f"    CONTENT:\n{article.get('text') or 'N/A'}")
            print("-" * 70) # Visual line break between articles
            
    print("\n" + "=" * 70)
    print("                     END OF FULL CONTENT INGESTION                 ")
    print("=" * 70)

if __name__ == "__main__":
    ingest_news()