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
    """OUT: FeedParserDict  →  .feed (channel info)  +  .entries (list of article objects)"""

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

def enlist_single_rss_articles_metadata(single_rss_metadata) -> list:
    """Pull metadata (Title, URL, GUID, Published Date, Category) from each feed entry."""
    """IN:  FeedParserDict  (from fetch_single_rss)
       OUT: list[dict]  →  [{ title, url, guid, published, category }, ...]  — no body text yet"""
    
    return [{
        'title': entry.get('title', 'N/A'),
        'url': entry.link,
        'guid': entry.get('guid', 'N/A'),
        'published': entry.get('published', 'N/A'),
        'category': entry.get('category', 'N/A'),
         
    } for entry in single_rss_metadata.entries]


async def fetch_single_url_trafilatura(session, single_article_metadata :dict):
    """Fetch one article's HTML and append extracted body text to its metadata dict.
 
    IN:  dict  →  { title, url, guid, published, category }
    OUT: dict  →  { title, url, guid, published, category, text }  or None on failure"""

    article_url = single_article_metadata['url']

    try:
        async with session.get(article_url) as response:
            if response.status != 200:
                print(f"[!] Critical Error: Failed to fetch HTML Page from: {article_url} with status code {response.status}")
                return None 
            
            # Walrus operator (:=) assigns + checks in one step,
            # preventing .as_dict() being called on a None return value
            # Pass the URL to bare_extraction to improve its internal heuristics
            if (raw_doc := trafilatura.bare_extraction(await response.text(), url=article_url)) is not None:
                extracted_data = raw_doc.as_dict()

                # Merge the body text from Trafilatura
                single_article_metadata['text'] = extracted_data.get('text')

                return single_article_metadata  
            else:
                print(f"Warning: Trafilatura couldn't extract anything from {article_url}")
                return None
    except Exception as e:
        print(f"Error processing article : {e}")
        return None

async def fetch_single_rss_trafilatura(session, single_rss_metadata :list):

    """Fetch all articles from one feed simultaneously."""
    """    IN:  list[dict]       →  [{ title, url, guid, published, category }, ...]
    OUT: list[dict|None]  →  [{ title, url, guid, published, category, text }, ...]"""
    # One gather() per feed — every article in this feed fetched at the same time

    trafilatura_tasks = [fetch_single_url_trafilatura(session, single_article_metadata) for single_article_metadata in single_rss_metadata]
    single_rss_content = await asyncio.gather(*trafilatura_tasks)
    return single_rss_content

# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────
# One shared aiohttp session handles all concurrent requests across both phases.

async def fetch_all_feed(feed_urls: list):
    """Two-phase pipeline: fetch all feeds, then fetch all articles — both fully concurrent.
 
    Phase 1 → all_rss_metadata : list[FeedParserDict|None]   (one per feed URL)
    Phase 2 → articles per feed enriched with 'text'
 
    OUT: list[list[dict|None]]  →  [[feed1_articles], [feed2_articles], ...]
    """

    async with aiohttp.ClientSession() as session:


        # Phase 1: all RSS feeds fetched at the same time
        feedparser_tasks = [fetch_single_rss(session, url) for url in feed_urls]
        all_rss_metadata = await asyncio.gather(*feedparser_tasks)
        
        # Phase 2: queue article fetches for every successful feed
        article_tasks = []
        for rss_metadata in all_rss_metadata:
            if rss_metadata is not None:
                articles_metadata = enlist_single_rss_articles_metadata(rss_metadata)
                article_tasks.append(fetch_single_rss_trafilatura(session, articles_metadata))


        # All article fetches across all feeds run simultaneously
        return await asyncio.gather(*article_tasks)      # list[list[dict|None]]

    
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