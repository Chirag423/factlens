import feedparser
import trafilatura
def ingest_news():
    # --- PHASE 1: FETCH THE FEED ---

    url = 'https://www.thehindu.com/business/Economy/feeder/default.rss'
    rss_metadata = feedparser.parse(url)
    # Check if feedparser failed or found nothin
    if rss_metadata.bozo:
        print(f"[!] Critical Error: Unable to fetch or parse RSS feed from: {url}", rss_metadata.bozo_exception)
        return # Stop the function completely because we have no data to process
    
    # --- PHASE 2: PROCESS THE ARTICLES ---

    for i,entry in enumerate(rss_metadata.entries[:5]):  # Limit to first 5 articles for testing
        print(f"Article {i+1}: {entry.title}")
        # Isolate individual article actions so one failure doesn't ruin the whole batch
        try:
            # Download and parse the full article content
            article_content = trafilatura.fetch_url(entry.link)
            if not article_content:
                print(f"Warning: Unable to fetch content for article {i+1} at {entry.link}")
                continue   # Skip this article and jump straight to the next loop iteration
            extracted_data = trafilatura.bare_extraction(article_content).as_dict()
            if extracted_data:
                author = extracted_data.get('author') or 'N/A'
                text = (extracted_data.get('text') or 'N/A')#[:200]  # Print first 200 characters for testing
                print(f"Author: {author}, \nContent: {text}...\n")
            else:
                print(f"Warning: No extractable content found for article {i+1} at {entry.link}")
        except Exception as e:
            print(f"Error processing article {i+1}: {e}")

if __name__ == "__main__":
    ingest_news()