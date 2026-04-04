"""
Stage C: Content Parsing

Handles additional scraping/parsing when Tavily's built-in content extraction
isn't sufficient. Uses trafilatura for clean text extraction from HTML.

Design decision: Tavily advanced search already extracts content, so we only
scrape supplementally if content is too short (< 200 chars). This saves time
and respects rate limits.
"""
import time
import requests
import trafilatura
from .config import Config, SearchResult, truncate_text, logger


def enrich_content(results: list[SearchResult], config: Config) -> list[SearchResult]:
    """
    Enrich search results with additional scraped content where needed.
    
    Strategy:
    - If Tavily already returned good content (>200 chars), keep it
    - Otherwise, fetch the page and extract with trafilatura
    - Truncate all content to max_content_chars to manage LLM context
    """
    enriched = []
    scrape_count = 0

    for result in results:
        if len(result.content) >= 200:
            # Tavily content is sufficient
            result.content = truncate_text(result.content, config.max_content_chars)
            enriched.append(result)
            continue

        # Need to scrape
        try:
            if scrape_count > 0:
                time.sleep(config.scrape_delay)  # politeness

            logger.info(f"Scraping: {result.url}")
            resp = requests.get(
                result.url,
                timeout=config.request_timeout,
                headers={"User-Agent": "AgenticSearchBot/1.0 (research project)"},
            )
            resp.raise_for_status()

            extracted = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )

            if extracted:
                result.content = truncate_text(extracted, config.max_content_chars)
                scrape_count += 1
            else:
                # fallback to snippet
                logger.warning(f"Trafilatura extraction failed for {result.url}")

        except Exception as e:
            logger.warning(f"Scrape failed for {result.url}: {e}")

        enriched.append(result)

    logger.info(
        f"Content enrichment: {len(enriched)} results, "
        f"{scrape_count} additionally scraped"
    )
    return enriched