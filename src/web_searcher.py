"""
Stage B: Web Search

Uses Tavily's search API to find relevant web pages for the query.
Tavily returns both snippets AND extracted content, reducing our scraping needs.
"""
from tavily import TavilyClient
from .config import Config, SearchResult, logger


def search_web(search_terms: str, config: Config) -> list[SearchResult]:
    """
    Search the web using Tavily and return structured results.
    
    Tavily's 'advanced' search depth returns richer content extraction,
    which helps our LLM extract entities without additional scraping.
    """
    client = TavilyClient(api_key=config.tavily_api_key)

    try:
        response = client.search(
            query=search_terms,
            search_depth="advanced",       # deeper content extraction (2 credits)
            max_results=config.max_search_results,
            include_raw_content=False,      # raw HTML not needed; 'content' field suffices
        )
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []

    results = []
    for item in response.get("results", []):
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", "")[:500],   # Tavily 'content' is the snippet
            content=item.get("content", ""),           # full extracted text
            score=item.get("score", 0.0),
        ))

    logger.info(f"Search returned {len(results)} results for: {search_terms}")
    return results