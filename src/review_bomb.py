"""
Review Bomb: Popularity & Consensus Review Re-ranking

After entity extraction, this module:
1. For each entity, searches for reviews / popularity signals
2. Collects quantitative metrics (review count, avg rating, search result count)
3. Builds a "popularity profile" description per entity
4. Uses LLM to produce a final re-ranked ordering with a consensus score

Design decisions:
- We batch entities into groups to avoid excessive API calls
  (e.g., 15 entities = 15 Tavily searches + 1 LLM call for ranking)
- We search for "{entity_name} reviews ratings" to find review data
- Popularity signals collected: star ratings, review counts, mentions,
  number of search results (proxy for web presence)
- The LLM re-ranker sees ALL popularity profiles and produces a single
  ordered ranking with scores from 0-100
- This is an OPTIONAL stage — toggled by the caller
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tavily import TavilyClient
from anthropic import Anthropic
from .config import Config, Entity, safe_json_parse, truncate_text, logger


# ── Step 1: Gather popularity signals per entity ─────────────────────────────

def _search_entity_reviews(
    entity_name: str,
    entity_type: str,
    tavily_client: TavilyClient,
) -> dict:
    """
    Search for reviews/popularity info for a single entity.
    Returns a dict with raw signals.
    """
    query = f"{entity_name} reviews ratings"

    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",      # 1 credit each, keep costs low
            max_results=5,
        )
    except Exception as e:
        logger.warning(f"Review search failed for '{entity_name}': {e}")
        return {
            "entity_name": entity_name,
            "search_result_count": 0,
            "snippets": [],
            "error": str(e),
        }

    results = response.get("results", [])
    snippets = [r.get("content", "")[:300] for r in results]

    return {
        "entity_name": entity_name,
        "search_result_count": len(results),
        "snippets": snippets,
        "urls": [r.get("url", "") for r in results],
    }


def gather_popularity_signals(
    entities: list[Entity],
    entity_type: str,
    config: Config,
    max_concurrent: int = 3,
) -> list[dict]:
    """
    Search for reviews/popularity for each entity in parallel.
    Returns a list of signal dicts, one per entity.
    """
    tavily_client = TavilyClient(api_key=config.tavily_api_key)
    signals = [None] * len(entities)

    def _search(idx: int, entity: Entity):
        name = entity.attributes.get("name", "")
        if not name:
            return idx, {"entity_name": "", "search_result_count": 0, "snippets": []}
        return idx, _search_entity_reviews(name, entity_type, tavily_client)

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = [pool.submit(_search, i, e) for i, e in enumerate(entities)]
        for future in as_completed(futures):
            idx, result = future.result()
            signals[idx] = result

    logger.info(f"Gathered popularity signals for {len(entities)} entities")
    return signals


# ── Step 2: LLM re-ranking based on popularity profiles ─────────────────────

RERANK_PROMPT = """You are a popularity and quality ranking system.

Given a list of entities with their review/popularity data gathered from the web, produce a re-ranked list ordered by overall quality, popularity, and consensus sentiment.

For each entity, consider:
- Number of search results (web presence / popularity)
- Star ratings mentioned in snippets
- Review counts mentioned
- Sentiment of review snippets (positive vs negative)
- General reputation signals

Return ONLY a JSON array, ordered from best to worst. Each element:
[
  {{
    "name": "Entity Name",
    "popularity_score": 85,
    "review_summary": "Brief 1-2 sentence consensus summary",
    "key_signals": "e.g. 4.5 stars on Google, 200+ reviews, widely recommended"
  }}
]

The popularity_score should be 0-100 where:
- 90-100: Exceptional, universally praised
- 70-89: Very good, mostly positive
- 50-69: Average, mixed reviews
- 30-49: Below average, notable issues
- 0-29: Poor or very little information available"""


def rerank_by_popularity(
    entities: list[Entity],
    signals: list[dict],
    config: Config,
) -> list[Entity]:
    """
    Use LLM to re-rank entities based on gathered popularity signals.
    Adds popularity_score, review_summary, key_signals to each entity's attributes.
    Returns entities in new order.
    """
    # Build the input for the LLM
    profiles = []
    for i, (entity, signal) in enumerate(zip(entities, signals)):
        name = entity.attributes.get("name", f"Entity {i}")
        snippet_text = "\n".join(signal.get("snippets", []))
        profiles.append(
            f"--- {name} ---\n"
            f"Web results found: {signal.get('search_result_count', 0)}\n"
            f"Review snippets:\n{truncate_text(snippet_text, 1000)}\n"
        )

    profiles_text = "\n".join(profiles)

    client = Anthropic(api_key=config.anthropic_api_key)

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=3000,
            system=RERANK_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Re-rank these entities by popularity and quality:\n\n{profiles_text}",
                }
            ],
        )
    except Exception as e:
        logger.error(f"Re-ranking LLM call failed: {e}")
        return entities  # fail open

    result_text = response.content[0].text
    parsed = safe_json_parse(result_text)

    if not parsed or not isinstance(parsed, list):
        logger.warning("Failed to parse re-ranking response, returning original order")
        return entities

    # Build a name → ranking map
    ranking_map = {}
    for rank_item in parsed:
        name = rank_item.get("name", "")
        ranking_map[name.lower().strip()] = rank_item

    # Re-order entities and enrich with popularity data
    scored_entities = []
    for entity in entities:
        name = entity.attributes.get("name", "")
        rank_data = ranking_map.get(name.lower().strip(), {})

        # Add popularity fields to attributes
        entity.attributes["popularity_score"] = rank_data.get("popularity_score", 0)
        entity.attributes["review_summary"] = rank_data.get("review_summary", "No data")
        entity.attributes["key_signals"] = rank_data.get("key_signals", "")

        scored_entities.append(entity)

    # Sort by popularity_score descending
    scored_entities.sort(
        key=lambda e: e.attributes.get("popularity_score", 0),
        reverse=True,
    )

    logger.info(
        f"Re-ranked {len(scored_entities)} entities. "
        f"Top: {scored_entities[0].attributes.get('name', '?')} "
        f"({scored_entities[0].attributes.get('popularity_score', 0)})"
        if scored_entities else "No entities to rank"
    )

    return scored_entities


# ── Public API ───────────────────────────────────────────────────────────────

def review_bomb(
    entities: list[Entity],
    entity_type: str,
    config: Config,
) -> tuple[list[Entity], list[str]]:
    """
    Full review bomb pipeline: gather signals → re-rank.
    
    Returns:
        (reranked_entities, new_columns) where new_columns are the
        popularity columns added to the schema.
    """
    if not entities:
        return entities, []

    logger.info(f"Starting review bomb for {len(entities)} entities")

    # Step 1: Gather signals
    signals = gather_popularity_signals(entities, entity_type, config)

    # Step 2: Re-rank
    reranked = rerank_by_popularity(entities, signals, config)

    new_columns = ["popularity_score", "review_summary", "key_signals"]

    logger.info("Review bomb complete")
    return reranked, new_columns