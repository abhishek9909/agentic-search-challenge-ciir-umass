"""
Stage A: Query Classification

Determines whether a user query is a valid "topic query" that seeks a list
of entities, and if so, decomposes it into:
  - search_terms: what to actually search the web for
  - post_filters: constraints to verify/filter after retrieval
  - entity_type: what kind of entity we're looking for
  - suggested_columns: what attributes to extract
"""
import json
from anthropic import Anthropic
from .config import Config, QueryAnalysis, safe_json_parse, logger


CLASSIFICATION_PROMPT = """You are a query analysis system. Your job is to determine if a user query is a "topic query" — one that seeks a LIST of entities (companies, restaurants, tools, people, places, products, etc.) that could be displayed in a structured table.

VALID topic queries examples:
- "AI startups in healthcare" → wants a list of companies
- "best pizza places in Brooklyn" → wants a list of restaurants  
- "open source database tools" → wants a list of software tools
- "best hiking trails in Colorado" → wants a list of trails

INVALID (not topic queries):
- "What is photosynthesis?" → factual/definitional question
- "How do I fix a leaky faucet?" → how-to question
- "Tell me a joke" → conversational request
- "Who is the president?" → single-answer question
- "Translate hello to Spanish" → task request

For VALID topic queries, you must also:
1. Identify the entity type (restaurant, company, tool, person, etc.)
2. Decompose the query AGGRESSIVELY into search_terms + post_filters:
   - search_terms: the MINIMAL core noun phrase that captures what kind of
     entity we're looking for. Keep this SHORT and BROAD.
   - post_filters: EVERY constraint that narrows the results. Be generous —
     when in doubt, make it a filter. Filters get verified later via per-entity
     search, so they don't need to be search-engine-friendly.
   
   Put these in POST_FILTERS (not search_terms):
   - Numeric/quantitative constraints: "> 10M funding", "under $50", "4+ stars", "500+ reviews"
   - Country or region constraints when they're country-scale or larger: "in US", "in Europe", "based in Asia"
   - Feature/attribute constraints: "open source", "serves halal", "has outdoor seating", "accepts Medicaid"
   - Time constraints: "founded after 2020", "raised in 2024"
   - Scale/size constraints: "Fortune 500", "small startups", "publicly traded"
   
   Keep in SEARCH_TERMS only:
   - The core entity type noun phrase: "search engine startups", "pizza restaurants", "database tools"
   - Very specific city/neighborhood names when they are the PRIMARY locator:
     "pizza in Brooklyn" → search: "pizza Brooklyn". But country-level ("in US")
     should still be a filter because US is too broad to narrow the search.
   
   Examples:
   
   Query: "search engine startups in US that get more than 10M funding"
   → search_terms: "search engine startups"
   → post_filters: ["based in US", "funding > $10M"]
   
   Query: "Asian restaurants in Chicago that serve halal food"
   → search_terms: "Asian restaurants Chicago"
   → post_filters: ["serves halal food"]
   
   Query: "open source database tools written in Rust"
   → search_terms: "open source database tools"
   → post_filters: ["written in Rust"]
   
   Query: "top pizza places in Brooklyn"
   → search_terms: "top pizza places Brooklyn"
   → post_filters: []
   
   Query: "AI startups in healthcare with Series A funding"
   → search_terms: "AI healthcare startups"
   → post_filters: ["raised Series A funding"]

3. Suggest 4-7 relevant columns for the output table.

Respond with ONLY this JSON (no other text):
{
  "is_topic_query": true/false,
  "entity_type": "restaurant|company|tool|person|place|product|event|other",
  "search_terms": "the actual web search query to use",
  "post_filters": ["constraint1", "constraint2"],
  "suggested_columns": ["name", "column2", "column3", ...],
  "reasoning": "Brief explanation of your analysis"
}"""


def classify_query(query: str, config: Config) -> QueryAnalysis:
    """
    Classify a query and decompose it into search terms + filters.
    
    Returns a QueryAnalysis with all fields populated.
    """
    client = Anthropic(api_key=config.anthropic_api_key)

    response = client.messages.create(
        model=config.llm_model,
        max_tokens=500,
        system=CLASSIFICATION_PROMPT,
        messages=[{"role": "user", "content": f"Query: {query}"}],
    )

    result_text = response.content[0].text
    parsed = safe_json_parse(result_text)

    if parsed is None:
        logger.warning(f"Failed to parse classification response: {result_text[:200]}")
        return QueryAnalysis(
            is_topic_query=False,
            reasoning="Failed to parse LLM response",
        )

    return QueryAnalysis(
        is_topic_query=parsed.get("is_topic_query", False),
        entity_type=parsed.get("entity_type", ""),
        search_terms=parsed.get("search_terms", query),
        post_filters=parsed.get("post_filters", []),
        suggested_columns=parsed.get("suggested_columns", []),
        reasoning=parsed.get("reasoning", ""),
    )