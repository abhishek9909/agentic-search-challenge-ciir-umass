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
2. Decompose COMPLEX queries into search_terms vs post_filters:
   - search_terms: the core query to search the web with
   - post_filters: additional constraints that are HARDER to search for directly
     and should be verified AFTER retrieving results using LLM analysis
   
   Example: "Asian restaurants in Chicago that serve halal food"
   → search_terms: "Asian restaurants in Chicago"
   → post_filters: ["serves halal food"]
   
   The rule: if a constraint is an uncommon attribute that search engines won't 
   reliably filter on, make it a post_filter. Location, cuisine type, and primary 
   category should stay in search_terms.

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