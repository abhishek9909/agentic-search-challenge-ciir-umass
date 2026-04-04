"""
Stage C (continued): Entity Extraction

Uses the LLM to extract structured entity data from scraped web content.
This is the core intelligence of the pipeline — turning unstructured text
into a clean table.

Design decisions:
- We batch all content into a SINGLE LLM call rather than one-per-page.
  This gives the LLM cross-page context to deduplicate and merge info.
- The LLM dynamically infers the schema based on the query + content,
  guided by the suggested_columns from classification.
- Each cell value includes its source URL for traceability.
- We cap at max_entities to keep output manageable.
"""
import json
from anthropic import Anthropic
from .config import Config, SearchResult, QueryAnalysis, Entity, safe_json_parse, logger


EXTRACTION_PROMPT = """You are a structured data extraction system. Given a topic query and web content from multiple sources, extract a list of entities.

RULES:
1. Extract REAL entities found in the provided content. Do NOT invent data.
2. Each entity should have the columns specified. Use null for unknown values.
3. For EACH entity, include a "source_urls" field listing the URLs where info was found.
4. Deduplicate: if the same entity appears in multiple sources, merge the info.
5. Return at most {max_entities} entities.
6. Prioritize entities with more complete information.

The user's query is: "{query}"
Entity type to extract: "{entity_type}"
Columns to extract: {columns}

Respond with ONLY a JSON array (no other text). Each element is one entity:
[
  {{
    "name": "Entity Name",
    "col2": "value2",
    "col3": "value3",
    "source_urls": ["https://where-this-was-found.com"]
  }}
]"""


def extract_entities(
    query: str,
    analysis: QueryAnalysis,
    results: list[SearchResult],
    config: Config,
) -> tuple[list[Entity], list[str]]:
    """
    Extract structured entities from search results using LLM.
    
    Returns:
        (entities, columns) - list of Entity objects and the column order
    """
    if not results:
        return [], analysis.suggested_columns

    # Build content block for LLM
    content_parts = []
    for i, r in enumerate(results):
        content_parts.append(
            f"--- SOURCE {i+1} ---\n"
            f"URL: {r.url}\n"
            f"Title: {r.title}\n"
            f"Content:\n{r.content}\n"
        )
    all_content = "\n".join(content_parts)

    # Format the extraction prompt
    prompt = EXTRACTION_PROMPT.format(
        query=query,
        entity_type=analysis.entity_type,
        columns=json.dumps(analysis.suggested_columns),
        max_entities=config.max_entities,
    )

    client = Anthropic(api_key=config.anthropic_api_key)

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=4096,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract entities from these search results:\n\n{all_content}",
                }
            ],
        )
    except Exception as e:
        logger.error(f"Entity extraction LLM call failed: {e}")
        return [], analysis.suggested_columns

    result_text = response.content[0].text
    parsed = safe_json_parse(result_text)

    if parsed is None:
        logger.error(f"Failed to parse extraction response: {result_text[:300]}")
        return [], analysis.suggested_columns

    # Handle both formats:
    # 1. A flat list: [{"name": "X", "col2": "Y", "source_urls": [...]}, ...]
    # 2. A dict with "entities" key: {"entities": [...], "columns": [...]}
    if isinstance(parsed, list):
        raw_entities = parsed
    elif isinstance(parsed, dict):
        raw_entities = parsed.get("entities", [])
        # If the dict itself looks like a single entity (has "name"), wrap it
        if not raw_entities and "name" in parsed:
            raw_entities = [parsed]
    else:
        logger.error(f"Unexpected parsed type: {type(parsed)}")
        return [], analysis.suggested_columns

    # Convert to Entity objects
    entities = []
    for raw in raw_entities:
        # Separate source_urls from attribute columns
        source_urls = raw.pop("source_urls", [])
        if isinstance(source_urls, str):
            source_urls = [source_urls]

        # Build sources dict: map each attribute to source URLs
        sources = {}
        for key in raw:
            if source_urls:
                sources[key] = source_urls[0]  # primary source

        entities.append(Entity(
            attributes=raw,
            sources=sources,
        ))

    # Infer columns from the entities themselves (preserves LLM's column order)
    if entities:
        # Use the keys from the first entity, then add any extras from others
        seen = set()
        columns = []
        for entity in entities:
            for key in entity.attributes:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    else:
        columns = analysis.suggested_columns

    logger.info(f"Extracted {len(entities)} entities with {len(columns)} columns")
    return entities, columns