"""
Post-Filter: LLM-based re-ranking/filtering for complex queries.

For compound queries like "Asian restaurants in Chicago that serve halal",
the initial search handles "Asian restaurants in Chicago" and this stage
checks each entity against the post-filter constraints ("serves halal").

Design decision: We use the LLM to evaluate each constraint against the
entity's attributes and source content, rather than simple keyword matching.
This handles fuzzy/semantic constraints that aren't directly stated in the data.
"""
import json
from anthropic import Anthropic
from .config import Config, Entity, safe_json_parse, logger


FILTER_PROMPT = """You are a filter/verification system. Given a list of entities and filter constraints, determine which entities satisfy ALL constraints.

For each entity, evaluate whether the available information supports the constraint.
- If the constraint is clearly satisfied → keep
- If the constraint is clearly NOT satisfied → remove
- If uncertain/unknown → keep but mark confidence as "uncertain"

Constraints to check: {constraints}

Respond with ONLY this JSON:
{{
  "filtered_entities": [
    {{
      "index": 0,
      "keep": true,
      "confidence": "high|medium|low",
      "reasoning": "brief explanation"
    }}
  ]
}}"""


def apply_post_filters(
    entities: list[Entity],
    post_filters: list[str],
    config: Config,
) -> list[Entity]:
    """
    Filter entities based on post-hoc constraints using LLM.
    
    If no post_filters, returns entities unchanged.
    """
    if not post_filters or not entities:
        return entities

    logger.info(f"Applying post-filters: {post_filters} to {len(entities)} entities")

    # Build entity summary for LLM
    entity_summaries = []
    for i, entity in enumerate(entities):
        entity_summaries.append(f"Entity {i}: {json.dumps(entity.attributes)}")

    entities_text = "\n".join(entity_summaries)
    prompt = FILTER_PROMPT.format(constraints=json.dumps(post_filters))

    client = Anthropic(api_key=config.anthropic_api_key)

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=2000,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Evaluate these entities:\n\n{entities_text}",
                }
            ],
        )
    except Exception as e:
        logger.error(f"Post-filter LLM call failed: {e}")
        return entities  # fail open: return all if filtering fails

    result_text = response.content[0].text
    parsed = safe_json_parse(result_text)

    if parsed is None:
        logger.warning("Failed to parse filter response, returning all entities")
        return entities

    filtered = []
    for item in parsed.get("filtered_entities", []):
        idx = item.get("index", -1)
        keep = item.get("keep", True)
        confidence = item.get("confidence", "low")

        if 0 <= idx < len(entities) and keep:
            entities[idx].confidence = (
                1.0 if confidence == "high"
                else 0.7 if confidence == "medium"
                else 0.4
            )
            filtered.append(entities[idx])

    logger.info(f"Post-filter: {len(entities)} → {len(filtered)} entities")
    return filtered