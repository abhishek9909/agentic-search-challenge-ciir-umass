"""
Strict Post Filter Mode

For queries with post-filter constraints (e.g. "Asian restaurants in Chicago
that serve halal food"), this module:

1. For EACH (entity, constraint) pair, searches the web specifically for
   evidence of that constraint being satisfied.
2. Collects snippets per (entity, constraint).
3. Makes ONE batched LLM call to score all (entity, constraint) pairs:
   each score is 0-100 (0 = not satisfied, 100 = clearly satisfied).
4. Re-ranks entities by total constraint score (tiebreak: number of
   constraints fully met).
5. Adds "constraints_met" (e.g. "2/3") and "constraint_details" columns.

Design decisions:
- Skipped when post_filters is empty (nothing to verify).
- We cap at 3 constraints and run searches in parallel (max 3 concurrent)
  to respect Tavily rate limits.
- Basic search depth (1 credit each) — we don't need full page content,
  just snippets.
- One batched LLM call for scoring instead of one per pair — much cheaper.
- Entities are RE-RANKED not dropped. Users see everything with scores,
  can sort to see who satisfies constraints best.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tavily import TavilyClient
from anthropic import Anthropic
from .config import Config, Entity, safe_json_parse, truncate_text, logger


MAX_CONSTRAINTS = 3  # cap to control cost
CONSTRAINT_SATISFIED_THRESHOLD = 60  # score >= this counts as "met"


# ── Step 1: Search evidence for each (entity, constraint) pair ───────────────

def _search_constraint(
    entity_name: str,
    constraint: str,
    tavily_client: TavilyClient,
) -> dict:
    """Search for evidence that an entity satisfies a specific constraint."""
    query = f"{entity_name} {constraint}"

    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=3,
        )
    except Exception as e:
        logger.warning(f"Constraint search failed for '{query}': {e}")
        return {"snippets": [], "error": str(e)}

    results = response.get("results", [])
    snippets = [r.get("content", "")[:300] for r in results]

    return {
        "entity_name": entity_name,
        "constraint": constraint,
        "snippets": snippets,
        "result_count": len(results),
    }


def gather_constraint_evidence(
    entities: list[Entity],
    constraints: list[str],
    config: Config,
    max_concurrent: int = 3,
) -> dict:
    """
    Search for (entity, constraint) evidence pairs in parallel.
    
    Returns:
        A dict keyed by (entity_idx, constraint_idx) -> search result dict
    """
    tavily_client = TavilyClient(api_key=config.tavily_api_key)

    # Build all (entity, constraint) pairs
    pairs = []
    for e_idx, entity in enumerate(entities):
        name = entity.attributes.get("name", "")
        if not name:
            continue
        for c_idx, constraint in enumerate(constraints):
            pairs.append((e_idx, c_idx, name, constraint))

    evidence = {}

    def _search(e_idx, c_idx, name, constraint):
        result = _search_constraint(name, constraint, tavily_client)
        return (e_idx, c_idx), result

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = [
            pool.submit(_search, e_idx, c_idx, name, constraint)
            for e_idx, c_idx, name, constraint in pairs
        ]
        for future in as_completed(futures):
            key, result = future.result()
            evidence[key] = result

    logger.info(
        f"Gathered constraint evidence: "
        f"{len(entities)} entities × {len(constraints)} constraints = "
        f"{len(evidence)} searches"
    )
    return evidence


# ── Step 2: LLM scoring of all (entity, constraint) pairs ───────────────────

SCORING_PROMPT = """You are a constraint verification system. Given entities and web search evidence, score how well each entity satisfies each constraint.

For each (entity, constraint) pair, assign a score from 0-100:
- 90-100: Clearly satisfied with strong evidence
- 70-89: Very likely satisfied, good evidence
- 50-69: Possibly satisfied, weak/indirect evidence
- 30-49: Probably not satisfied
- 0-29: Clearly not satisfied or no evidence at all

Constraints to verify: {constraints}

Respond with ONLY a JSON array. One object per entity, with scores for each constraint in order:
[
  {{
    "entity_name": "Name",
    "scores": [85, 20, 70],
    "reasons": ["brief reason for score 1", "brief reason for score 2", "brief reason for score 3"]
  }}
]"""


def score_constraints(
    entities: list[Entity],
    constraints: list[str],
    evidence: dict,
    config: Config,
) -> list[dict]:
    """
    Use LLM to score how well each entity satisfies each constraint.
    
    Returns:
        List of dicts with keys: entity_name, scores, reasons
        One entry per entity, in the same order as input entities.
    """
    # Build evidence text for each entity
    entity_blocks = []
    for e_idx, entity in enumerate(entities):
        name = entity.attributes.get("name", f"Entity {e_idx}")
        lines = [f"ENTITY: {name}"]
        for c_idx, constraint in enumerate(constraints):
            key = (e_idx, c_idx)
            ev = evidence.get(key, {})
            snippets = "\n".join(ev.get("snippets", []))
            lines.append(f"  CONSTRAINT: {constraint}")
            lines.append(f"  EVIDENCE:\n{truncate_text(snippets, 800)}")
        entity_blocks.append("\n".join(lines))

    all_evidence = "\n\n---\n\n".join(entity_blocks)

    prompt = SCORING_PROMPT.format(constraints=json.dumps(constraints))
    client = Anthropic(api_key=config.anthropic_api_key)

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=8192,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Score these entities against the constraints:\n\n{all_evidence}",
                }
            ],
        )
    except Exception as e:
        logger.error(f"Constraint scoring LLM call failed: {e}")
        return []

    parsed = safe_json_parse(response.content[0].text)
    if not parsed or not isinstance(parsed, list):
        logger.warning("Failed to parse constraint scoring response")
        return []

    return parsed


# ── Step 3: Re-rank and annotate entities ──────────────────────────────────

def rerank_by_constraints(
    entities: list[Entity],
    constraints: list[str],
    scoring_results: list[dict],
) -> list[Entity]:
    """
    Add constraint_met count and details to each entity, then re-rank.
    
    Sort order:
      1. By number of constraints fully met (desc)
      2. Then by total score across all constraints (desc)
    """
    # Build a name-based lookup from scoring results
    scores_by_name = {}
    for item in scoring_results:
        name = item.get("entity_name", "").lower().strip()
        scores_by_name[name] = item

    total_constraints = len(constraints)

    for entity in entities:
        name = entity.attributes.get("name", "").lower().strip()
        item = scores_by_name.get(name, {})

        scores = item.get("scores", [])
        reasons = item.get("reasons", [])

        # Pad scores/reasons if LLM returned fewer than expected
        while len(scores) < total_constraints:
            scores.append(0)
        while len(reasons) < total_constraints:
            reasons.append("no data")

        # Count how many constraints are "met" (score >= threshold)
        met_count = sum(1 for s in scores if s >= CONSTRAINT_SATISFIED_THRESHOLD)
        total_score = sum(scores)

        # Build constraint_details string: "halal: 85 (mentioned in menu) | ..."
        detail_parts = []
        for constraint, score, reason in zip(constraints, scores, reasons):
            detail_parts.append(f"{constraint}: {score} ({reason})")
        details = " | ".join(detail_parts)

        entity.attributes["constraints_met"] = f"{met_count}/{total_constraints}"
        entity.attributes["constraint_details"] = details
        entity.attributes["_total_constraint_score"] = total_score
        entity.attributes["_met_count"] = met_count

    # Sort: met_count desc, then total_score desc
    entities.sort(
        key=lambda e: (
            e.attributes.get("_met_count", 0),
            e.attributes.get("_total_constraint_score", 0),
        ),
        reverse=True,
    )

    # Clean up helper fields (keep the display ones)
    for entity in entities:
        entity.attributes.pop("_total_constraint_score", None)
        entity.attributes.pop("_met_count", None)

    return entities


# ── Public API ───────────────────────────────────────────────────────────────

def strict_post_filter(
    entities: list[Entity],
    constraints: list[str],
    config: Config,
) -> tuple[list[Entity], list[str]]:
    """
    Run the full strict post filter pipeline.
    
    Args:
        entities: extracted entities from the pipeline
        constraints: post-filter constraints to verify (from query analysis)
        config: pipeline config
    
    Returns:
        (reranked_entities, new_columns) where new_columns are the
        constraint-related columns added to the schema.
    """
    if not entities or not constraints:
        return entities, []

    # Cap constraints to control cost
    if len(constraints) > MAX_CONSTRAINTS:
        logger.warning(
            f"Capping constraints from {len(constraints)} to {MAX_CONSTRAINTS}"
        )
        constraints = constraints[:MAX_CONSTRAINTS]

    logger.info(
        f"Starting strict post filter: {len(entities)} entities, "
        f"{len(constraints)} constraints"
    )

    # Step 1: Gather evidence per (entity, constraint)
    evidence = gather_constraint_evidence(entities, constraints, config)

    # Step 2: LLM scores all pairs in one call
    scoring_results = score_constraints(entities, constraints, evidence, config)

    if not scoring_results:
        logger.warning("No scoring results returned — skipping re-rank")
        return entities, []

    # Step 3: Re-rank and annotate
    reranked = rerank_by_constraints(entities, constraints, scoring_results)

    new_columns = ["constraints_met", "constraint_details"]

    logger.info(
        f"Strict post filter complete. Top entity: "
        f"{reranked[0].attributes.get('name', '?')} "
        f"({reranked[0].attributes.get('constraints_met', '?')})"
        if reranked else "No entities"
    )

    return reranked, new_columns