"""
Pipeline Orchestrator

Ties together all stages of the agentic search pipeline:
  A. Query Classification → is it a topic query? decompose it.
  B. Web Search → find relevant pages via Tavily
  C. Content Parsing → enrich content via scraping if needed
  C2. Entity Extraction → LLM extracts structured table from content
  D. Post-Filtering → LLM verifies entities against complex constraints
       (or D-strict: per-entity constraint search + scoring + re-rank)
  E. Review Bomb (optional) → search reviews per entity, re-rank by popularity

This module provides both a step-by-step API and a single run() function.
"""
import time
from .config import Config, PipelineResult, Timer, logger
from .query_classifier import classify_query
from .web_searcher import search_web
from .content_parser import enrich_content
from .entity_extractor import extract_entities
from .post_filter import apply_post_filters
from .strict_post_filter import strict_post_filter
from .review_bomb import review_bomb


def run(
    query: str,
    config: Config | None = None,
    enable_review_bomb: bool = False,
    enable_strict_post_filter: bool = False,
) -> PipelineResult:
    """
    Run the full agentic search pipeline on a query.
    
    Args:
        query: The user's topic query
        config: Configuration (uses defaults + env vars if None)
    
    Returns:
        PipelineResult with entities, columns, and metadata
    """
    if config is None:
        config = Config()
    config.validate()

    result = PipelineResult(query=query)
    start_time = time.time()

    # ── Stage A: Classify the query ──────────────────────────────────────
    with Timer("Stage A: Query Classification"):
        analysis = classify_query(query, config)
        result.analysis = analysis

    if not analysis.is_topic_query:
        logger.info(f"Query rejected: not a topic query. Reason: {analysis.reasoning}")
        result.processing_time = time.time() - start_time
        return result

    logger.info(
        f"Query accepted: entity_type={analysis.entity_type}, "
        f"search='{analysis.search_terms}', "
        f"post_filters={analysis.post_filters}"
    )

    # ── Stage B: Search the web ──────────────────────────────────────────
    with Timer("Stage B: Web Search"):
        search_results = search_web(analysis.search_terms, config)
        result.search_results_count = len(search_results)

    if not search_results:
        result.errors.append("No search results found")
        result.processing_time = time.time() - start_time
        return result

    # ── Stage C: Enrich content ──────────────────────────────────────────
    with Timer("Stage C: Content Enrichment"):
        enriched_results = enrich_content(search_results, config)

    # ── Stage C2: Extract entities ───────────────────────────────────────
    with Timer("Stage C2: Entity Extraction"):
        entities, columns = extract_entities(query, analysis, enriched_results, config)

    if not entities:
        result.errors.append("No entities extracted from content")
        result.processing_time = time.time() - start_time
        return result

    # ── Stage D: Post-filter (for complex queries) ───────────────────────
    if analysis.post_filters:
        if enable_strict_post_filter:
            with Timer("Stage D-strict: Strict Post Filter"):
                entities, strict_cols = strict_post_filter(
                    entities, analysis.post_filters, config
                )
                # Add constraint columns to schema (after name if present)
                for col in reversed(strict_cols):
                    if col not in columns:
                        name_idx = columns.index("name") + 1 if "name" in columns else len(columns)
                        columns.insert(name_idx, col)
        else:
            with Timer("Stage D: Post-Filtering"):
                entities = apply_post_filters(entities, analysis.post_filters, config)

    # ── Stage E: Review Bomb (optional) ──────────────────────────────────
    if enable_review_bomb and entities:
        with Timer("Stage E: Review Bomb"):
            entities, new_cols = review_bomb(entities, analysis.entity_type, config)
            # Add popularity columns to the front (after name)
            for col in reversed(new_cols):
                if col not in columns:
                    # Insert after "name" if it exists, else at end
                    name_idx = columns.index("name") + 1 if "name" in columns else len(columns)
                    columns.insert(name_idx, col)

    result.entities = entities
    result.columns = columns
    result.processing_time = time.time() - start_time

    logger.info(
        f"Pipeline complete: {len(entities)} entities, "
        f"{len(columns)} columns, {result.processing_time:.2f}s"
    )

    return result


def display_result(result: PipelineResult) -> str:
    """Format a PipelineResult as a readable string with table."""
    import pandas as pd

    lines = []
    lines.append(f"Query: {result.query}")
    lines.append(f"Is topic query: {result.analysis.is_topic_query}")

    if not result.analysis.is_topic_query:
        lines.append(f"Reason: {result.analysis.reasoning}")
        return "\n".join(lines)

    lines.append(f"Entity type: {result.analysis.entity_type}")
    lines.append(f"Search terms: {result.analysis.search_terms}")
    if result.analysis.post_filters:
        lines.append(f"Post-filters: {result.analysis.post_filters}")
    lines.append(f"Search results: {result.search_results_count}")
    lines.append(f"Entities found: {len(result.entities)}")
    lines.append(f"Processing time: {result.processing_time:.2f}s")

    if result.errors:
        lines.append(f"Errors: {result.errors}")

    if result.entities and result.columns:
        # Build DataFrame
        rows = [e.attributes for e in result.entities]
        df = pd.DataFrame(rows)
        # Reorder columns to match result.columns, adding missing ones
        ordered_cols = [c for c in result.columns if c in df.columns]
        extra_cols = [c for c in df.columns if c not in result.columns]
        df = df[ordered_cols + extra_cols]

        lines.append("\n" + df.to_string(index=False))

        # Source summary
        lines.append("\n--- Sources ---")
        for i, entity in enumerate(result.entities):
            name = entity.attributes.get("name", f"Entity {i}")
            unique_sources = set(entity.sources.values())
            if unique_sources:
                lines.append(f"  {name}: {', '.join(s for s in unique_sources if s)}")

    return "\n".join(lines)