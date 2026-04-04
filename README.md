# Agentic Search

A system that takes a topic query (e.g., *"AI startups in healthcare"*, *"top pizza places in Brooklyn"*) and produces a structured, source-traced table of discovered entities with relevant attributes — all sourced from the live web.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd agentic-search
pip install -r requirements.txt

# 2. Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export TAVILY_API_KEY=tvly-...

# 3. Run the server (serves both API and UI)
python api/server.py
# → open http://localhost:8000
```

**API keys needed:**
- **Anthropic** — [console.anthropic.com](https://console.anthropic.com/) (for Claude Haiku 4.5)
- **Tavily** — [app.tavily.com](https://app.tavily.com/) (1,000 free credits/month, no credit card)

---

## What It Does

Type a topic query → get back a sortable table of real entities extracted from the web, each cell traceable to its source URL.

**Example:** `"AI startups in healthcare"` →

| name | focus_area | funding | headquarters |
|------|-----------|---------|-------------|
| Tempus | Precision medicine | $1.1B | Chicago |
| Biofourmis | Remote monitoring | $300M | Boston |
| ... | ... | ... | ... |

Every value links back to the web page it was extracted from.

---

## Architecture

The pipeline has six stages, each a separate module:

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  A. Query Classification (LLM) │  Is this a topic query?
│     Decompose into search       │  What entity type?
│     terms + post-filters        │  What columns to extract?
└──────────────┬──────────────────┘
               │
    ▼                        (reject non-topic queries)
┌─────────────────────────────────┐
│  B. Web Search (Tavily)         │  10 results with extracted content
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  C. Content Enrichment          │  Supplement thin results via
│     (trafilatura)               │  scraping + text extraction
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  C2. Entity Extraction (LLM)   │  Single batched call:
│      all sources → structured   │  deduplicate, merge, extract
│      JSON array                 │  flat table with source URLs
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  D. Post-Filter (LLM)          │  For complex queries only —
│     (optional)                  │  verify constraints like
│                                 │  "serves halal", "has outdoor seating"
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  E. Review Bomb (LLM + Tavily) │  Optional: search reviews per entity,
│     (optional)                  │  score 0-100, re-rank by popularity
└──────────────┬──────────────────┘
               │
    ▼
  Structured Table + Sources
```

### File Structure

```
agentic-search/
├── README.md
├── requirements.txt
├── .env.example
├── api/
│   └── server.py              # FastAPI server (API + serves frontend)
├── frontend/
│   └── index.html             # Single-file UI (no build tools)
├── src/
│   ├── config.py              # Shared config, dataclasses, utilities
│   ├── query_classifier.py    # Stage A: LLM query classification
│   ├── web_searcher.py        # Stage B: Tavily web search
│   ├── content_parser.py      # Stage C: trafilatura content enrichment
│   ├── entity_extractor.py    # Stage C2: LLM entity extraction
│   ├── post_filter.py         # Stage D: LLM post-hoc filtering
│   ├── review_bomb.py         # Stage E: popularity re-ranking
│   └── pipeline.py            # Orchestrator tying all stages together
├── notebooks/
│   └── poc_pipeline.py        # Step-by-step POC (9 testable cells)
└── data/
    └── queries.json           # 55 annotated queries (40 train, 15 eval)
```

---

## Design Decisions & Tradeoffs

### 1. Query Decomposition for Complex Queries

**Problem:** A query like *"Asian restaurants in Chicago that serve halal food"* mixes a searchable core (*Asian restaurants in Chicago*) with a constraint that search engines can't reliably filter (*serves halal*).

**Decision:** The LLM classifier decomposes queries into `search_terms` (sent to the search API) and `post_filters` (verified by a separate LLM call after entity extraction). This two-phase approach avoids overburdening the search stage with constraints it can't handle, while still enforcing them.

**Tradeoff:** Post-filtering depends on whether the scraped content mentions the constraint. If no source mentions "halal", the filter can't verify it — entities get kept with low confidence rather than wrongly discarded.

### 2. Batched Entity Extraction (Single LLM Call)

**Problem:** We could call the LLM once per search result page, but that loses cross-page context (duplicates, conflicting info).

**Decision:** All search result content is concatenated and sent in a single LLM call. The LLM sees everything at once and can deduplicate entities that appear across multiple sources, merge partial information, and pick the most complete version.

**Tradeoff:** This means the input can be large (10 pages × ~15K chars each). We mitigate by truncating each page to `max_content_chars` (15K default) and capping at 10 search results. Even so, this is the most token-heavy call in the pipeline.

### 3. Dynamic Schema Inference

**Problem:** Different queries need different table columns. *"restaurants"* need address/price/rating; *"startups"* need funding/founders/sector.

**Decision:** The classifier suggests columns based on the query, and the extractor uses them as guidance — but the LLM can adjust based on what's actually in the content. Columns are inferred from the first entity's keys, so the schema adapts to the data.

**Tradeoff:** Column names aren't always consistent across runs for the same query. A more rigid approach would predefine schemas per entity type, but that limits the system's ability to handle novel query types.

### 4. Tavily as Primary Content Source (Minimal Scraping)

**Problem:** Scraping is slow, blocked by many sites, and fragile.

**Decision:** We use Tavily's `advanced` search depth, which returns extracted content alongside results. We only scrape supplementally when Tavily's content is too short (<200 chars). This cuts scraping from ~10 requests down to typically 0-2.

**Tradeoff:** Tavily's `advanced` search costs 2 credits per query instead of 1. But it saves significant latency and avoids scraping failures. The 1,000 free monthly credits support ~500 advanced searches.

### 5. Review Bomb as Opt-In

**Problem:** Popularity re-ranking requires searching for each entity individually — 15 entities means 15 additional Tavily searches.

**Decision:** Review Bomb is off by default, toggled via a checkbox in the UI or `enable_review_bomb=True` in the API. When enabled, it searches `"{entity name} reviews ratings"` per entity (basic depth, 1 credit each), collects signals, and uses a single LLM call to score and re-rank.

**Tradeoff:** Significantly increases both cost and latency. A typical query goes from ~$0.11/10 Tavily credits to ~$0.18/40 Tavily credits with Review Bomb. Latency roughly doubles. The parallel search (3 concurrent) helps, but it's still the most expensive stage.

### 6. Haiku Over Sonnet

**Problem:** The pipeline makes 2-4 LLM calls per query. Sonnet is ~25× more expensive than Haiku per token.

**Decision:** Default to Claude Haiku 4.5 for all LLM calls. The tasks (classification, extraction, filtering, ranking) are structured-output tasks where Haiku performs well — they don't need Sonnet's deeper reasoning.

**Tradeoff:** Haiku occasionally produces malformed JSON or less precise extractions. We compensate with robust JSON parsing (fence stripping, truncation repair) and higher `max_tokens` (8192) to avoid mid-response truncation. The cost savings (~$0.11 vs ~$2+ per query with Sonnet) justify the tradeoff. Users can override to Sonnet via config for higher-stakes queries.

### 7. Truncated JSON Recovery

**Problem:** Haiku sometimes runs out of tokens mid-JSON, producing `[{...}, {... (truncated)`. Standard JSON parsers reject the entire response.

**Decision:** `safe_json_parse` has a 4-strategy approach:
1. Strip markdown code fences (` ```json `)
2. Direct parse
3. Find outermost `[...]` or `{...}`
4. **Truncation repair** — walk through character-by-character with brace counting, extract every *complete* `{...}` object, discard the truncated tail

**Tradeoff:** We may lose 1-2 entities from the end of a truncated response, but we never lose the entire result. A warning is logged when repair kicks in.

### 8. Source Tracing Per Cell

**Problem:** Users need to know *where* each piece of information came from.

**Decision:** Each entity carries a `sources` dict mapping column names to the URL where that value was found. In the UI, cells with sources render as clickable links.

**Tradeoff:** Source granularity is per-entity (primary source URL), not truly per-cell — the LLM assigns the same source URL to all attributes from the same page. True per-cell tracing would require a more complex extraction prompt and significantly more tokens.

---

## Cost & Performance

Measured on representative queries using Claude Haiku 4.5 + Tavily:

| Mode | Anthropic Cost | Tavily Credits | Latency |
|------|---------------|----------------|---------|
| Standard pipeline | ~$0.11 | ~10 | 10-20s |
| With Review Bomb | ~$0.18 | ~40 | 20-40s |

**Tavily free tier:** 1,000 credits/month. Standard queries use ~10 credits each (~100 queries/month). With Review Bomb, ~40 credits each (~25 queries/month).

**Anthropic cost:** Haiku 4.5 is very cheap. Even heavy usage stays under $5/month.

---

## API Reference

### `POST /search`

```json
{
  "query": "AI startups in healthcare",
  "review_bomb": false
}
```

**Response:**

```json
{
  "query": "AI startups in healthcare",
  "analysis": {
    "is_topic_query": true,
    "entity_type": "company",
    "search_terms": "AI startups in healthcare",
    "post_filters": [],
    "suggested_columns": ["name", "focus_area", "funding", "location"],
    "reasoning": "..."
  },
  "columns": ["name", "focus_area", "funding", "location"],
  "entities": [
    {
      "attributes": {
        "name": "Tempus",
        "focus_area": "Precision medicine",
        "funding": "$1.1B",
        "location": "Chicago"
      },
      "sources": {
        "name": "https://example.com/article",
        "focus_area": "https://example.com/article"
      },
      "confidence": 1.0
    }
  ],
  "search_results_count": 10,
  "processing_time": 14.2,
  "errors": []
}
```

### `GET /health`

Returns `{"status": "ok"}`.

### `GET /`

Serves the frontend UI.

---

## Running the POC Notebook

The notebook script has 9 independently-runnable cells:

```bash
# Run all cells end-to-end
python notebooks/poc_pipeline.py

# Or import specific cells in Python/IPython:
from notebooks.poc_pipeline import cell_1_setup, cell_9_test_review_bomb
config = cell_1_setup()
cell_9_test_review_bomb(config)
```

| Cell | What it tests |
|------|--------------|
| 1 | Setup & config validation |
| 2 | Load query dataset |
| 3 | Query classification (accepts/rejects) |
| 4 | Tavily web search |
| 5 | Content enrichment via scraping |
| 6 | LLM entity extraction |
| 7 | Full pipeline (multiple queries) |
| 8 | Complex query with post-filtering |
| 9 | Review Bomb before/after comparison |

---

## Query Dataset

`data/queries.json` contains 55 annotated queries:

- **40 train** — used during development to test pipeline behavior
- **15 eval** — held out, includes 3 negative examples (non-topic queries that should be rejected)
- **10 negative examples** — conversational/factual queries for classifier testing

Each query is annotated with: category, complexity level (simple/moderate/complex), expected entity type, expected columns, pre-decomposed search terms, and post-filters.

---

## Known Limitations

1. **Schema inconsistency** — The same query may produce slightly different column names across runs (e.g., `"price"` vs `"price_range"`). There's no schema registry enforcing consistency.

2. **Entity coverage depends on search quality** — If Tavily's top 10 results don't mention an entity, it won't appear. Niche or very new entities may be missed.

3. **No pagination / "search more"** — The system returns one batch of up to 50 entities. There's no mechanism to fetch more results beyond the initial search.

4. **Source tracing is approximate** — Sources are per-entity (which URL the entity came from), not truly per-cell. If the LLM merges info from two sources, only the primary source is tracked.

5. **Review Bomb is noisy for obscure entities** — If an entity has very little web presence, the review search returns irrelevant results, leading to unreliable popularity scores.

6. **Rate limits** — Tavily free tier is 1 request/second. The Review Bomb's parallel searches (3 concurrent) may occasionally hit this. The pipeline doesn't currently retry on rate limit errors.

7. **No caching** — Every query runs the full pipeline fresh. Repeated queries re-search and re-extract, costing the same credits each time.

---

## Possible Extensions

- **Caching layer** — Cache search results and extracted entities by query hash, with a TTL for freshness
- **Multi-round search** — If the first round yields few entities, automatically broaden the query and search again
- **Schema registry** — Pre-define column schemas per entity type for consistency across runs
- **Streaming** — Stream entities to the UI as they're extracted rather than waiting for the full pipeline
- **Export** — CSV/JSON download of results
- **Deployment** — Containerize with Docker, deploy to Fly.io or Railway free tier