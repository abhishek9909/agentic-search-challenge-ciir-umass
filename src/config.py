"""
Configuration and shared utilities for the agentic search pipeline.
"""
import os
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentic_search")

# ── Configuration ────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Central configuration. Reads from env vars with sensible defaults."""

    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    max_search_results: int = 10       # how many URLs to fetch from Tavily
    max_entities: int = 50             # cap on entities in final table
    max_content_chars: int = 15000     # max chars per page to send to LLM
    request_timeout: int = 30          # seconds
    scrape_delay: float = 0.5          # politeness delay between scrapes

    def __post_init__(self):
        load_dotenv()        
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.tavily_api_key = self.tavily_api_key or os.getenv("TAVILY_API_KEY", "")

    def validate(self):
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise ValueError(f"Missing required API keys: {', '.join(missing)}")


# ── Data classes for pipeline state ──────────────────────────────────────────
@dataclass
class SearchResult:
    """A single search result from Tavily."""
    title: str
    url: str
    snippet: str
    content: str = ""         # full extracted content (from Tavily or scraping)
    score: float = 0.0

@dataclass
class Entity:
    """A single extracted entity with its attributes and source tracing."""
    attributes: dict = field(default_factory=dict)     # column_name -> value
    sources: dict = field(default_factory=dict)         # column_name -> source_url
    confidence: float = 1.0

@dataclass
class QueryAnalysis:
    """Result of analyzing/classifying a user query."""
    is_topic_query: bool = False
    entity_type: str = ""                # e.g. "restaurant", "company", "tool"
    search_terms: str = ""               # what to actually search for
    post_filters: list = field(default_factory=list)   # constraints to apply after search
    suggested_columns: list = field(default_factory=list)
    reasoning: str = ""

@dataclass
class PipelineResult:
    """Final output of the full pipeline."""
    query: str
    analysis: Optional[QueryAnalysis] = None
    entities: list = field(default_factory=list)        # list of Entity
    columns: list = field(default_factory=list)          # ordered column names
    search_results_count: int = 0
    processing_time: float = 0.0
    errors: list = field(default_factory=list)

    def to_dict(self):
        return {
            "query": self.query,
            "analysis": asdict(self.analysis),
            "columns": self.columns,
            "entities": [asdict(e) for e in self.entities],
            "search_results_count": self.search_results_count,
            "processing_time": self.processing_time,
            "errors": self.errors,
        }


# ── Utility functions ────────────────────────────────────────────────────────
def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def safe_json_parse(text: str) -> Optional[dict]:
    """Try to parse JSON from LLM output, handling common issues."""
    # strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        # try to find JSON array
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


class Timer:
    """Simple context manager for timing operations."""
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        if self.label:
            logger.info(f"[Timer] {self.label}: {self.elapsed:.2f}s")