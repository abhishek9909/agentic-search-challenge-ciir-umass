"""
Minimal API server for the agentic search pipeline.

Endpoints:
  POST /search  — run a topic query, returns structured entities
  GET  /health  — health check
  GET  /        — serves the frontend

Run:
  cd agentic-search && python api/server.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.config import Config
from src.pipeline import run

app = FastAPI(title="Agentic Search", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend",
)


class SearchRequest(BaseModel):
    query: str
    review_bomb: bool = False
    strict_post_filter: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    try:
        config.validate()
    except ValueError as e:
        raise HTTPException(500, str(e))

    result = run(
        req.query.strip(),
        config,
        enable_review_bomb=req.review_bomb,
        enable_strict_post_filter=req.strict_post_filter,
    )
    return result.to_dict()


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)