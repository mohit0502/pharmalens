"""
api/main.py
PharmaLens FastAPI backend.

Routes
------
GET  /api/indications              — list all indication slugs + display names
GET  /api/companies                — list all companies with tickers
GET  /api/stocks                   — live stock prices for every company (ticker bar)
GET  /api/indication/{slug}        — indication wiki content + structured meta
GET  /api/company/{slug}           — company wiki content + meta + live stock
GET  /api/company/{slug}/trials    — structured trial data (parsed, not raw markdown)
GET  /api/company/{slug}/events    — structured "Recent events" data (canonical CSV)
GET  /api/company/{slug}/news      — BioSpace + yfinance news for one company, last 7 days
GET  /api/news                     — BioSpace articles relevant to tracked companies
GET  /api/news/article             — fetch + parse a single BioSpace article by URL
POST /api/ask                      — Q&A agent, streams SSE

Start:
    uvicorn api.main:app --reload --port 8000
"""

from . import bootstrap  # must run before any google-genai import  # noqa: F401

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import news
from .agent import run_agent
from .tools import get_stock_price, read_wiki_page
from agents.wiki_gcs import read_wiki

load_dotenv()

# ── paths + reference data ────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "reference"

INDICATIONS: dict = json.loads((REFERENCE_DIR / "indications.json").read_text())
COMPANIES: dict = json.loads((REFERENCE_DIR / "companies.json").read_text())

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="PharmaLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _strip_code_fence(content: str) -> str:
    """Remove ```markdown … ``` wrapper the compiler sometimes adds."""
    s = content.strip()
    if s.startswith("```markdown"):
        s = s[len("```markdown"):].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split YAML frontmatter from body. Returns ({}, full_content) if none."""
    content = _strip_code_fence(content)
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, parts[2].strip()
    return {}, content


# ── routes ────────────────────────────────────────────────────────────────────


@app.get("/api/indications")
def get_indications() -> list[dict]:
    """All indication slugs with display names and reference metadata."""
    result = []
    for slug, ref_meta in INDICATIONS.items():
        display_name = slug.replace("-", " ").title()
        content = read_wiki(f"indications/{slug}/_index.md")
        if content:
            fm, _ = _parse_frontmatter(content)
            display_name = fm.get("display_name", display_name)
        result.append({"slug": slug, "display_name": display_name, **ref_meta})
    return result


@app.get("/api/companies")
def get_companies() -> list[dict]:
    """All company slugs with tickers and active indication mapping."""
    return [{"slug": slug, **meta} for slug, meta in COMPANIES.items()]


@app.get("/api/stocks")
def get_stocks() -> list[dict]:
    """Live stock prices for every tracked company (for the ticker bar).
    yfinance calls are network-bound, so fetch all tickers concurrently
    instead of one-by-one — serially this took 10+ seconds for ~20 companies."""
    tickered = [(slug, meta) for slug, meta in COMPANIES.items() if meta.get("ticker")]
    # Cap concurrency — firing all ~20 requests at once risks Yahoo Finance
    # throttling a burst from one IP, which would cost more than it saves.
    with ThreadPoolExecutor(max_workers=min(8, len(tickered) or 1)) as pool:
        stocks = pool.map(lambda sm: get_stock_price(sm[1]["ticker"]), tickered)
    return [
        {"slug": slug, "full_name": meta["full_name"], **stock}
        for (slug, meta), stock in zip(tickered, stocks)
    ]


@app.get("/api/indication/{slug}")
def get_indication(slug: str) -> dict:
    """
    Structured data for one indication.
    Returns reference metadata merged with wiki frontmatter, plus the wiki body.
    """
    if slug not in INDICATIONS:
        raise HTTPException(status_code=404, detail=f"Indication '{slug}' not found")

    content = read_wiki(f"indications/{slug}/_index.md")
    if content:
        fm, wiki_body = _parse_frontmatter(content)
        meta = {**INDICATIONS[slug], **fm}
    else:
        meta = INDICATIONS[slug]
        wiki_body = ""

    return {"slug": slug, "meta": meta, "wiki": wiki_body}


@app.get("/api/company/{slug}")
def get_company(slug: str) -> dict:
    """
    Structured data for one company.
    Returns reference metadata, wiki body, and a live stock quote.
    """
    if slug not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company '{slug}' not found")

    company_meta = COMPANIES[slug]
    company_content = read_wiki(f"companies/{slug}.md")
    if company_content:
        _, wiki_body = _parse_frontmatter(company_content)
    else:
        wiki_body = ""

    ticker = company_meta.get("ticker", "")
    stock = get_stock_price(ticker) if ticker else {}

    # Build drug → [indication slugs] map by scanning each active indication's wiki
    drug_indications: dict[str, list[str]] = {}
    company_drugs = set(company_meta.get("drugs", []))
    for ind_slug in company_meta.get("indications_active", []):
        ind_content = read_wiki(f"indications/{ind_slug}/_index.md")
        if not ind_content:
            continue
        fm, _ = _parse_frontmatter(ind_content)
        for drug in [*fm.get("drugs_approved", []), *fm.get("drugs_pipeline", [])]:
            if drug in company_drugs:
                drug_indications.setdefault(drug, []).append(ind_slug)

    return {
        "slug": slug,
        "meta": company_meta,
        "wiki": wiki_body,
        "stock": stock,
        "drug_indications": drug_indications,
    }


@app.get("/api/company/{slug}/stock-history")
def get_stock_history_route(slug: str, period: str = "1d") -> dict:
    """
    Historical OHLCV data for the company's ticker via yfinance.
    period: 1d | 5d | 1mo | 1y
    Returns {ticker, prev_close, summary, candles: [{t,o,h,l,c,v}]}
    """
    from .tools import get_stock_history

    if slug not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company '{slug}' not found")
    if not COMPANIES[slug].get("ticker"):
        raise HTTPException(status_code=404, detail="No ticker for this company")

    result = get_stock_history(slug, period)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


def _normalize_status(raw: str) -> str:
    """Canonical form: lowercase, spaces/commas/hyphens → underscore, collapsed."""
    from .tools import normalize_status
    return normalize_status(raw)


def _count_concluded_trials(trials: list, lookback_days: int = 365) -> int:
    """
    A trial is 'concluded' if:
      - TERMINATED or WITHDRAWN: always counted (high-signal regardless of date)
      - COMPLETED with primary_completion_date in the lookback window
      - ACTIVE_NOT_RECRUITING that has passed its primary_completion_date,
        also within the lookback window
    """
    from datetime import datetime, timedelta

    cutoff  = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    today   = datetime.now().strftime("%Y-%m-%d")
    count   = 0

    for t in trials:
        status = _normalize_status(str(t.get("status") or ""))
        pcd    = str(t.get("primary_completion_date") or "")

        if status in ("terminated", "withdrawn"):
            count += 1
            continue

        is_done = (
            status == "completed" or
            (status == "active_not_recruiting" and pcd and pcd < today)
        )

        if is_done and pcd >= cutoff:
            count += 1

    return count


@app.get("/api/company/{slug}/trials")
def get_company_trials(slug: str) -> dict:
    """
    Parse the per-company trials wiki and return structured trial data.
    Includes stats (active, concluded, with results) and phase distribution.
    """
    from .tools import parse_company_trials

    if slug not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company '{slug}' not found")

    trials = parse_company_trials(slug)
    if not trials:
        return {"trials": [], "stats": {"active": 0, "completed_90d": 0, "with_results": 0}, "phases": []}

    active_trials = [t for t in trials if t["is_active"]]
    with_results  = [t for t in trials if t.get("has_results")]
    concluded     = _count_concluded_trials(trials)

    # Phase distribution for chart
    phase_order = {"1": 0, "1/2": 1, "2": 2, "2/3": 3, "3": 4, "4": 5, "?": 6}
    phase_map: dict[str, dict] = {}
    for t in trials:
        pd_key = t["phase_display"]
        raw    = str(t.get("phase") or "?").strip()
        if pd_key not in phase_map:
            phase_map[pd_key] = {"phase": pd_key, "sort": phase_order.get(raw, 7), "active": 0, "completed": 0}
        if t["is_active"]:
            phase_map[pd_key]["active"] += 1
        else:
            phase_map[pd_key]["completed"] += 1

    phases = sorted(phase_map.values(), key=lambda x: x["sort"])

    return {
        "trials": trials,
        "stats": {
            "active":        len(active_trials),
            "completed_90d": concluded,
            "with_results":  len(with_results),
        },
        "phases": phases,
    }


@app.get("/api/company/{slug}/events")
def get_company_events(slug: str) -> dict:
    """
    Structured "Recent events" data for one company, read directly from the
    canonical CSV log (agents/wiki_gcs.py:read_company_events) rather than
    regex-parsing the markdown table — same rationale as /trials above.
    """
    from .tools import parse_company_events

    if slug not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company '{slug}' not found")

    return {"events": parse_company_events(slug)}


# ── News (BioSpace reader) ─────────────────────────────────────────────────────


@app.get("/api/news")
def get_news() -> dict:
    """BioSpace articles mentioning a tracked company, newest first."""
    try:
        return {"articles": news.get_relevant_articles()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch news: {e}")


@app.get("/api/news/article")
def get_news_article(url: str) -> dict:
    """Fetch and parse a single BioSpace article by URL."""
    try:
        return news.get_article(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch article: {e}")


@app.get("/api/company/{slug}/news")
def get_company_news_route(slug: str) -> dict:
    """BioSpace + yfinance news for one company, last 7 days, newest first."""
    if slug not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company '{slug}' not found")
    try:
        return {"articles": news.get_company_news(slug)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch company news: {e}")


# ── Q&A agent (SSE) ───────────────────────────────────────────────────────────


class ChatTurn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    indication: str | None = None
    company: str | None = None
    article: str | None = None
    history: list[ChatTurn] = []


@app.post("/api/ask")
async def ask(body: AskRequest) -> StreamingResponse:
    """
    Run the PharmaLens Q&A agent and stream events as SSE.

    Event types:
      tool_call   — agent is about to call a tool
      tool_result — tool returned (first 300 chars shown)
      text        — model text chunk
      done        — stream complete; full_text contains the assembled answer

    `history` carries prior turns from the same browser-side chat session (the
    frontend holds it in React state, cleared on refresh or "Clear chat") so
    follow-up questions like "tell me more about X" have the prior exchange as
    context — the backend itself is stateless across requests.
    """

    async def event_stream():
        history = [t.model_dump() for t in body.history]
        async for event in run_agent(
            body.question, body.indication, body.company, body.article, history
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
