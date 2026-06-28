"""
api/tools.py
Wiki read/list helpers and yfinance stock price lookup.
These are called both by FastAPI route handlers and by the Q&A agent tool loop.
"""

import re
import time
import yfinance as yf
import yaml
import json
from pathlib import Path

from agents.wiki_gcs import (
    read_wiki, list_wiki, search_wiki as _search_wiki, search_paths, read_company_events,
)
from .news import COMPANIES

_REFERENCE_DIR = Path(__file__).parent.parent / "reference"
_DRUGS_BY_COMPANY: dict[str, list[str]] | None = None


def _drugs_for_company(slug: str) -> list[str]:
    """INNs of drugs tracked under this company in reference/drugs.json.
    Loaded once and cached for the life of the process — the reference file
    only changes via a deploy, never at runtime."""
    global _DRUGS_BY_COMPANY
    if _DRUGS_BY_COMPANY is None:
        drugs = json.loads((_REFERENCE_DIR / "drugs.json").read_text())
        by_company: dict[str, list[str]] = {}
        for inn, meta in drugs.items():
            by_company.setdefault(meta.get("company", ""), []).append(inn)
        _DRUGS_BY_COMPANY = by_company
    return _DRUGS_BY_COMPANY.get(slug, [])

_STOCK_CACHE_TTL_SECONDS = 20
_stock_cache: dict[str, tuple[float, dict]] = {}

_HISTORY_INTERVAL = {"1d": "5m", "5d": "30m", "1mo": "1d", "1y": "1d"}


def resolve_ticker(company_slug: str) -> str | None:
    """Look up a company's ticker from reference/companies.json. Lets callers
    (the Q&A agent in particular) work with the same company slugs used
    everywhere else in the app instead of needing to already know tickers."""
    return COMPANIES.get(company_slug, {}).get("ticker")


def normalize_status(raw: str) -> str:
    """Canonical form: lowercase, spaces/commas/hyphens → underscore, collapsed."""
    return re.sub(r"[\s,\-]+", "_", (raw or "").strip().lower()).strip("_")


def normalize_phase(raw: str) -> str:
    """Canonical form for combined-phase trials: "1 | 2", "1|2", "1 /2" all
    collapse to "1/2". The same trial phase has been written with different
    separator styles across companies/batches (confirmed in production:
    "1 | 2" / "1/2" / "1|2" all present for what's the same Phase 1/2
    bucket), which fragmented the phase-distribution chart into duplicate
    bars for the same phase. Also treats a literal "None"/"N/A" string
    (the same YAML-string-vs-null gotcha documented elsewhere in this repo)
    as unspecified rather than a real phase value."""
    raw = (raw or "").strip()
    if not raw or raw.lower() in ("none", "n/a", "null", "?"):
        return "?"
    return re.sub(r"\s*\|\s*", "/", raw)


def parse_company_trials(slug: str) -> list[dict]:
    """Parse wiki/trials/{slug}.md into a list of trial dicts (frontmatter only),
    sorted newest primary_completion_date first. Shared by the FastAPI route and
    the Q&A agent's get_company_trials tool — both need the same structured data,
    not the raw markdown (which is too large for the agent's tool-result budget)."""
    content = read_wiki(f"trials/{slug}.md")
    if not content:
        return []
    blocks = re.split(r"^---$", content, flags=re.MULTILINE)
    trials = []
    for block in blocks:
        try:
            meta = yaml.safe_load(block.strip())
        except yaml.YAMLError:
            continue
        if not isinstance(meta, dict) or "trial_id" not in meta:
            continue
        # YAML parses unquoted dates as datetime.date — normalize to str for
        # consistent sorting/serialization (mixed str/date entries otherwise
        # break both the sort comparison and json.dumps).
        for date_field in ("primary_completion_date", "last_updated"):
            if meta.get(date_field) is not None:
                meta[date_field] = str(meta[date_field])
        raw_phase = normalize_phase(str(meta.get("phase") or ""))
        meta["phase"] = raw_phase
        meta["phase_display"] = f"Phase {raw_phase}" if raw_phase != "?" else "Phase unspecified"
        meta["is_active"] = normalize_status(str(meta.get("status") or "")) in {
            "recruiting", "active", "not_yet_recruiting",
            "enrolling_by_invitation", "approved_for_marketing",
            "active_not_recruiting",
        }
        trials.append(meta)
    trials.sort(key=lambda t: t.get("primary_completion_date") or "", reverse=True)
    return trials


def parse_company_events(slug: str) -> list[dict]:
    """Read the canonical per-company event log (CSV, written deterministically
    by the compiler — never parsed out of markdown). Returns [{date, type, event,
    signal, source}], newest first. Empty rows from a missing/blank CSV become []."""
    rows = read_company_events(slug)
    events = [
        {
            "date":   r.get("date", ""),
            "type":   r.get("type", ""),
            "event":  r.get("event", ""),
            "signal": r.get("signal", ""),
            "source": r.get("source", ""),
        }
        for r in rows
        if r.get("event")
    ]
    events.sort(key=lambda e: e["date"], reverse=True)
    return events


def read_wiki_page(page_path: str) -> str:
    """Read a wiki page. Returns an error string if the path doesn't exist."""
    content = read_wiki(page_path)
    if not content:
        return f"Page not found: {page_path}"
    stripped = content.strip()
    if stripped.startswith("```markdown"):
        stripped = stripped[len("```markdown"):].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
        content = stripped
    return content


def list_wiki_pages(prefix: str = "") -> list[str]:
    """List .md pages under wiki/<prefix>."""
    pages = list_wiki(prefix)
    if not pages:
        return [f"Directory not found: {prefix}"] if prefix else []
    return pages


def search_wiki(query: str, prefix: str = "") -> list[dict]:
    """Full-text search across wiki files. Returns [{path, snippet}] up to 20 matches."""
    return _search_wiki(query, prefix)


def search_company_wiki(company_slug: str, query: str) -> list[dict]:
    """Full-text search scoped to one company's known pages — its company page,
    trial roster, drug pages, and event log entries — instead of every page in
    the wiki. Exists for terms that won't be found via read_wiki_page (e.g. a
    drug not yet in reference/drugs.json, like Casgevy for Vertex) but that
    still show up in plain-text coverage of that company: company-wide
    search_wiki() has to scan the entire (multi-thousand-page) wiki to find a
    sparse, scattered term and can blow the tool-call timeout, whereas this is
    bounded to a handful of files and stays fast regardless of wiki size."""
    paths = [f"companies/{company_slug}.md", f"trials/{company_slug}.md"]
    paths += [f"drugs/{inn}.md" for inn in _drugs_for_company(company_slug)]

    # Event pages aren't named by a fixed slug pattern (e.g. both
    # "vertex-pharmaceuticals-..." and "vertex-..." exist for the same
    # company) — match on any sufficiently distinctive token from the slug.
    tokens = [t for t in company_slug.split("-") if len(t) > 3]
    if tokens:
        for path in list_wiki("events"):
            if any(t in path for t in tokens):
                paths.append(path)

    results = search_paths(query, paths)
    # The events the company-events CSV log already has on file (separate from
    # the wiki pages above) are the canonical source for the page's own
    # "Recent events" table — surface a direct match there too in case the
    # term appears in event text but not in the underlying markdown files.
    query_lower = query.lower()
    for row in read_company_events(company_slug):
        if query_lower in row.get("event", "").lower():
            results.append({
                "path": f"company_events/{company_slug}",
                "snippet": f"{row.get('date', '')}: {row.get('event', '')}",
            })
    return results


def get_stock_price(ticker: str) -> dict:
    """Return current price, change, and % change for a ticker via yfinance.
    Cached for _STOCK_CACHE_TTL_SECONDS — the ticker bar and company page both
    call this on every load, and yfinance is slow enough that re-fetching the
    same ticker within a few seconds just adds latency for no fresher data."""
    ticker = ticker.upper()
    cached = _stock_cache.get(ticker)
    if cached and time.time() - cached[0] < _STOCK_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = fi.last_price
        prev = fi.previous_close
        change = price - prev if price and prev else 0.0
        change_pct = (change / prev * 100) if prev else 0.0
        result = {
            "ticker": ticker,
            "price": round(price, 2) if price else None,
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
        }
    except Exception as e:
        result = {"ticker": ticker, "price": None, "error": str(e)}
    _stock_cache[ticker] = (time.time(), result)
    return result


def get_stock_history(company_slug: str, period: str = "1mo") -> dict:
    """Historical OHLCV + a compact summary (start/end price, period high/low,
    % change) for a company's ticker. period: 1d | 5d | 1mo | 1y. The summary
    fields exist so the Q&A agent can answer "why did the price move" questions
    without needing to eyeball a full candle list itself."""
    ticker = resolve_ticker(company_slug)
    if not ticker:
        return {"error": f"Unknown company slug: {company_slug}"}

    interval = _HISTORY_INTERVAL.get(period, "1d")
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        stock = get_stock_price(ticker)
        prev_close = (
            round(stock["price"] - stock["change"], 4)
            if stock.get("price") is not None
            else None
        )

        candles = [
            {
                "t": str(idx),
                "o": round(row.Open, 4),
                "h": round(row.High, 4),
                "l": round(row.Low, 4),
                "c": round(row.Close, 4),
                "v": int(row.Volume),
            }
            for idx, row in df.iterrows()
        ]

        closes = [c["c"] for c in candles]
        summary = {
            "start_price": closes[0] if closes else None,
            "end_price": closes[-1] if closes else None,
            "pct_change_over_period": (
                round((closes[-1] / closes[0] - 1) * 100, 2) if len(closes) >= 2 and closes[0] else None
            ),
            "period_high": max((c["h"] for c in candles), default=None),
            "period_low": min((c["l"] for c in candles), default=None),
        }

        return {"ticker": ticker, "period": period, "prev_close": prev_close, "summary": summary, "candles": candles}
    except Exception as e:
        return {"ticker": ticker, "period": period, "error": str(e)}
