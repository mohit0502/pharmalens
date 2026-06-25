"""
agents/wiki_gcs.py
Wiki file adapter for PharmaLens.

When GCS_MODE=true (production / Cloud Run), wiki pages and the processing
state file live in a bucket we own — separate from the raw input data bucket
(pharmalens-raw, owned by the data-engineering side) so the two have independent
failure domains, retention policies, and IAM. Raw input reads still go through
agents/gcs.py against GCS_BUCKET; this module writes the compiled output to
WIKI_BUCKET:
  gs://<WIKI_BUCKET>/wiki/<page_path>
  gs://<WIKI_BUCKET>/state/processing_state.json

When GCS_MODE is unset or false (local dev), all operations fall back to the
local wiki/ and agents/ directories — no code changes needed for dev workflow.

Environment variables (all optional in dev):
  GCS_MODE    = true            → use GCS storage
  WIKI_BUCKET = pharmalens-wiki  → output bucket name (default: pharmalens-wiki)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.logger import get_logger

logger = get_logger("pharmalens.wiki_gcs")

try:
    BASE_DIR = Path(__file__).parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

LOCAL_WIKI_DIR = BASE_DIR / "wiki"
WIKI_GCS_PREFIX = "wiki"
STATE_GCS_KEY = "state/processing_state.json"
LOCK_GCS_KEY = "state/compiler.lock"

# Canonical per-company event log — the source of truth for each company page's
# "Recent events" table. Stored as CSV (not JSON) since it's a flat tabular
# record (date, type, event, signal, source) with no nesting, and Python — not
# the LLM — owns rendering the markdown table from this data on every flush.
EVENTS_CSV_PREFIX = "state/company_events"
LOCAL_EVENTS_DIR = BASE_DIR / "data" / "company_events"
EVENT_CSV_FIELDS = ["date", "type", "event", "signal", "source", "file_path"]

# Canonical per-company earnings-intelligence log — same shape of fix as the
# events log above, for the section of the company page that grows one entry
# per financial filing forever (one paragraph per 8-K/10-Q, every quarter,
# indefinitely) rather than one row per discrete event.
EARNINGS_CSV_PREFIX = "state/company_earnings"
LOCAL_EARNINGS_DIR = BASE_DIR / "data" / "company_earnings"
EARNINGS_CSV_FIELDS = ["date", "text", "file_path"]

# Same fix, applied to the two append-forever sections on drug pages and the
# cross-company event feed on indication hub pages.
DRUG_TIMELINE_CSV_PREFIX = "state/drug_timeline"
LOCAL_DRUG_TIMELINE_DIR = BASE_DIR / "data" / "drug_timeline"
DRUG_TIMELINE_CSV_FIELDS = ["date", "event", "type", "file_path"]

DRUG_EVIDENCE_CSV_PREFIX = "state/drug_clinical_evidence"
LOCAL_DRUG_EVIDENCE_DIR = BASE_DIR / "data" / "drug_clinical_evidence"
DRUG_EVIDENCE_CSV_FIELDS = ["date", "text", "file_path"]

# Drug page "Management sentiment" — confirmed in testing to be the same
# unbounded-growth problem (semaglutide's body hit MAX_TOKENS at 170KB): one
# rolling paragraph that accumulated commentary from every earnings filing
# ever processed, reproduced in full on every flush.
DRUG_SENTIMENT_CSV_PREFIX = "state/drug_management_sentiment"
LOCAL_DRUG_SENTIMENT_DIR = BASE_DIR / "data" / "drug_management_sentiment"
DRUG_SENTIMENT_CSV_FIELDS = ["date", "text", "file_path"]

INDICATION_EVENTS_CSV_PREFIX = "state/indication_events"
LOCAL_INDICATION_EVENTS_DIR = BASE_DIR / "data" / "indication_events"
INDICATION_EVENTS_CSV_FIELDS = ["date", "event", "signal", "file_path"]
# Task timeout on the Cloud Run job is 1 day — a lock older than this can only
# mean the process that held it died without releasing it, so treat it as stale.
LOCK_STALE_AFTER_SECONDS = 20 * 3600

# Wiki content only changes once a day (the compiler job's daily run), so a
# long-lived process (the Render API) can cache aggressively — this avoids a
# GCS round-trip on every single page read for every request. A write updates
# the cache immediately, so a process never sees its own write as stale.
_CACHE_TTL_SECONDS = 180
_page_cache: dict[str, tuple[float, str]] = {}      # page_path -> (cached_at, content)
_list_cache: dict[str, tuple[float, list[str]]] = {}  # prefix -> (cached_at, pages)

_client_singleton = None


def _gcs_enabled() -> bool:
    return os.environ.get("GCS_MODE", "").lower() in ("true", "1", "yes")


def _bucket_name() -> str:
    return os.environ.get("WIKI_BUCKET", "pharmalens-wiki")


def _client():
    # Creating storage.Client() does credential/auth setup each time — expensive
    # if repeated on every call. Reuse one client for the life of the process.
    global _client_singleton
    if _client_singleton is None:
        from google.cloud import storage
        _client_singleton = storage.Client()
    return _client_singleton


# ── wiki reads / writes ───────────────────────────────────────────────────────

def read_wiki(page_path: str) -> str:
    """Read a wiki page. Returns '' if not found."""
    if _gcs_enabled():
        cached = _page_cache.get(page_path)
        if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1]
        from google.api_core.exceptions import NotFound
        try:
            blob = _client().bucket(_bucket_name()).blob(f"{WIKI_GCS_PREFIX}/{page_path}")
            content = blob.download_as_text(encoding="utf-8")
        except NotFound:
            content = ""
        except Exception as e:
            logger.warning(f"WIKI | GCS read failed for {page_path}: {e}")
            # serve a stale cache entry rather than nothing, if we have one
            return cached[1] if cached else ""
        _page_cache[page_path] = (time.time(), content)
        return content
    full_path = LOCAL_WIKI_DIR / page_path
    return full_path.read_text() if full_path.exists() else ""


def write_wiki(page_path: str, content: str) -> str:
    """Write a wiki page. Returns page_path."""
    if _gcs_enabled():
        try:
            blob = _client().bucket(_bucket_name()).blob(f"{WIKI_GCS_PREFIX}/{page_path}")
            blob.upload_from_string(content, content_type="text/markdown; charset=utf-8")
            logger.debug(f"WIKI | written gs://{_bucket_name()}/{WIKI_GCS_PREFIX}/{page_path}")
        except Exception as e:
            logger.error(f"WIKI | GCS write failed for {page_path}: {e}")
            raise
        _page_cache[page_path] = (time.time(), content)
        _list_cache.clear()  # a new page may have appeared — invalidate listing cache
    else:
        full_path = LOCAL_WIKI_DIR / page_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    return page_path


def list_wiki(prefix: str = "") -> list[str]:
    """Return sorted list of .md page paths relative to the wiki root."""
    if _gcs_enabled():
        cached = _list_cache.get(prefix)
        if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1]
        try:
            gcs_prefix = f"{WIKI_GCS_PREFIX}/{prefix}" if prefix else f"{WIKI_GCS_PREFIX}/"
            pages = []
            for blob in _client().list_blobs(_bucket_name(), prefix=gcs_prefix):
                if blob.name.endswith(".md"):
                    rel = blob.name[len(WIKI_GCS_PREFIX) + 1:]
                    pages.append(rel)
            pages = sorted(pages)
            _list_cache[prefix] = (time.time(), pages)
            return pages
        except Exception as e:
            logger.warning(f"WIKI | GCS list failed: {e}")
            return cached[1] if cached else []
    search_dir = LOCAL_WIKI_DIR / prefix if prefix else LOCAL_WIKI_DIR
    if not search_dir.exists():
        return []
    return sorted(
        str(p.relative_to(LOCAL_WIKI_DIR))
        for p in search_dir.rglob("*.md")
        if ".ipynb_checkpoints" not in p.parts
    )


def search_wiki(query: str, prefix: str = "") -> list[dict]:
    """Full-text search across wiki pages. Returns [{path, snippet}] up to 20 matches.

    A query with no literal match anywhere in the wiki (common — the Q&A agent
    sometimes searches for things that simply aren't in any page, e.g. transient
    stock-price commentary) can't early-exit on the 20-match cap, so it always
    has to read every page. In production that's hundreds of GCS round-trips;
    done serially this took minutes and looked indistinguishable from a hang.
    Read them concurrently instead."""
    query_lower = query.lower()
    results: list[dict] = []

    if _gcs_enabled():
        # Route through list_wiki()/read_wiki() instead of raw blob calls — both
        # are cached, so a repeated search (or a search after a recent read/list)
        # mostly hits memory instead of GCS.
        # 40 workers measured fastest against the real bucket (~11s for a full
        # 2768-page scan) — higher counts (e.g. 100) were *slower*, likely from
        # contention on the GCS client's underlying connection pool.
        paths = list_wiki(prefix)
        with ThreadPoolExecutor(max_workers=40) as pool:
            contents = pool.map(read_wiki, paths)
        for rel, content in zip(paths, contents):
            if not content or query_lower not in content.lower():
                continue
            _append_snippet(results, rel, content, query_lower)
            if len(results) >= 20:
                break
        return results

    # local fallback
    search_dir = LOCAL_WIKI_DIR / prefix if prefix else LOCAL_WIKI_DIR
    if not search_dir.exists():
        return []
    for p in sorted(search_dir.rglob("*.md")):
        if ".ipynb_checkpoints" in p.parts:
            continue
        try:
            content = p.read_text()
        except Exception:
            continue
        if query_lower not in content.lower():
            continue
        rel = str(p.relative_to(LOCAL_WIKI_DIR))
        _append_snippet(results, rel, content, query_lower)
        if len(results) >= 20:
            break
    return results


def _append_snippet(results: list[dict], rel: str, content: str, query_lower: str) -> None:
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if query_lower in line.lower():
            start = max(0, i - 1)
            end = min(len(lines), i + 3)
            snippet = "\n".join(lines[start:end]).strip()
            results.append({"path": rel, "snippet": snippet[:400]})
            return


# ── generic append-only CSV log (GCS-backed when GCS_MODE=true) ──────────────
# Backs every canonical, never-reproduced-by-the-LLM section across page types:
# company events/earnings, drug timeline/clinical evidence, indication events.
# Each is a flat per-entity log, deduped by file_path so a retried/replayed
# source document never produces a duplicate row — full-rewrite on append since
# these logs stay small (one row per source document touching that entity).

def _read_csv_log(gcs_prefix: str, local_dir: Path, slug: str) -> list[dict]:
    import csv
    import io

    if _gcs_enabled():
        from google.api_core.exceptions import NotFound
        try:
            blob = _client().bucket(_bucket_name()).blob(f"{gcs_prefix}/{slug}.csv")
            content = blob.download_as_text(encoding="utf-8")
        except NotFound:
            return []
        except Exception as e:
            logger.warning(f"CSV_LOG | GCS read failed for {gcs_prefix}/{slug}: {e}")
            return []
    else:
        path = local_dir / f"{slug}.csv"
        if not path.exists():
            return []
        content = path.read_text()

    if not content.strip():
        return []
    return list(csv.DictReader(io.StringIO(content)))


def _append_csv_log(
    gcs_prefix: str, local_dir: Path, slug: str, fields: list[str], new_rows: list[dict],
) -> None:
    import csv
    import io

    existing = _read_csv_log(gcs_prefix, local_dir, slug)
    seen_files = {r.get("file_path") for r in existing}
    to_add = [r for r in new_rows if r.get("file_path") not in seen_files]
    if not to_add:
        return

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(existing + to_add)
    content = buf.getvalue()

    if _gcs_enabled():
        try:
            blob = _client().bucket(_bucket_name()).blob(f"{gcs_prefix}/{slug}.csv")
            blob.upload_from_string(content, content_type="text/csv; charset=utf-8")
        except Exception as e:
            logger.error(f"CSV_LOG | GCS write failed for {gcs_prefix}/{slug}: {e}")
            raise
    else:
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / f"{slug}.csv").write_text(content)


def read_company_events(slug: str) -> list[dict]:
    """Read the canonical event log for one company. Returns [] if not found."""
    return _read_csv_log(EVENTS_CSV_PREFIX, LOCAL_EVENTS_DIR, slug)


def append_company_events(slug: str, new_rows: list[dict]) -> None:
    """Append rows to a company's canonical event log (see _append_csv_log)."""
    _append_csv_log(EVENTS_CSV_PREFIX, LOCAL_EVENTS_DIR, slug, EVENT_CSV_FIELDS, new_rows)


def read_company_earnings(slug: str) -> list[dict]:
    """Read the canonical earnings-intelligence log for one company. Returns [] if not found."""
    return _read_csv_log(EARNINGS_CSV_PREFIX, LOCAL_EARNINGS_DIR, slug)


def append_company_earnings(slug: str, new_rows: list[dict]) -> None:
    """Append rows to a company's canonical earnings-intelligence log (see _append_csv_log)."""
    _append_csv_log(EARNINGS_CSV_PREFIX, LOCAL_EARNINGS_DIR, slug, EARNINGS_CSV_FIELDS, new_rows)


def read_drug_timeline(inn: str) -> list[dict]:
    """Read the canonical timeline log for one drug. Returns [] if not found."""
    return _read_csv_log(DRUG_TIMELINE_CSV_PREFIX, LOCAL_DRUG_TIMELINE_DIR, inn)


def append_drug_timeline(inn: str, new_rows: list[dict]) -> None:
    """Append rows to a drug's canonical timeline log (see _append_csv_log)."""
    _append_csv_log(DRUG_TIMELINE_CSV_PREFIX, LOCAL_DRUG_TIMELINE_DIR, inn, DRUG_TIMELINE_CSV_FIELDS, new_rows)


def read_drug_clinical_evidence(inn: str) -> list[dict]:
    """Read the canonical clinical-evidence log for one drug. Returns [] if not found."""
    return _read_csv_log(DRUG_EVIDENCE_CSV_PREFIX, LOCAL_DRUG_EVIDENCE_DIR, inn)


def append_drug_clinical_evidence(inn: str, new_rows: list[dict]) -> None:
    """Append rows to a drug's canonical clinical-evidence log (see _append_csv_log)."""
    _append_csv_log(DRUG_EVIDENCE_CSV_PREFIX, LOCAL_DRUG_EVIDENCE_DIR, inn, DRUG_EVIDENCE_CSV_FIELDS, new_rows)


def read_drug_management_sentiment(inn: str) -> list[dict]:
    """Read the canonical management-sentiment log for one drug. Returns [] if not found."""
    return _read_csv_log(DRUG_SENTIMENT_CSV_PREFIX, LOCAL_DRUG_SENTIMENT_DIR, inn)


def append_drug_management_sentiment(inn: str, new_rows: list[dict]) -> None:
    """Append rows to a drug's canonical management-sentiment log (see _append_csv_log)."""
    _append_csv_log(DRUG_SENTIMENT_CSV_PREFIX, LOCAL_DRUG_SENTIMENT_DIR, inn, DRUG_SENTIMENT_CSV_FIELDS, new_rows)


def read_indication_events(slug: str) -> list[dict]:
    """Read the canonical event log for one indication hub. Returns [] if not found."""
    return _read_csv_log(INDICATION_EVENTS_CSV_PREFIX, LOCAL_INDICATION_EVENTS_DIR, slug)


def append_indication_events(slug: str, new_rows: list[dict]) -> None:
    """Append rows to an indication hub's canonical event log (see _append_csv_log)."""
    _append_csv_log(INDICATION_EVENTS_CSV_PREFIX, LOCAL_INDICATION_EVENTS_DIR, slug, INDICATION_EVENTS_CSV_FIELDS, new_rows)


# ── state file (GCS-backed when GCS_MODE=true) ───────────────────────────────

def load_state() -> dict:
    """Load processing state JSON. Returns empty state if not found."""
    empty: dict = {"processed_files": {}, "processed_nct_ids": {}, "last_lint_run": None}
    if _gcs_enabled():
        try:
            import json
            blob = _client().bucket(_bucket_name()).blob(STATE_GCS_KEY)
            if blob.exists():
                return json.loads(blob.download_as_text(encoding="utf-8"))
            return empty
        except Exception as e:
            logger.warning(f"STATE | GCS load failed: {e}")
            return empty
    return empty  # caller handles local fallback via STATE_FILE


def save_state(state: dict) -> None:
    """Persist processing state JSON to GCS. No-op when GCS_MODE is not set."""
    if not _gcs_enabled():
        return  # caller handles local write via STATE_FILE
    import json
    try:
        blob = _client().bucket(_bucket_name()).blob(STATE_GCS_KEY)
        blob.upload_from_string(
            json.dumps(state, indent=2),
            content_type="application/json; charset=utf-8",
        )
        logger.debug(f"STATE | saved to gs://{_bucket_name()}/{STATE_GCS_KEY}")
    except Exception as e:
        logger.error(f"STATE | GCS save failed: {e}")
        raise


# ── run lock (prevents overlapping pipeline runs, e.g. Scheduler firing while a
#    manual/backlog run is still in progress) ─────────────────────────────────

def acquire_lock() -> bool:
    """Try to acquire the pipeline run lock. Returns True if acquired.
    Returns False if another run already holds a fresh (non-stale) lock.
    No-op (always succeeds) when GCS_MODE is not set — local dev has no
    concurrent-run risk worth guarding against."""
    if not _gcs_enabled():
        return True
    import json
    import time
    blob = _client().bucket(_bucket_name()).blob(LOCK_GCS_KEY)
    if blob.exists():
        try:
            data = json.loads(blob.download_as_text(encoding="utf-8"))
            age = time.time() - data.get("started_at", 0)
            if age < LOCK_STALE_AFTER_SECONDS:
                logger.warning(f"LOCK | held by another run, age {age:.0f}s — refusing to start")
                return False
            logger.warning(f"LOCK | found stale lock (age {age:.0f}s) — overriding")
        except Exception as e:
            logger.warning(f"LOCK | unreadable lock file, overriding: {e}")
    blob.upload_from_string(
        json.dumps({"started_at": time.time()}),
        content_type="application/json; charset=utf-8",
    )
    return True


def release_lock() -> None:
    """Release the pipeline run lock. No-op when GCS_MODE is not set."""
    if not _gcs_enabled():
        return
    try:
        blob = _client().bucket(_bucket_name()).blob(LOCK_GCS_KEY)
        if blob.exists():
            blob.delete()
    except Exception as e:
        logger.warning(f"LOCK | release failed: {e}")
