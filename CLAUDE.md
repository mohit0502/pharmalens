# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

**Compiler pipeline (local dev):**
```bash
python agents/orchestrator.py                   # process all new files from GCS + local raw/
python agents/orchestrator.py --limit 10        # round-robin subset across doc types (dev)
python agents/orchestrator.py retry-timeouts    # re-queue files that timed out previously
python agents/orchestrator.py retry-failed      # re-queue all failed files (429s, 499s, timeouts)
python agents/orchestrator.py schedule          # in-process daily loop — LOCAL DEV ONLY, see Deployment
```

**State inspection:**
```bash
python agents/state.py summary       # files processed, by doc type + status
python agents/state.py unprocessed   # list files not yet compiled
python agents/state.py ncts          # all NCT IDs tracked + sponsor
python agents/state.py export        # write agents/processing_log.csv
python agents/state.py reset         # wipe state — forces full reprocess next run
```

**Data pipeline scripts (ad-hoc / local use):**
```bash
python scripts/fetch_trial_pubmed.py --dry-run           # preview NCT-based PubMed fetch
python scripts/fetch_trial_pubmed.py --company novo-nordisk
python scripts/run_edgar.py                              # process only local edgar files
python scripts/build_reference.py                        # discover new drugs → drugs_proposed.json
python scripts/build_reference.py --merge                # promote proposed → reference/drugs.json
```

**API backend:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend && npm install && npm run dev    # :5173, proxies /api → :8000 via Vite
cd frontend && npm run build
```

**DataPipelines (containerised GCP jobs — run via Cloud Scheduler, not locally):**
```bash
# ctgov-job: seed a company
GCS_BUCKET_NAME=pharmalens-raw SEED_MODE=true python DataPipelines/ctgov-job/main.py

# pubmed-job: single drug, local test
python DataPipelines/pubmed-job/main.py --drug semaglutide --local-out /tmp/pubmed-test

# pubmed-job: dry run
python DataPipelines/pubmed-job/main.py --dry-run --days-back 30
```

**Compiler job (Cloud Run, production):**
```bash
gcloud run jobs execute pharmalens-compiler --region us-central1   # manual trigger
gcloud builds submit --config DataPipelines/compiler-job/cloudbuild.yaml .   # rebuild + push image
gcloud run jobs update pharmalens-compiler --image us-central1-docker.pkg.dev/project-fba8fa53-e9e5-49d2-8dc/pharmalens/compiler-job:latest --region us-central1   # redeploy after rebuild
```

---

## Repository layout

```
pharmalens/
├── agents/             ← compiler pipeline (LLM chain + state + GCS adapters)
│   ├── compiler.py     ← 3-step LLM chain (extract → validate → write)
│   ├── orchestrator.py ← pipeline driver: cache builder + per-file loop + flush + run-lock
│   ├── state.py        ← processing_state.json tracker + NCT ID registry + index.md updater
│   ├── gcs.py          ← raw INPUT adapter (gs://pharmalens-raw) — transparent GCS/local
│   ├── wiki_gcs.py      ← wiki/state OUTPUT adapter (gs://pharmalens-wiki) — transparent GCS/local + run-lock
│   ├── lint.py          ← weekly wiki health check (Gemini 2.5 Pro) — reads/writes via wiki_gcs
│   ├── cost.py          ← token usage ledger (Gemini 2.5 Flash pricing)
│   └── prompts/
│       ├── CLAUDE.md               ← LLM system schema (NOT the project CLAUDE.md)
│       ├── compiler_{type}.txt     ← extraction prompts per doc type
│       └── templates/{type}.md     ← wiki page formatting templates
├── DataPipelines/      ← containerised GCP ingestion jobs
│   ├── ctgov-job/      ← CT.gov → GCS (monthly seed + 30-day delta) — DE-owned (Sarthak)
│   ├── pubmed-job/     ← PubMed → GCS (weekly, Clinical Trials + Meta-Analyses) — DE-owned
│   ├── edgar-8K-job/   ← SEC 8-K filings → GCS (event-driven, 11 US companies) — DE-owned
│   ├── edgar-10Q-job/  ← SEC 10-Q filings → GCS (quarterly) — DE-owned
│   └── compiler-job/   ← Cloud Run Job wrapper for agents/orchestrator.py — ours
│       ├── Dockerfile      ← copies agents/ + reference/ only (not wiki/ or raw/ — those are GCS)
│       ├── entrypoint.py   ← calls run_daily_pipeline() once, then exits
│       └── cloudbuild.yaml ← explicit docker build config (Dockerfile lives in a subdirectory)
├── api/
│   ├── main.py         ← FastAPI backend (routes + SSE Q&A stream)
│   ├── agent.py        ← Gemini agentic Q&A loop with 5 tools
│   ├── tools.py         ← read_wiki_page, list_wiki_pages, search_wiki, get_stock_price, parse_company_trials
│   └── bootstrap.py    ← writes GOOGLE_CREDENTIALS_JSON to temp file on Render (Render only — Cloud Run uses an attached service account instead)
├── frontend/src/       ← React 18 + Vite + D3 (no TypeScript)
├── reference/
│   ├── companies.json  ← 20 companies: ticker, EDGAR CIK, aliases, active indications
│   ├── drugs.json      ← 27 drugs by INN: brands, aliases, company, indication slugs
│   ├── indications.json← 7 indications: ICD-10, MeSH, aliases, ctgov_condition
│   └── drugs_proposed.json ← auto-generated by build_reference.py; needs human review
├── raw/                ← mostly .gitkeep; real data lives on GCS (gs://pharmalens-raw)
├── wiki/               ← LOCAL DEV ONLY compiled markdown pages. In production (GCS_MODE=true)
│                           the wiki lives on GCS at gs://pharmalens-wiki/wiki/ instead — see Deployment.
├── data/               ← LOCAL DEV ONLY CSV state caches (company_events, company_earnings,
│                           drug_timeline, drug_clinical_evidence, drug_management_sentiment,
│                           indication_events) — gitignored. In production these live on GCS
│                           under gs://pharmalens-wiki/state/ instead — see wiki_gcs.py.
└── credentials/        ← gitignored GCP service-account key files (local dev only)
```

---

## Architecture

### Data flow (production, GCS_MODE=true)

```
DataPipelines/ (GCP Cloud Scheduler jobs, DE-owned)
    ↓ uploads to
GCS bucket: pharmalens-raw  (raw/ — input data, owned by the DE's project)
    ↓ read by
agents/gcs.py → list_raw_blobs() + read_blob()
    ↓ fed into
agents/orchestrator.py (Cloud Run Job, triggered daily by Cloud Scheduler)
    ↓ per file
agents/compiler.py → 3-step chain (extract → validate → write)
    ↓ produces, via agents/wiki_gcs.py
GCS bucket: pharmalens-wiki  (wiki/ + state/ — output, owned by our project)
    ↓ read by
api/main.py (FastAPI, on Render) + api/agent.py (Q&A)
    ↓ served to
frontend/src/ (React dashboard + AI chat, on Vercel)
```

**Why two buckets:** `pharmalens-raw` (input) is owned and maintained by the data-engineering side; `pharmalens-wiki` (output: compiled wiki pages + `processing_state.json`) is owned by us. Keeping them separate means a mistake or policy change on either side can't take down the other — raw data is cheap to re-fetch, the compiled wiki represents real LLM compute cost and shouldn't share a blast radius with someone else's bucket.

**Local dev (GCS_MODE unset):** both adapters fall back to the local filesystem — `agents/gcs.py` reads `raw/`, `agents/wiki_gcs.py` reads/writes `wiki/`, `agents/processing_state.json`, and the per-entity CSV caches under `data/` (company events/earnings, drug timeline/clinical-evidence/management-sentiment, indication events). No code changes needed to switch between local and cloud.

### GCS adapters

**`agents/gcs.py`** — raw INPUT adapter. `list_raw_blobs()` returns a unified `list[Path]` — local `raw/` files first, then GCS blobs (bucket from `GCS_BUCKET` env var, default `pharmalens-raw`) not already present locally. The compiler never knows which source a file came from. `read_blob()` reads local if the path exists, else downloads from GCS. Local copy always wins on conflict.

**`agents/wiki_gcs.py`** — wiki/state OUTPUT adapter. When `GCS_MODE=true`, `read_wiki()`/`write_wiki()`/`list_wiki()`/`search_wiki()` operate against the bucket named by `WIKI_BUCKET` (default `pharmalens-wiki`), at `wiki/<page_path>` and `state/processing_state.json`. When `GCS_MODE` is unset, all four fall back to the local `wiki/` directory and `agents/processing_state.json`. This module also owns the **run lock** (`acquire_lock()`/`release_lock()`, backed by a `state/compiler.lock` GCS object) — `run_daily_pipeline()` calls this first and refuses to start if another run already holds a fresh lock, preventing Cloud Scheduler from firing a second execution on top of a still-running backlog run. A lock older than 20 hours is treated as stale (the process that held it must have died) and overridden.

### Credentials

**Local:** `.env` sets `GOOGLE_APPLICATION_CREDENTIALS=credentials/project-fba8fa53-...json`. Every Python entry point calls `load_dotenv()` before creating any Google client. `gsutil`/`gcloud` in the shell won't see this `.env` — either export manually, run `gcloud auth activate-service-account --key-file=...` for that shell session, or use the Python SDK directly.

**Render (API, production):** `api/bootstrap.py` reads `GOOGLE_CREDENTIALS_JSON` (full service-account JSON string set in the Render dashboard) and writes it to a temp file before any `genai.Client()` is created.

**Cloud Run Job (compiler, production):** no JSON-paste needed — the job runs as the `pharmalens-tracker@project-fba8fa53-e9e5-49d2-8dc.iam.gserviceaccount.com` service account, attached directly via `--service-account` at job creation. Authentication is automatic (ADC), same credential used for local dev's `.env`.

**Important:** `load_dotenv()` must precede `genai.Client()` in every entry point — the client reads env vars at construction time.

### 3-step compiler chain (`agents/compiler.py`)

**Step 1 — Extract:**
- `ctgov` files: pure Python (`extract_ctgov_python()`) — uses alias maps to resolve company, drugs, indications from PascalCase CT.gov fields (`NCTId`, `HasResults`, `OverallStatus`, etc.). Saves one LLM call per file.
- `pubmed` files: one LLM call, plus `pmid` is injected deterministically in Python afterward (read straight from the raw JSON) — the LLM is never asked to echo it back, since that field was previously silently dropped.
- All other types (`edgar_8k`, `edgar_10q`, `genepool`): one LLM call with a per-doc-type extraction prompt, using a Gemini context cache.

**Step 2 — Collect (pure Python):**
Validates entities against `reference/*.json`. Routes signals into four batch buffers (company, trial, drug, indication). Event pages are the exception — each has a unique slug and is written immediately per file.

**Step 3 — Flush:**
After the per-file loop, `flush_buffered_pages()` fires all entity page writes in parallel (up to 20 threads, one LLM call per entity, no pool-level timeout — each call is individually bounded by the 300s HTTP client timeout instead). Each entity receives ALL signals accumulated across the batch run. Files are only marked processed in state after a successful flush — but `mark_file_processed()` itself writes incrementally (one GCS round-trip per file, not one atomic batch at the end), so a run cancelled mid-flush still durably persists everything completed up to that point.

**Gemini context caching:**
`orchestrator.py` builds caches once per pipeline run — one per doc type present in the queue, one per wiki page type (always 5). Each cache contains system prompt + extraction or template content. Saves ~70% of token costs. Cache TTL is 3 hours (`10800s`) — enough to cover one daily compiler run with margin, short enough that a stale cache from the previous day isn't still being billed for storage by the time the next day's run creates a fresh one. (Previously 30 hours/`108000s`; that overlap was identified as the dominant driver of Vertex AI's "Input Text Caching Storage" cost in a billing review.) Model: `gemini-2.5-flash` for compiler + Q&A, `gemini-2.5-pro` for lint.

### State tracking (`agents/state.py`)

`processing_state.json` (local file in dev, `gs://pharmalens-wiki/state/processing_state.json` in production) tracks:
- `processed_files`: path → `{processed_at, doc_type, company, drug, status, content_hash}`
- `processed_nct_ids`: NCT ID → `{first_seen, primary_sponsor}` — entries are never removed, even if the source file is later reprocessed, so re-running a changed trial still correctly resolves to `"update"` not `"new"`.
- `last_lint_run`: ISO timestamp

**Hash-based delta:** If a file's MD5 matches its stored hash, it's skipped — even if re-encountered in a new date folder. This handles ctgov cross-folder dedup (same trial, new `YYYY-MM-DD/` directory). Note: a prompt/template fix alone does **not** trigger reprocessing — only a changed raw source file hash does. To force a re-run after a prompt change, use `reset_failed_files()`/`reset_timeout_files()` patterns or manually clear specific entries from state.

**NCT action:** `get_nct_action()` returns `"new"` or `"update"` based on whether the NCT ID has been seen before. Step 3 uses this to add vs. update a trial section; the prompt explicitly lists which NCT IDs are new vs. update and instructs "no duplicates."

**`index.md` is actively maintained** by `update_index_py()`, called after every flush — it is not regenerated from scratch, only appended to (skips entries already linked). Separately, the Q&A agent's wiki navigation uses `build_wiki_map()`, which scans the wiki at query time rather than reading `index.md` — the two serve different purposes (a literal page vs. live agent context).

### Wiki structure

**Trial pages** (`wiki/trials/{company}.md`) are **per-company, not per NCT ID.** Each file contains multiple YAML frontmatter blocks separated by `---` — one block per trial. `api/main.py` / `api/tools.py:parse_company_trials()` splits on `---` and calls `yaml.safe_load()` on each block. **Do not use `---` as a markdown horizontal rule inside trial pages.** Trial frontmatter includes `result_summary` (a plain-English one-line summary, no stats/jargon) alongside the denser verbatim `primary_result_value` — `result_summary` must be real YAML `null` (not the string `"None"`) when no result exists; same rule for `primary_result_value`.

**Drug/company/indication pages** have a single YAML frontmatter block at the top. Company pages' "Recent events" table has a `Source` column — `PMID:{pmid}` for research-type rows when available, blank otherwise; the Signal column must be a plain sentiment word (`Bullish`/`Neutral`/`Bearish`/etc.), never a wikilink, and the Event text must never embed the sentiment word itself.

`agents/prompts/CLAUDE.md` is the LLM compiler's schema and rules, injected into every system prompt. Changes there affect all compiler behaviour.

### API (`api/main.py`)

Key routes:
- `GET /api/company/{slug}/trials` — calls `api/tools.py:parse_company_trials()`, which parses the per-company trial wiki and computes stats. `_count_concluded_trials()` has specific logic: terminated/withdrawn always count; completed/active-not-recruiting only count if `primary_completion_date` falls within the lookback window.
- `POST /api/ask` — streams SSE events: `tool_call`, `tool_result`, `text`, `done`. Q&A agent (`api/agent.py`) has **7 tools**: `read_wiki_page`, `list_wiki_pages`, `search_wiki`, `get_stock_price`, `get_stock_history`, `get_company_news`, `get_company_trials`. `get_company_trials` exists because reading `trials/{company}.md` directly gets truncated (tool results are capped at 30,000 chars) for companies with many trials — it returns the same data as compact, pre-sorted (newest-first) JSON instead. `get_company_news` and `get_stock_history` are for time-sensitive questions the static wiki can't answer.
- `GET /api/company/{slug}/news` — proxies `api/news.py:get_company_news()` for the frontend news grid.
- `GET /api/news` — proxies `api/news.py:get_relevant_articles()` for the sidebar feed.
- `GET /api/article` — fetches and parses a single BioSpace article by URL; used for article-aware Q&A (the article body is injected into the agent's user message directly, not fetched by a tool).

**`api/news.py`** — news read + cache layer (no LLM, no GCS). Fetches BioSpace's news sitemap to discover articles mentioning tracked companies; fetches individual article pages on demand. Also combines BioSpace + Yahoo Finance news for `get_company_news()`. Three in-process caches: articles list (15 min TTL), individual article bodies (1 hr), per-company combined feed (15 min). All stock tools in `api/tools.py` accept a company slug and resolve to a ticker internally via `resolve_ticker()` — callers never pass tickers directly.

### Frontend (`frontend/src/`)

React 18 + Vite, no TypeScript. `App.jsx` holds global state: exactly one of `activeIndication` or `activeCompany` is set at a time. Vite proxy forwards `/api` → `:8000` in dev (no CORS issues). In production set `VITE_API_URL` to the Render backend URL.

`frontend/src/parseWiki.js` parses raw wiki markdown into structured sections consumed by components. `TrialsPanel.jsx` uses D3 for the stacked phase bar chart, renders a clickable NCT ID link to clinicaltrials.gov on each trial result card, and a clickable `PMID:xxxxx` link to PubMed on each research event row. `StockChart.jsx` is a D3 candlestick chart with period selector.

---

## DataPipelines (GCP jobs — `DataPipelines/`)

`ctgov-job`, `pubmed-job`, `edgar-8K-job`, `edgar-10Q-job` are written by Sarthak Mistry (DE teammate), containerised with Docker, deployed as Cloud Run jobs triggered by Cloud Scheduler — all writing to `gs://pharmalens-raw/raw/`. `compiler-job` (ours) is the Cloud Run Job wrapper around `agents/orchestrator.py` — see Deployment below.

### `ctgov-job/main.py`

- **Output path:** `gs://pharmalens-raw/raw/ctgov/{company-slug}/{YYYY-MM-DD}/{NCT-ID}.json`
- **JSON schema:** PascalCase keys matching CT.gov API — `NCTId`, `HasResults`, `OverallStatus`, `Phase`, `LeadSponsorName`, `InterventionName`, etc.
- **Results enrichment:** When `HasResults: true`, fetches the `resultsSection` from CT.gov and attaches it to the JSON as `trial["resultsSection"]`.
- **Modes:** `SEED_MODE=true` → 1-year lookback; default → 30-day delta.
- **Append-only:** Skips blobs that already exist in GCS.
- **Filters:** Phase 2/3/4 only; statuses: RECRUITING, ACTIVE\_NOT\_RECRUITING, COMPLETED, TERMINATED.

### `pubmed-job/main.py`

- **Output path:** `gs://pharmalens-raw/raw/pubmed/{drug-slug}/{YYYY-MM-DD}/abstract-{pmid}.json`
- **JSON schema:** lowercase snake\_case — `pmid`, `doi`, `pubmed_date`, `title`, `journal`, `abstract` (dict of labelled sections), `mesh_major_topics`, `first_author` (dict with `name` + `affiliation`).
- **Scope:** Clinical Trials + Meta-Analyses only; default 7-day lookback.
- **Rate limits:** 3 req/s without `NCBI_API_KEY`, 10 req/s with it.
- **Tracked drugs:** 23 drugs across GLP-1, oncology, cardiovascular, CNS, rare disease.
- Supports `--local-out DIR` for local testing without GCP credentials.

### `edgar-8K-job/main.py` / `edgar-10Q-job/main.py`

- **Output path:** `gs://pharmalens-raw/raw/edgar/{company-slug}/8K/{YYYY-MM-DD}-8K-{accession}.txt`
- **Scope:** 11 US-listed companies (non-US companies like Roche, Novartis, AstraZeneca have no SEC filings).
- Uses `edgartools` library. Strips HTML with BeautifulSoup. Extracts press release exhibit (EX-99.1/EX-99.2) when present.

---

## GCS bucket state

**`pharmalens-raw`** · Project: `project-fba8fa53-e9e5-49d2-8dc` · Region: `us-central1` · owned/written by the DE side, read-only for us except where explicitly granted.
- `raw/` — ctgov, pubmed, edgar input data (~7,250+ files and growing; DataPipelines jobs append continuously).
- `logs/` — `ctgov_enriched.md` / `ctgov_no_pubmed.md`, uploaded by the ctgov-job.

**`pharmalens-wiki`** · Project: `project-fba8fa53-e9e5-49d2-8dc` · Region: `us-central1` · owned and written by us.
- `wiki/` — all compiled markdown pages (companies, drugs, trials, events, indications, `index.md`).
- `state/processing_state.json` — the processing state tracker.
- `state/compiler.lock` — run-lock object (see GCS adapters above); normally absent except during an active run.

Both buckets are accessed by the `pharmalens-tracker@project-fba8fa53-e9e5-49d2-8dc.iam.gserviceaccount.com` service account — `roles/storage.objectAdmin` on `pharmalens-wiki` (full control, since we own it) and a narrower grant on `pharmalens-raw` (object read + write to `raw/`, granted cross-bucket by the DE).

**Local `raw/`** — only `.gitkeep` placeholders. All substantive raw data lives on GCS. **Local `wiki/`** is dev-only; production wiki is the GCS copy above.

---

## Schema notes

**ctgov JSON keys are PascalCase** (`HasResults`, `NCTId`, `OverallStatus`). The compiler's `preprocess_ctgov()` uses these exact names — `data.get("HasResults")`. Any script checking for `has_results` (lowercase) will get zero matches. This burned us once when checking GCS coverage.

**pubmed `first_author`** is a dict `{"name": "...", "affiliation": "..."}` in the DataPipelines version. The older `scripts/fetch_trial_pubmed.py` wrote it as a plain string — schema mismatch if mixing sources; compiler code that does `.get()` on it will crash on the older format.

**YAML parses unquoted dates as `datetime.date`, not `str`.** Any code reading trial frontmatter (e.g. `primary_completion_date`) must coerce to `str` before sorting or JSON-serializing — mixed quoted/unquoted entries across the wiki otherwise raise `TypeError` on comparison and `json.dumps()` failures. `api/tools.py:parse_company_trials()` does this coercion explicitly.

**Adding a new doc type** requires: a prompt file at `agents/prompts/compiler_{type}.txt`, and a new branch in `classify_document()` in `compiler.py`.

**Adding a company/drug/indication** means editing `reference/*.json`. The compiler enforces the closed-world assumption and will not create wiki pages for entities absent from reference data — but this is currently only enforced for drugs/companies, not indications: a trial whose true condition isn't one of the 7 tracked indication slugs can still get force-fit into the wrong slug rather than left untagged. Known gap, not yet fixed.

---

## Deployment

### API (Render)

Single web service (`render.yaml`), `uvicorn api.main:app`. Env vars: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI=True`, `GCS_MODE=true`, `WIKI_BUCKET=pharmalens-wiki`, and `GOOGLE_CREDENTIALS_JSON` (paste the `pharmalens-tracker` service-account JSON in the Render dashboard — `sync: false` in `render.yaml`, must be set manually). Note `render.yaml` only takes effect on a Blueprint sync; setting env vars directly in the Render dashboard is the reliable path and takes effect immediately on next deploy.

### Compiler pipeline (Cloud Run Job + Cloud Scheduler)

The daily compiler run is **not** the `orchestrator.py schedule` command — that in-process loop is local-dev-only. Production uses:

- **`DataPipelines/compiler-job/`** — Dockerfile copies only `agents/` + `reference/` (not `wiki/` or `raw/`, both GCS-backed) and `entrypoint.py`, which calls `run_daily_pipeline()` once and exits. Built via `gcloud builds submit --config DataPipelines/compiler-job/cloudbuild.yaml .` (a Cloud Build config, not `--tag`, is required because the Dockerfile lives in a subdirectory) and pushed to Artifact Registry (`us-central1-docker.pkg.dev/project-fba8fa53-e9e5-49d2-8dc/pharmalens/compiler-job`).
- **Cloud Run Job** `pharmalens-compiler` (region `us-central1`) — runs as `pharmalens-tracker`, 2 CPU / 2Gi memory, task timeout 1 day. Env vars: `GCS_MODE=true`, `GCS_BUCKET=pharmalens-raw` (raw input), `WIKI_BUCKET=pharmalens-wiki` (compiled output — these are deliberately different buckets, see Architecture above), `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI=True`.
- **Cloud Scheduler** job `pharmalens-compiler-daily` — cron `0 7 * * *`, timezone `America/Chicago`, hits the Cloud Run Jobs REST API (`POST .../namespaces/{project}/jobs/pharmalens-compiler:run`) via OAuth as the default compute service account (granted `roles/run.invoker`).
- **Concurrency safety:** the run-lock in `agents/wiki_gcs.py` (see GCS adapters) means if the Scheduler fires while a manual/backlog run is still in progress, the new execution detects the lock and exits immediately rather than running a second compiler in parallel against the same state file.
- After rebuilding the image, **always** run `gcloud run jobs update pharmalens-compiler --image ...:latest --region us-central1` — pushing a new image tag does not retroactively update an existing job definition.

**Google GenAI SDK:** `google-genai` (the unified SDK). Do not use the deprecated `google-generativeai` or `google-cloud-aiplatform` packages.
