# orchestrator.py — pure Python traffic cop + cache builder
# Coordinates pipeline execution: builds caches, finds new files, invokes compiler, runs lint.
import os
import sys
import re
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime
import schedule
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # must be before genai.Client()

client = genai.Client()

FLASH_MODEL = "gemini-2.5-flash"
FILE_TIMEOUT_SECONDS = int(os.getenv("FILE_TIMEOUT_SECONDS", "180"))       # 3 min default
EDGAR_TIMEOUT_SECONDS = int(os.getenv("EDGAR_TIMEOUT_SECONDS", "300"))     # 5 min for 8K/10Q
EDGAR_DOC_TYPES = {"edgar_8k", "edgar_10q"}

try:
    BASE_DIR = Path(__file__).parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

WIKI_DIR = BASE_DIR / "wiki"
LOG_FILE = WIKI_DIR / "log.md"

from agents.logger import get_logger
from agents.state import (
    get_unprocessed_files,
    mark_file_processed,
    should_run_lint,
    mark_lint_run,
    reset_timeout_files,
    reset_failed_files,
)
from agents.compiler import (
    classify_document,
    get_company_from_path,
    get_drug_from_path,
    build_system_prompt,
    load_prompt,
    load_page_template,
    flush_buffered_pages,
)

logger = get_logger("pharmalens.orchestrator")


# ── cache builders ────────────────────────────────────────────────────────────

def create_cache(system_prompt: str, label: str, ttl: str = "10800s") -> str:
    """Create a single Gemini context cache and return its name."""
    cache = client.caches.create(
        model=FLASH_MODEL,
        config=types.CreateCachedContentConfig(
            system_instruction=system_prompt,
            ttl=ttl,
        ),
    )
    logger.info(
        f"CACHE | '{label}' created: {cache.name} "
        f"({cache.usage_metadata.total_token_count} tokens)"
    )
    return cache.name


def build_extraction_caches(unprocessed: list[Path]) -> dict[str, str]:
    """Create one Gemini cache per doc type present in the unprocessed file list.
    Each cache contains: system prompt + extraction prompt for that doc type.

    Only caches doc types actually needed for this pipeline run — no wasted
    cache slots for doc types with no files queued.
    Skips any doc type whose prompt file is missing (ValueError from load_prompt).

    Returns dict keyed by doc_type string → cache name.
    """
    doc_types_needed  = {classify_document(f) for f in unprocessed}
    extraction_caches = {}
    system_prompt     = build_system_prompt()

    for doc_type in doc_types_needed:
        if doc_type == "unknown":
            continue
        try:
            extraction_caches[doc_type] = create_cache(
                system_prompt + load_prompt(doc_type),
                label=f"extraction_{doc_type}",
            )
        except ValueError as e:
            logger.warning(f"CACHE | Skipping extraction cache for '{doc_type}': {e}")

    return extraction_caches


def build_template_caches() -> dict[str, str]:
    """Create one Gemini cache per wiki page type.
    Each cache contains: system prompt + page formatting template.

    All 5 page types are always cached regardless of which pages will be
    written this run — the full set is always needed since any doc type
    can trigger any combination of page writes.
    Skips any page type whose template file is missing.

    Returns dict keyed by page_type string → cache name.
    """
    page_types     = ["drug", "company", "trial", "event", "indication_hub"]
    template_caches = {}
    system_prompt  = build_system_prompt()

    for page_type in page_types:
        try:
            template_caches[page_type] = create_cache(
                system_prompt + load_page_template(page_type),
                label=f"template_{page_type}",
            )
        except ValueError as e:
            logger.warning(f"CACHE | Skipping template cache for '{page_type}': {e}")

    return template_caches


# ── log helpers ───────────────────────────────────────────────────────────────

def append_log(file_path: Path, status: str, doc_type: str) -> None:
    """Append a human-readable entry to wiki/log.md (local dev only).
    In GCS_MODE the audit trail lives in Cloud Logging — local file writes are skipped.
    Do not parse this file for logic — use state.py instead.
    """
    from agents.wiki_gcs import _gcs_enabled
    if _gcs_enabled():
        # Cloud Logging captures the full audit trail; no file append needed.
        return
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
    entry = (
        f"\n## [{timestamp}] ingest | {file_path}\n"
        f"doc_type: {doc_type}\n"
        f"status: {status}\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(entry)


# ── main workflows ────────────────────────────────────────────────────────────

def run_compiler_on_file(
    file_path: Path,
    extraction_caches: dict,
    template_caches: dict,
    company_buffer: dict | None = None,
    trial_buffer: dict | None = None,
    drug_buffer: dict | None = None,
    indication_buffer: dict | None = None,
) -> tuple[Path, str, str] | None:
    """Call the compiler agent on a single new file.
    Returns (file_path, doc_type, company) on success so the caller can mark
    it processed only after the flush succeeds. Returns None on timeout/error
    (those are marked immediately since no signals were buffered).
    """
    from agents.compiler import compile_document

    def get_date_from_filename(path: Path) -> str | None:
        """Extract YYYY-MM-DD from filename if present."""
        match = re.search(r"\d{4}-\d{2}-\d{2}", path.stem)
        return match.group(0) if match else None

    doc_type = classify_document(file_path)
    company  = get_company_from_path(file_path)
    drug     = get_drug_from_path(file_path)
    file_date = get_date_from_filename(file_path)

    if doc_type == "unknown":
        logger.warning(f"ORCHESTRATOR | SKIP | Cannot classify: {file_path}")
        mark_file_processed(file_path, "unknown", "skipped_unknown_type")
        append_log(file_path, "skipped_unknown_type", "unknown")
        return None

    context = {
        "doc_type": doc_type,
        "company":  company,
        "drug":     drug,
        "file_date": file_date,
    }

    timeout = EDGAR_TIMEOUT_SECONDS if doc_type in EDGAR_DOC_TYPES else FILE_TIMEOUT_SECONDS
    logger.info(f"ORCHESTRATOR | COMPILE | {file_path.name} → type: {doc_type} | timeout: {timeout}s")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        compile_document, file_path, context, extraction_caches, template_caches,
        company_buffer, trial_buffer, drug_buffer, indication_buffer,
    )
    executor.shutdown(wait=False)
    try:
        future.result(timeout=timeout)
        # Do NOT mark success here — deferred until after flush_buffered_pages() confirms
        # wiki pages were actually written. See run_daily_pipeline().
        logger.info(f"ORCHESTRATOR | DONE | {file_path.name}")
        return (file_path, doc_type, company, drug)
    except concurrent.futures.TimeoutError:
        timeout_msg = f"timeout: exceeded {timeout}s"
        mark_file_processed(file_path, doc_type, timeout_msg, company, drug)
        append_log(file_path, timeout_msg, doc_type)
        logger.warning(f"ORCHESTRATOR | TIMEOUT | {file_path.name} skipped after {timeout}s")
        return None
    except Exception as e:
        error_msg = f"error: {str(e)[:100]}"
        mark_file_processed(file_path, doc_type, error_msg, company, drug)
        append_log(file_path, error_msg, doc_type)
        logger.error(f"ORCHESTRATOR | ERROR | {file_path}: {e}")
        return None


def _pick_subset(files: list[Path], limit: int) -> list[Path]:
    """Round-robin across doc types so a small limit still covers each type."""
    from collections import defaultdict
    buckets: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        buckets[classify_document(f)].append(f)
    result: list[Path] = []
    keys = sorted(buckets)
    i = 0
    while len(result) < limit:
        key = keys[i % len(keys)]
        if buckets[key]:
            result.append(buckets[key].pop(0))
        i += 1
        if all(len(v) == 0 for v in buckets.values()):
            break
    return result


def run_daily_pipeline(limit: int | None = None, file_list: list[Path] | None = None) -> None:
    """Main daily job: build caches, find unprocessed files, compile each one.
    Guarded by a GCS-backed lock so the Cloud Scheduler trigger can never start
    a second run on top of one that's still in progress (e.g. a long backlog
    catch-up run still going when the next day's 07:00 trigger fires)."""
    from agents.wiki_gcs import acquire_lock, release_lock
    if not acquire_lock():
        logger.warning("ORCHESTRATOR | Another pipeline run is already in progress — skipping this run")
        return
    try:
        _run_daily_pipeline(limit, file_list=file_list)
    finally:
        release_lock()


def _run_daily_pipeline(limit: int | None = None, file_list: list[Path] | None = None) -> None:
    logger.info(f"{'=' * 60}")
    logger.info(f"ORCHESTRATOR | Pipeline starting")

    if file_list is not None:
        # Caller has already selected exactly which files to run (e.g. a
        # doc-type-filtered sample) — skip get_unprocessed_files()/_pick_subset's
        # round-robin entirely so the caller's exact selection is preserved.
        unprocessed = file_list
        logger.info(f"ORCHESTRATOR | Explicit file list mode: running {len(unprocessed)} files")
    else:
        # get new files first — needed to know which extraction caches to build
        unprocessed = get_unprocessed_files()
        logger.info(f"ORCHESTRATOR | Found {len(unprocessed)} new files to process")

        if limit is not None:
            unprocessed = _pick_subset(unprocessed, limit)
            logger.info(f"ORCHESTRATOR | Subset mode: running {len(unprocessed)} files (--limit {limit})")

    if unprocessed:
        # build caches once per pipeline run — shared across all compile calls
        logger.info("ORCHESTRATOR | Building caches...")
        extraction_caches = build_extraction_caches(unprocessed)
        template_caches   = build_template_caches()
        logger.info(
            f"ORCHESTRATOR | Caches ready — "
            f"{len(extraction_caches)} extraction, {len(template_caches)} template"
        )

        # all entity buffers accumulate signals across files; flushed once after the loop
        company_buffer:    dict = {}
        trial_buffer:      dict = {}
        drug_buffer:       dict = {}
        indication_buffer: dict = {}

        # success entries deferred — only marked processed after flush confirms wiki writes
        pending_success: list[tuple] = []

        for file_path in unprocessed:
            result = run_compiler_on_file(
                file_path, extraction_caches, template_caches,
                company_buffer, trial_buffer, drug_buffer, indication_buffer,
            )
            if result is not None:
                pending_success.append(result)

        # write all buffered pages in one parallel pass — one LLM call per entity
        total_signals = sum(
            len(v) for buf in (company_buffer, trial_buffer, drug_buffer, indication_buffer)
            for v in buf.values()
        )

        # Map each contributing file -> the set of page_paths its signals feed into,
        # so a partial flush failure only withholds success-marking from the files
        # that actually fed the failed page(s) — not every file in the batch just
        # because some other, unrelated page in the same flush succeeded.
        file_to_pages: dict[str, set] = {}

        def _index_buffer(buf: dict, path_fn) -> None:
            for key, entries in buf.items():
                page_path = path_fn(key)
                for entry in entries:
                    fp = entry.get("file_path")
                    if fp:
                        file_to_pages.setdefault(fp, set()).add(page_path)

        _index_buffer(company_buffer,    lambda slug: f"companies/{slug}.md")
        _index_buffer(trial_buffer,      lambda sponsor: f"trials/{sponsor}.md")
        _index_buffer(drug_buffer,       lambda slug: f"drugs/{slug}.md")
        _index_buffer(indication_buffer, lambda slug: f"indications/{slug}/_index.md")

        all_failed = False  # set when flush itself raises — withhold marking for everyone
        failed_pages: set = set()
        if total_signals:
            logger.info(
                f"ORCHESTRATOR | Flushing — "
                f"company={len(company_buffer)}, trial={len(trial_buffer)}, "
                f"drug={len(drug_buffer)}, indication={len(indication_buffer)} "
                f"({total_signals} total signals)"
            )
            try:
                written, failed_pages = flush_buffered_pages(
                    company_buffer, trial_buffer, drug_buffer, indication_buffer, template_caches,
                )
                if not written and not failed_pages:
                    all_failed = True
                    logger.warning(
                        f"ORCHESTRATOR | Flush wrote 0 pages — skipping success marking "
                        f"for {len(pending_success)} file(s) so they retry next run"
                    )
                elif failed_pages:
                    logger.warning(
                        f"ORCHESTRATOR | {len(failed_pages)} page(s) failed to write: "
                        f"{sorted(failed_pages)} — files that fed only these pages will retry next run"
                    )
            except Exception as exc:
                all_failed = True
                logger.error(
                    f"ORCHESTRATOR | Flush raised an exception — skipping success marking "
                    f"for {len(pending_success)} file(s) so they retry next run. Error: {exc}"
                )

        # mark a file processed only if none of the pages it contributed to failed
        marked = 0
        skipped = 0
        if not all_failed:
            for file_path, doc_type, company, drug in pending_success:
                contributed = file_to_pages.get(str(file_path), set())
                if contributed & failed_pages:
                    skipped += 1
                    continue
                mark_file_processed(file_path, doc_type, "success", company, drug)
                append_log(file_path, "success", doc_type)
                marked += 1
        if marked:
            logger.info(f"ORCHESTRATOR | Marked {marked} file(s) as processed")
        if skipped:
            logger.warning(f"ORCHESTRATOR | Left {skipped} file(s) unmarked — fed a failed page, will retry next run")
    else:
        logger.info("ORCHESTRATOR | No new files — skipping cache build")

    # run lint if due (checked against state.py timestamp, not log.md)
    if should_run_lint():
        logger.info("ORCHESTRATOR | Weekly lint check due — running...")
        from agents.lint import run_lint
        run_lint()
        mark_lint_run()

    from agents.cost import ledger
    ledger.report()
    ledger.reset()

    logger.info("ORCHESTRATOR | Pipeline complete")


def run_timeout_retry_pipeline() -> None:
    """Re-run the pipeline for all files previously skipped due to timeout."""
    reset_files = reset_timeout_files()
    if not reset_files:
        logger.info("ORCHESTRATOR | No timeout files to retry")
        return
    logger.info(f"ORCHESTRATOR | Retrying {len(reset_files)} timeout file(s):")
    for f in reset_files:
        logger.info(f"ORCHESTRATOR |   {f.name}")
    run_daily_pipeline()


def run_failed_retry_pipeline() -> None:
    """Re-run the pipeline for all failed files (429s, 499s, timeouts, transient errors).
    Skips files whose failures are known structural bugs."""
    reset_files = reset_failed_files()
    if not reset_files:
        logger.info("ORCHESTRATOR | No failed files to retry")
        return
    logger.info(f"ORCHESTRATOR | Retrying {len(reset_files)} failed file(s)")
    run_daily_pipeline()


def run_once(limit: int | None = None) -> None:
    """Run the pipeline once immediately — useful during development."""
    run_daily_pipeline(limit=limit)


def run_scheduled() -> None:
    """Schedule daily run at 07:00 — for production use."""
    schedule.every().day.at("07:00").do(run_daily_pipeline)
    logger.info("ORCHESTRATOR | Scheduled. Running daily at 07:00. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "once"
    limit = None
    if "--limit" in args:
        idx = args.index("--limit")
        try:
            limit = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Usage: orchestrator.py [once|schedule|retry-timeouts] [--limit N]")
            sys.exit(1)

    if cmd in ("--help", "-h", "help"):
        print("Usage: orchestrator.py [once|schedule|retry-timeouts|retry-failed] [--limit N]")
        sys.exit(0)
    elif cmd == "schedule":
        run_scheduled()
    elif cmd == "retry-timeouts":
        run_timeout_retry_pipeline()
    elif cmd == "retry-failed":
        run_failed_retry_pipeline()
    else:
        run_once(limit=limit)