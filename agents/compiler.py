"""
agents/compiler.py
PharmaLens wiki compiler — 3-step LLM chain.

Step 1: Extract entities and signals from a raw document  (1 LLM call, cached)
Step 2: Determine which wiki pages need updating           (pure Python)
Step 3: Write or update each wiki page                    (1 LLM call per page, cached)

Called by orchestrator.py for each new unprocessed file.
Caches are built once per pipeline run in the orchestrator and passed through context.
"""

import json
import os
import random
import re
import time
from pathlib import Path
from datetime import date
from typing import Literal
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from agents.logger import get_logger
from agents.cost import ledger

logger = get_logger("pharmalens.compiler")

# ── setup ─────────────────────────────────────────────────────────────────────

load_dotenv()  # must be before genai.Client()

client = genai.Client(http_options=types.HttpOptions(timeout=300_000))  # 300s in ms

FLASH_MODEL = "gemini-2.5-flash"

try:
    BASE_DIR = Path(__file__).parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

WIKI_DIR      = BASE_DIR / "wiki"
REFERENCE_DIR = BASE_DIR / "reference"
PROMPTS_DIR   = BASE_DIR / "agents" / "prompts"
TEMPLATES_DIR = PROMPTS_DIR / "templates"

SCHEMA = (PROMPTS_DIR / "CLAUDE.md").read_text()

# load reference data once at module level
DRUGS      = json.loads((REFERENCE_DIR / "drugs.json").read_text())
INDICATIONS = json.loads((REFERENCE_DIR / "indications.json").read_text())
COMPANIES  = json.loads((REFERENCE_DIR / "companies.json").read_text())

# build reverse alias maps — used for entity resolution
BRAND_TO_INN = {}
for inn, data in DRUGS.items():
    for alias in data.get("aliases", []):
        BRAND_TO_INN[alias.lower()] = inn
    for brand in data.get("brand_names", []):
        BRAND_TO_INN[brand.lower()] = inn

ALIAS_TO_INDICATION = {}
for slug, data in INDICATIONS.items():
    for alias in data.get("aliases", []):
        ALIAS_TO_INDICATION[alias.lower()] = slug
    for code in data.get("icd10", []):
        ALIAS_TO_INDICATION[code.lower()] = slug
    if data.get("mesh"):
        ALIAS_TO_INDICATION[data["mesh"].lower()] = slug
    if data.get("ctgov_condition"):
        ALIAS_TO_INDICATION[data["ctgov_condition"].lower()] = slug

ALIAS_TO_COMPANY = {}
for slug, data in COMPANIES.items():
    for alias in data.get("aliases", []):
        ALIAS_TO_COMPANY[alias.lower()] = slug


# ── Step 1 extraction schema (Pydantic) ────────────────────────────────────────
# Passed as response_schema so Gemini's decoding is constrained to this shape —
# response_mime_type="application/json" alone only asks nicely; this enforces it.
# The 5-level sentiment enum lets the company "Recent events" table render the
# Signal column directly from extracted data, with no LLM judgment call needed
# at page-write time (see _SIGNAL_LABELS / _build_event_row in flush_buffered_pages).

SentimentLevel = Literal[
    "bullish", "moderately_bullish", "neutral", "moderately_bearish", "bearish",
]

EventType = Literal[
    "fda_approval", "fda_rejection", "fda_warning", "label_expansion",
    "trial_completion", "trial_termination", "trial_initiation",
    "earnings_signal", "pipeline_update", "patent_event", "news", "pubmed_result",
]


class DrugMention(BaseModel):
    name: str
    brand_name: str | None = None
    revenue_usd_m: float | None = None
    revenue_growth_pct: float | None = None
    direction: str | None = None
    sentiment: SentimentLevel | None = None
    commentary: str | None = None
    is_pipeline: bool | None = None
    is_blockbuster: bool | None = None


class ClinicalFindings(BaseModel):
    # Identifies which tracked drug (INN, matching reference/drugs.json) this
    # finding is actually about. Without this, a document mentioning multiple
    # tracked drugs (common in multi-drug press releases) has no way to tell
    # which drug's page a finding belongs on — every mentioned drug would get
    # offered the same finding, risking either a dropped paragraph or, worse,
    # misattributing one drug's trial result to another drug's page.
    subject_drug: str | None = None
    trial_name: str | None = None
    study_design: str | None = None
    sample_size: str | None = None
    comparator: str | None = None
    primary_outcome: str | None = None
    primary_result: str | None = None
    secondary_results: list[str] = []
    safety_note: str | None = None
    conclusions_verbatim: str | None = None
    journal: str | None = None
    publication_year: str | None = None
    industry_sponsored: bool | None = None


class ExtractionResult(BaseModel):
    all_drugs_mentioned: list[DrugMention] = []
    companies_mentioned: list[str] = []
    indications_mentioned: list[str] = []
    trial_ids: list[str] = []
    event_type: EventType | None = None
    event_summary: str | None = None
    sentiment: SentimentLevel | None = None
    sentiment_reasoning: str | None = None
    key_facts: list[str] = []
    event_date: str | None = None
    requires_new_event_page: bool = False
    suggested_event_slug: str | None = None
    clinical_findings: ClinicalFindings | None = None


# ── Step 3 written-page frontmatter schemas ─────────────────────────────────
# These validate the LLM's *written* YAML frontmatter against the templates in
# agents/prompts/templates/*.md, at flush time — a different artifact and
# pipeline stage from the Step-1 extraction models above. Kept permissive
# (str | None, not strict enums) for now: the goal is catching structural
# breakage (missing required fields, wrong shape — what actually caused the
# AbbVie/Amgen incidents) without rejecting legitimate value variations we
# haven't seen yet. Tighten enums later once real drift patterns emerge.
#
# Date fields accept str | date: YAML parses unquoted dates as datetime.date,
# not str (see CLAUDE.md "Schema notes" — this already burned the project
# once in api/tools.py:parse_company_trials() and would otherwise make every
# legitimately-unquoted date in the wiki a false-positive validation failure).
DateOrStr = str | date | None


class TrialClinicalFindings(BaseModel):
    study_design: str | None = None
    sample_size: str | int | None = None
    comparator: str | None = None
    primary_outcome: str | None = None
    primary_result: str | None = None
    secondary_results: str | list | None = None
    safety_note: str | None = None
    conclusions_verbatim: str | None = None
    journal: str | None = None
    publication_year: str | int | None = None
    industry_sponsored: bool | None = None


class TrialFrontmatter(BaseModel):
    trial_id: str
    title: str
    phase: str | int | None = None
    status: str | None = None
    primary_sponsor: str
    co_sponsors: list[str] = []
    drugs: list[str] = []
    indications: list[str] = []
    enrollment: int | str | None = None
    primary_endpoint: str | None = None
    primary_completion_date: DateOrStr = None

    @field_validator("phase", mode="before")
    @classmethod
    def _normalize_phase(cls, v):
        # The LLM writes combined-phase trials with whatever separator style
        # it feels like at flush time ("1 | 2", "1|2", "1/2" all confirmed
        # present in production for the same Phase 1/2 concept), which
        # fragmented the frontend's phase-distribution chart into duplicate
        # bars. Canonicalizing here — at the write-time validation
        # checkpoint — fixes it at the source for every future trial, not
        # just at display time (see api/tools.py:normalize_phase(), which
        # remains the read-side safety net for trial pages written before
        # this validator existed).
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "none", "n/a", "null")):
            return None
        if isinstance(v, str):
            return re.sub(r"\s*\|\s*", "/", v.strip())
        return v
    has_results: bool
    primary_result_value: str | None = None
    result_summary: str | None = None
    clinical_findings: TrialClinicalFindings | None = None
    last_updated: DateOrStr = None

    # NOTE: a cross-field "clinical_findings must be null when has_results is
    # false" rule was tried and dropped — sweeping the existing wiki showed
    # ~150+ pre-existing trial blocks across many companies violate this in
    # practice (has_results: false with a populated clinical_findings), which
    # is exactly the kind of legitimate-but-unanticipated variation this
    # schema is meant to tolerate. Enforcing it here would freeze those
    # trials going forward — the opposite of this validation's purpose.


class CompanyFrontmatter(BaseModel):
    company: str
    full_name: str | None = None
    ticker: str | None = None
    exchange: str | None = None
    indications_active: list[str] = []
    blockbuster_drugs: list[str] = []
    pipeline_drugs: list[str] = []
    last_earnings_date: DateOrStr = None
    last_updated: DateOrStr = None


class DrugFrontmatter(BaseModel):
    drug: str
    brand_names: list[str] = []
    company: str
    indications: list[str] = []
    drug_class: str | None = None
    status: str | None = None
    fda_approval_date: DateOrStr = None
    patent_expiry: DateOrStr = None
    black_box_warning: bool | None = None
    blockbuster: bool | None = None
    management_sentiment: SentimentLevel | None = None
    sentiment_score: str | None = None
    last_earnings_signal: str | None = None
    reimbursement_flag: bool | None = None
    latest_event: str | None = None
    trials: list[str] = []
    last_updated: DateOrStr = None


class IndicationHubFrontmatter(BaseModel):
    indication: str
    display_name: str | None = None
    icd10: list[str] = []
    drugs_approved: list[str] = []
    drugs_pipeline: list[str] = []
    companies_active: list[str] = []
    active_trials: int | str | None = None
    last_updated: DateOrStr = None


def _extract_frontmatter(content: str) -> dict | None:
    """Pull and parse the leading --- YAML frontmatter block. None if missing/invalid."""
    m = re.match(r"^\s*---\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


# ── wiki helpers ──────────────────────────────────────────────────────────────

def read_wiki_page(page_path: str) -> str:
    """Read an existing wiki page. Returns empty string if not found."""
    from agents.wiki_gcs import read_wiki
    return read_wiki(page_path)


def _strip_fenced_code_block(content: str) -> str:
    """Strip ```markdown ... ``` or ``` ... ``` wrapper the LLM sometimes adds."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped[stripped.index("\n") + 1:]  # drop opening fence line
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")].rstrip()
    return stripped


def write_wiki_page(page_path: str, content: str) -> str:
    """Write content to a wiki page."""
    from agents.wiki_gcs import write_wiki
    return write_wiki(page_path, _strip_fenced_code_block(content))


# ── prompt helpers ────────────────────────────────────────────────────────────

def load_prompt(doc_type: str) -> str:
    """Load the extraction prompt for a given document type.
    Raises ValueError if no prompt file exists — fail loud, not silent.
    """
    prompt_file = PROMPTS_DIR / f"compiler_{doc_type}.txt"
    if prompt_file.exists():
        return prompt_file.read_text()
    raise ValueError(
        f"No prompt file found for document type '{doc_type}'. "
        f"Expected: {PROMPTS_DIR}/compiler_{doc_type}.txt — "
        f"Either add the prompt file or update classify_document() in orchestrator.py"
    )


def load_page_template(page_type: str) -> str:
    """Load the formatting template for a specific wiki page type.
    Only injected in Step 3 calls — not in Step 1 extraction or index updates.
    Raises ValueError if template file is missing.
    """
    template_file = TEMPLATES_DIR / f"{page_type}.md"
    if template_file.exists():
        return template_file.read_text()
    raise ValueError(
        f"No template found for page type '{page_type}'. "
        f"Expected: {TEMPLATES_DIR}/{page_type}.md"
    )


def build_system_prompt() -> str:
    """Build the system instruction: CLAUDE.md schema + reference data summary."""
    drug_summary = {
        k: {
            "inn": v["inn"],
            "brand_names": v["brand_names"],
            "company": v["company"],
            "indications": v["indications"],
        }
        for k, v in DRUGS.items()
    }
    indication_summary = {
        k: {"aliases": v.get("aliases", []), "icd10": v.get("icd10", [])}
        for k, v in INDICATIONS.items()
    }

    return f"""{SCHEMA}

Reference data (use for entity resolution — do NOT invent entity names):
DRUGS: {json.dumps(drug_summary, indent=2)}
INDICATIONS: {json.dumps(indication_summary, indent=2)}

Entity resolution rules:
- Always normalize brand names to INN using the DRUGS reference
- Always normalize indication aliases to indication slugs using INDICATIONS
- Never create wiki pages for entities not in the reference data
- If an entity is not in reference data, note it in extraction output but do not create pages
"""


# ── document pre-processors ───────────────────────────────────────────────────

def _compact_pubmed_results(text: str) -> str:
    """Strip author list, affiliations, and COI from a PubMed MEDLINE string.
    Returns title + structured abstract sections only (~1–2k chars vs ~10k raw).
    """
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    citation_line = ""
    title = ""
    for i, para in enumerate(paragraphs):
        if re.match(r'^\d+\.', para):
            citation_line = para.splitlines()[0].strip()  # first line only (journal + year)
            if i + 1 < len(paragraphs):
                title = paragraphs[i + 1]
            break

    abstract_start = re.search(
        r'^(BACKGROUND|OBJECTIVE|OBJECTIVES|PURPOSE|AIMS?|METHODS?|DESIGN|SETTING|PARTICIPANTS|FINDINGS|RESULTS|CONCLUSIONS?|INTERPRETATION|SUMMARY):',
        text, re.MULTILINE,
    )
    stop = re.search(
        r'^(Copyright|DOI:|PMCID:|PMID:|Conflict of interest)',
        text, re.MULTILINE | re.IGNORECASE,
    )

    abstract_text = ""
    if abstract_start:
        end = stop.start() if stop else len(text)
        abstract_text = text[abstract_start.start():end].strip()

    parts = []
    if citation_line:
        parts.append(f"Citation: {citation_line}")
    if title:
        parts.append(f"Title: {title}")
    if abstract_text:
        parts.append(abstract_text)
    return "\n\n".join(parts)


def preprocess_ctgov(raw: str) -> str:
    """Parse CT.gov JSON and return only the fields defined in our extraction spec.
    Filters interventions to DRUG and BIOLOGICAL types only.
    Falls back to raw text if JSON is malformed.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:12000]

    relevant = {
        "nct_id":                    data.get("NCTId"),
        "brief_title":               data.get("BriefTitle"),
        "overall_status":            data.get("OverallStatus"),
        "phase":                     data.get("Phase"),
        "study_type":                data.get("StudyType"),
        "start_date":                data.get("StartDate"),
        "primary_completion_date":   data.get("PrimaryCompletionDate"),
        "completion_date":           data.get("CompletionDate"),
        "last_updated":              data.get("LastUpdatePostDate"),
        "lead_sponsor":              data.get("LeadSponsorName"),
        "lead_sponsor_class":        data.get("LeadSponsorClass"),
        "collaborator":              data.get("CollaboratorName"),
        "conditions":                data.get("Condition"),
        "brief_summary":             data.get("BriefSummary"),
        "enrollment_count":          data.get("EnrollmentCount"),
        "enrollment_type":           data.get("EnrollmentType"),
        "intervention_type":         data.get("InterventionType"),
        "intervention_name":         data.get("InterventionName"),
        "primary_outcome_measure":   data.get("PrimaryOutcomeMeasure"),
        "primary_outcome_timeframe": data.get("PrimaryOutcomeTimeFrame"),
        "has_results":               data.get("HasResults"),
    }

    # attach compacted PubMed abstract when the DE enrichment job has run
    results_text = data.get("results")
    if results_text and isinstance(results_text, str):
        compacted = _compact_pubmed_results(results_text)
        if compacted:
            relevant["pubmed_results"] = compacted

    # keep interventions only if at least one component is DRUG or BIOLOGICAL.
    # InterventionType is a pipe-delimited string e.g. "DRUG | DRUG | OTHER"
    int_type = relevant.get("intervention_type")
    if int_type:
        types = [t.strip().upper() for t in str(int_type).split("|")]
        if not any(t in ("DRUG", "BIOLOGICAL") for t in types):
            relevant["intervention_type"] = None
            relevant["intervention_name"] = None

    relevant = {k: v for k, v in relevant.items() if v is not None}
    return json.dumps(relevant, indent=2)


def preprocess_pubmed(raw: str) -> str:
    """Parse PubMed JSON and return only the fields defined in our extraction spec.
    Falls back to raw text if JSON is malformed.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:12000]

    relevant = {
        "pmid":               data.get("pmid"),
        "doi":                data.get("doi"),
        "pubmed_date":        data.get("pubmed_date"),
        "title":              data.get("title"),
        "journal":            data.get("journal"),
        "journal_abbr":       data.get("journal_abbr"),
        "publication_types":  data.get("publication_types", []),
        "abstract":           data.get("abstract"),
        "mesh_major_topics":  data.get("mesh_major_topics", []),
        "first_author":       data.get("first_author"),
        "industry_sponsored": data.get("industry_sponsored"),
        "industry_note":      data.get("industry_note"),
    }

    relevant = {k: v for k, v in relevant.items() if v is not None}
    return json.dumps(relevant, indent=2)


def preprocess_text(raw: str, max_chars: int) -> str:
    """Clean and intelligently truncate plain text documents.

    Removes blank lines to compact the text. If the document exceeds max_chars,
    keeps the first 70% and last 30% with a truncation notice in between.

    The 70/30 split is deliberate for earnings transcripts:
    - First 70%: CEO prepared remarks, CFO financial summary (highest signal)
    - Last 30%: analyst Q&A session often contains forward guidance and risk flags
    - Middle: less critical boilerplate, legal disclaimers, repetitive detail
    """
    cleaned = "\n".join(line for line in raw.splitlines() if line.strip())

    if len(cleaned) <= max_chars:
        return cleaned

    keep_start    = int(max_chars * 0.70)
    keep_end      = int(max_chars * 0.30)
    chars_dropped = len(cleaned) - max_chars

    return (
        cleaned[:keep_start]
        + f"\n\n[... {chars_dropped:,} characters truncated — middle section omitted ...]\n\n"
        + cleaned[-keep_end:]
    )

def preprocess_document(file_path: Path, doc_type: str) -> str:
    """Pre-process a raw file before passing to the LLM.

    JSON files (ctgov, pubmed): parse and extract only the fields the compiler
    needs — avoids passing hundreds of irrelevant fields to the LLM.

    Text/HTML files (edgar, genepool): clean and truncate intelligently.
    8-K files get the exhibit extractor before truncation.
    10-Q files get more chars since they contain XBRL + MD&A sections.

    Returns a clean string ready for the Step 1 extraction prompt.
    """
    from agents.gcs import read_blob
    raw = read_blob(file_path, BASE_DIR)

    if doc_type == "ctgov":
        return preprocess_ctgov(raw)
    elif doc_type == "pubmed":
        return preprocess_pubmed(raw)
    elif doc_type == "edgar_8k":
        return preprocess_text(raw, max_chars=8000)
    elif doc_type == "edgar_10q":
        return preprocess_text(raw, max_chars=20000)
    elif doc_type == "genepool":
        return preprocess_text(raw, max_chars=8000)
    else:
        return preprocess_text(raw, max_chars=12000)


# ── ctgov pure-Python Step 1 ─────────────────────────────────────────────────

def extract_ctgov_python(preprocessed: dict) -> dict:
    """
    Pure-Python Step 1 replacement for ctgov files.
    Takes the preprocessed dict from preprocess_ctgov() (snake_case keys).

    For ctgov, Step 1 is almost entirely entity resolution against reference
    data — company from sponsor name, drugs from intervention name, indications
    from condition field. Python handles this accurately using the same alias
    maps the LLM would use, saving one LLM call per file.
    """
    # company: substring-match lead_sponsor against alias map
    sponsor = (preprocessed.get("lead_sponsor") or "").lower()
    company = next(
        (slug for alias, slug in ALIAS_TO_COMPANY.items() if alias in sponsor),
        None,
    )

    # drugs: split intervention_name by "|", match each segment against INN + brands
    intervention_str = str(preprocessed.get("intervention_name") or "")
    intervention_parts = [p.strip().lower() for p in intervention_str.split("|") if p.strip()]

    # terms that are definitely not drug names — skip them for untracked list
    _NON_DRUG_TERMS = {
        "placebo", "standard of care", "usual care", "best supportive care",
        "no intervention", "observation", "sham", "vehicle", "control",
        "lifestyle", "exercise", "diet", "surgery", "radiation",
    }

    matched_drugs: list[str] = []
    matched_part_indices: set[int] = set()
    for inn, info in DRUGS.items():
        brands = [b.lower() for b in info.get("brand_names", [])]
        for i, part in enumerate(intervention_parts):
            if inn in part or any(b in part for b in brands):
                if inn not in matched_drugs:
                    matched_drugs.append(inn)
                matched_part_indices.add(i)
                break

    # untracked: parts that didn't match any tracked drug and aren't noise
    untracked_names: list[str] = []
    for i, part in enumerate(intervention_parts):
        if i in matched_part_indices:
            continue
        if len(part) < 3:
            continue
        if any(noise in part for noise in _NON_DRUG_TERMS):
            continue
        untracked_names.append(part[0].upper() + part[1:])

    # indications: split conditions by "|", match each segment against alias map
    conditions_str = str(preprocessed.get("conditions") or "")
    condition_parts = [c.strip().lower() for c in conditions_str.split("|") if c.strip()]
    matched_indications: list[str] = []
    for part in condition_parts:
        for alias, slug in ALIAS_TO_INDICATION.items():
            if alias in part and slug not in matched_indications:
                matched_indications.append(slug)

    nct_id = preprocessed.get("nct_id")
    status  = (preprocessed.get("overall_status") or "").upper()
    _TERMINAL = {"COMPLETED", "TERMINATED", "WITHDRAWN"}
    requires_event = status in _TERMINAL and bool(company)
    event_type = {"COMPLETED": "trial_completion", "TERMINATED": "trial_termination"}.get(status)

    # Deterministic minimum sentiment for the company events table — judging the
    # actual result direction reliably without LLM interpretation isn't possible
    # here, so this only covers the part that IS unambiguous: stopping a trial
    # early is bearish; reaching completion per protocol is neutral by default
    # (clinical_findings, when attached below, carries the real result detail).
    sentiment = {
        "TERMINATED": "bearish",
        "WITHDRAWN":  "bearish",
        "COMPLETED":  "neutral",
    }.get(status)

    date = (
        preprocessed.get("primary_completion_date")
        or preprocessed.get("completion_date")
        or ""
    )[:10]
    if matched_drugs:
        drug_slug = matched_drugs[0]
    elif untracked_names:
        drug_slug = untracked_names[0].lower().replace(" ", "-")
    else:
        drug_slug = "unknown"
    slug = (
        f"{date}-{drug_slug}-{(nct_id or '').lower()}-{status.lower()}"
        if requires_event else None
    )

    key_facts = [f for f in [
        f"Phase: {preprocessed['phase']}"                       if preprocessed.get("phase") else None,
        f"Status: {status}",
        (f"Enrollment: {preprocessed['enrollment_count']} "
         f"({preprocessed.get('enrollment_type', '')})")        if preprocessed.get("enrollment_count") else None,
        f"Primary completion: {preprocessed['primary_completion_date']}"
                                                                if preprocessed.get("primary_completion_date") else None,
        f"Has results: {preprocessed['has_results']}"           if preprocessed.get("has_results") is not None else None,
    ] if f is not None]

    # One-line, deterministic event description — only built for terminal-status
    # trials (the only case that produces a company "Recent events" row or an
    # event page). Non-terminal ctgov documents never need a description since
    # ongoing trial status already lives on the trials/{company}.md page.
    event_summary = None
    if requires_event:
        phase = preprocessed.get("phase")
        title = preprocessed.get("brief_title") or nct_id
        verb = "terminated" if status == "TERMINATED" else "withdrawn" if status == "WITHDRAWN" else "completed"
        phase_label = f"Phase {phase} " if phase else ""
        event_summary = f"{title} ({phase_label}{nct_id}) {verb}."

    return {
        "all_drugs_mentioned": (
            [{"name": inn, "is_tracked": True} for inn in matched_drugs] +
            [{"name": name, "is_tracked": False} for name in untracked_names]
        ),
        "companies_mentioned":     [company] if company else [],
        "indications_mentioned":   matched_indications,
        "trial_ids":               [nct_id] if nct_id else [],
        "event_type":              event_type,
        "event_summary":           event_summary,
        "sentiment":               sentiment,
        "sentiment_reasoning":     None,
        "key_facts":               key_facts,
        "event_date":              date or None,
        "requires_new_event_page": requires_event,
        "suggested_event_slug":    slug,
        "clinical_findings":       None,
    }


def _clinical_finding_matches_drug(extracted: dict, drug_slug: str) -> bool:
    """True if a signal's clinical_findings actually belongs on drug_slug's page.

    Prefers the explicit subject_drug field set at extraction time. Falls back
    to "this document only mentions one tracked drug, so it's unambiguous"
    when subject_drug wasn't filled — but if the document mentions several
    tracked drugs and subject_drug is empty, this returns False rather than
    guessing, since attributing a finding to the wrong drug's page (a quiet
    factual error) is worse than skipping it (a logged, retried gap)."""
    cf = extracted.get("clinical_findings")
    if not cf or not isinstance(cf, dict):
        return False
    subject = (cf.get("subject_drug") or "").strip().lower()
    slug = drug_slug.strip().lower()
    if subject:
        return subject == slug or subject in slug or slug in subject
    tracked = [d["name"] for d in extracted.get("all_drugs_mentioned", []) if d.get("is_tracked")]
    return len(tracked) == 1 and tracked[0].strip().lower() == slug


def _extract_clinical_findings(pubmed_results_text: str) -> dict | None:
    """Targeted LLM call to extract clinical_findings from a compacted PubMed abstract.
    Called only when the DE enrichment job has attached a `results` field to a ctgov file.
    Returns the clinical_findings dict or None on parse failure.
    """
    prompt = f"""Extract clinical findings from this PubMed abstract. Return ONLY valid JSON with no markdown fences.

{pubmed_results_text}

{{
  "study_design": "RCT | meta-analysis | systematic review | observational | null",
  "sample_size": "N patients or null",
  "comparator": "placebo | standard of care | competitor drug name | null",
  "primary_outcome": "exact outcome measure as stated",
  "primary_result": "exact result with numbers and p-value or hazard ratio",
  "secondary_results": ["result with numbers", "result with numbers"],
  "safety_note": "key adverse event finding or null",
  "conclusions_verbatim": "copy the CONCLUSIONS or INTERPRETATION section exactly",
  "journal": "journal name",
  "publication_year": "YYYY",
  "industry_sponsored": true
}}"""

    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ClinicalFindings,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    ledger.record(response.usage_metadata)
    try:
        return ClinicalFindings.model_validate_json(response.text).model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"_extract_clinical_findings | schema validation failed: {e} — {response.text[:200]}")
        return None


# ── ctgov pre-LLM filter ─────────────────────────────────────────────────────

def _ctgov_passes_filter(parsed: dict) -> tuple[bool, str]:
    """
    Pure-Python gate applied before the Step 1 LLM call for ctgov files.
    Returns (True, "") to proceed, or (False, reason) to skip.

    Goal: keep ALL drug/biological trials from tracked companies — the ctgov
    pipeline is a company-level portfolio view, not drug-specific. Understanding
    a company's full pipeline (including untracked drugs and new indications)
    is the point.

    The only thing worth skipping: trials with NO drug or biological intervention
    at all (device studies, blood draws, behavioral/procedural arms).
    preprocess_ctgov() already nullifies intervention_type/name for those, so
    if both fields are absent from the preprocessed dict this trial has zero
    pharmaceutical signal.
    """
    if "intervention_type" not in parsed and "intervention_name" not in parsed:
        return False, (
            f"no drug/biological intervention "
            f"(status={parsed.get('overall_status')!r}, "
            f"conditions={parsed.get('conditions', [])[:2]})"
        )
    return True, ""


# ── path helpers ──────────────────────────────────────────────────────────────

def get_company_from_path(path: Path) -> str | None:
    """Extract company slug from path for edgar and ctgov files."""
    parts = path.parts
    for source in ("edgar", "ctgov"):
        if source in parts:
            idx = list(parts).index(source)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def get_drug_from_path(path: Path) -> str | None:
    """Extract drug INN from pubmed path."""
    parts = path.parts
    if "pubmed" in parts:
        idx = list(parts).index("pubmed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def classify_document(path: Path) -> str:
    """Infer document type from folder path — no LLM needed.
    Folder structure: raw/edgar/{company}/8k/, raw/edgar/{company}/10q/, or
    raw/edgar/{company}/6k/ (foreign private issuers — AstraZeneca, Novartis,
    GSK, Sanofi, Takeda all file 6-K instead of 8-K with the SEC).
    Returns: ctgov | edgar_8k | edgar_10q | genepool | pubmed | unknown
    """
    parts       = path.parts
    parts_lower = [p.lower() for p in parts]

    if "ctgov" in parts:
        return "ctgov"
    elif "edgar" in parts:
        if "10q" in parts_lower:
            return "edgar_10q"
        elif "8k" in parts_lower or "6k" in parts_lower:
            # 6-K (foreign private issuer material event disclosure) is
            # extracted identically to 8-K — same prompt, same preprocessing,
            # same purpose, just a different SEC form number for non-US filers.
            return "edgar_8k"
        else:
            logger.warning(f"classify_document | Unknown edgar subfolder for {path}")
            return "edgar_8k"
    elif "genepool" in parts:
        return "genepool"
    elif "pubmed" in parts:
        return "pubmed"
    else:
        return "unknown"


# ── 3-step compiler chain ─────────────────────────────────────────────────────

def compile_document_step1(
    file_path: Path,
    context: dict,
    raw_content: str,
    doc_type: str,
    system_prompt: str,
    cache_name: str | None = None,
) -> dict:
    """Step 1 — extract structured entities and signals from the raw document.
    One LLM call. Returns parsed JSON dict.

    system_prompt is the full text (built once by compile_document) — always
    passed, used as a fallback. cache_name, when set, is used instead via the
    `cached_content` field (previously this was sent as a literal
    `system_instruction` string, i.e. the cache was created and billed for
    storage but never actually applied to this call — see the
    `extraction_caches`/`build_extraction_caches` path in orchestrator.py).
    """
    step1_prompt = f"""Document to process:
---
{raw_content}
---

File path: {file_path}
Document type: {doc_type}
Company context (from path): {context.get('company', 'unknown')}
Drug context (from path): {context.get('drug', 'unknown')}
File date (use this as event_date and last_updated — do not use dates from document body): {context.get('file_date', 'unknown')}

Extract all relevant entities and signals. Return a JSON object:
{{
  "all_drugs_mentioned": [
    {{
      "name": "risankizumab",
      "brand_name": "Skyrizi",
      "revenue_usd_m": 3425,
      "revenue_growth_pct": 70.5,
      "direction": "growing",
      "sentiment": "bullish",
      "commentary": "one sentence from management or results",
      "is_pipeline": false,
      "is_blockbuster": true
    }}
  ],
  "companies_mentioned": ["novo-nordisk"],
  "indications_mentioned": ["glp1-obesity"],
  "trial_ids": ["NCT03819153"],
  "event_type": "fda_approval | trial_completion | trial_termination | earnings_signal | news | pubmed_result | null",
  "event_summary": "one sentence describing the key signal",
  "sentiment": "bullish | moderately_bullish | neutral | moderately_bearish | bearish | null",
  "sentiment_reasoning": "brief quote or paraphrase supporting sentiment",
  "key_facts": ["fact 1", "fact 2"],
  "event_date": "YYYY-MM-DD or null",
  "requires_new_event_page": true,
  "suggested_event_slug": "2026-04-02-semaglutide-ckd-fda-approval",
  "clinical_findings": {{
    "subject_drug": "the single tracked drug INN this specific finding is about — "
                     "null if the document mentions no clear single drug, or if you "
                     "cannot tell which of several mentioned drugs this finding concerns",
    "trial_name": "trial name/acronym this finding is from, e.g. DESTINY-Breast09, or null",
    "study_design": "RCT | meta-analysis | systematic review | observational | null",
    "sample_size": "3533 patients | null",
    "comparator": "placebo | standard of care | competitor drug | null",
    "primary_outcome": "exact outcome measure as stated in abstract",
    "primary_result": "exact result with numbers",
    "secondary_results": ["result 1 with numbers", "result 2 with numbers"],
    "safety_note": "any adverse event finding or null",
    "conclusions_verbatim": "copy the CONCLUSIONS section exactly as written",
    "journal": "journal name",
    "publication_year": "YYYY",
    "industry_sponsored": true
  }}
}}
IMPORTANT: if this document mentions multiple tracked drugs, subject_drug MUST
identify which one this specific clinical_findings entry is about — never leave
it null just because there are several drugs in the document; only leave it
null when the finding truly can't be attributed to one drug.
Return ONLY valid JSON with no markdown fences.
"""

    if cache_name:
        step1_config = types.GenerateContentConfig(
            cached_content=cache_name,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ExtractionResult,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    else:
        step1_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ExtractionResult,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    step1_response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=step1_prompt,
        config=step1_config,
    )
    ledger.record(step1_response.usage_metadata)

    try:
        return ExtractionResult.model_validate_json(step1_response.text).model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(
            f"Step 1 schema validation failed for {file_path}: {e}. "
            f"Response was: {step1_response.text[:300]}"
        )


def collect_entity_pages(
    extracted: dict,
    pages_to_update: list,
    context: dict,
    file_path: Path,
    company_buffer: dict | None = None,
    drug_buffer: dict | None = None,
    indication_buffer: dict | None = None,
) -> bool:
    """Step 2a — validate extracted entities against reference data.
    Routes each entity to its batch buffer (if provided) or to pages_to_update.
    Buffered pages are written once at end-of-batch via flush_buffered_pages().
    Returns True if any tracked entity was found.
    """
    # enrich all_drugs_mentioned with is_tracked flag (set here in Python, not by LLM)
    # Each entry is expected to be a dict (per the DrugMention schema), but this
    # has been hit by malformed/legacy-shaped entries in production (logged as
    # "'str' object has no attribute 'get'") — drop anything that isn't a dict
    # rather than crashing the whole file's processing over one bad entry.
    all_drugs_raw = extracted.get("all_drugs_mentioned", []) or []
    all_drugs = []
    for drug in all_drugs_raw:
        if not isinstance(drug, dict):
            logger.warning(f"STEP2 | SKIP DRUG MENTION | not a dict, got {type(drug).__name__}: {drug!r}")
            continue
        drug["is_tracked"] = drug.get("name", "") in DRUGS
        all_drugs.append(drug)
    extracted["all_drugs_mentioned"] = all_drugs

    # derive tracked drug names for page routing
    tracked_drug_names = [d["name"] for d in all_drugs if d.get("is_tracked")]

    # fallback: if LLM missed the company, inject from path context
    companies_mentioned = [c for c in (extracted.get("companies_mentioned", []) or []) if isinstance(c, str)]
    if not companies_mentioned and context.get("company") in COMPANIES:
        logger.warning(
            f"STEP2 | companies_mentioned empty — "
            f"falling back to path context: {context['company']}"
        )
        companies_mentioned = [context["company"]]

    entity_configs = [
        {
            "reference": DRUGS,
            "path_fn":   lambda e: f"drugs/{e}.md",
            "type":      "drug",
            "ref_name":  "reference/drugs.json",
            "entities":  tracked_drug_names,
            "buffer":    drug_buffer,
        },
        {
            "reference": COMPANIES,
            "path_fn":   lambda e: f"companies/{e}.md",
            "type":      "company",
            "ref_name":  "reference/companies.json",
            "entities":  companies_mentioned,
            "buffer":    company_buffer,
        },
        {
            "reference": INDICATIONS,
            "path_fn":   lambda e: f"indications/{e}/_index.md",
            "type":      "indication_hub",
            "ref_name":  "reference/indications.json",
            "entities":  [e for e in (extracted.get("indications_mentioned", []) or []) if isinstance(e, str)],
            "buffer":    indication_buffer,
        },
    ]

    found_any = False
    signal = {
        "file_path": str(file_path),
        "doc_type":  context.get("doc_type", ""),
        "extracted": extracted,
    }

    for config in entity_configs:
        for entity in config["entities"]:
            if not isinstance(entity, str):
                logger.warning(f"STEP2 | SKIP ENTITY | non-string {config['type']} mention: {entity!r}")
                continue
            normalized = entity if entity in config["reference"] else entity.lower()

            # fall back to alias maps when the LLM returns a name instead of a slug
            if normalized not in config["reference"]:
                if config["type"] == "indication_hub":
                    normalized = ALIAS_TO_INDICATION.get(entity.lower(), normalized)
                elif config["type"] == "company":
                    normalized = ALIAS_TO_COMPANY.get(entity.lower(), normalized)

            if normalized in config["reference"]:
                found_any = True
                buf = config["buffer"]
                if buf is not None:
                    buf.setdefault(normalized, []).append(signal)
                    logger.info(
                        f"STEP2 | BUFFER {config['type'].upper()} | {normalized} — "
                        f"signal queued for batch flush"
                    )
                else:
                    page_path = config["path_fn"](normalized)
                    pages_to_update.append({
                        "path":                page_path,
                        "type":                config["type"],
                        "entity":              normalized,
                        "current":             read_wiki_page(page_path),
                        "all_drugs_mentioned": all_drugs,
                    })
            else:
                logger.warning(
                    f"STEP2 | SKIP ENTITY | '{entity}' not in {config['ref_name']}"
                )

    return found_any


def collect_trial_and_event_pages(
    extracted: dict,
    pages_to_update: list,
    file_path: Path,
    context: dict,
    trial_buffer: dict | None = None,
) -> bool:
    """Step 2b — validate and accumulate trial and event page entries.
    Trials: buffered per-company if trial_buffer provided; else accumulated onto pages_to_update.
    Events: always written per-file (each event has a unique slug).
    Returns True if any tracked entity was found.
    """
    from agents.state import get_nct_action, mark_nct_processed

    found_any = False

    for nct_id in extracted.get("trial_ids", []) or []:
        if not isinstance(nct_id, str) or not (nct_id.startswith("NCT") and len(nct_id) == 11):
            logger.warning(f"STEP2 | SKIP ENTITY | Malformed trial ID: {nct_id!r}")
            continue

        primary_sponsor = next(
            (c for c in (extracted.get("companies_mentioned", []) or [])
             if isinstance(c, str) and (c in COMPANIES or c.lower() in COMPANIES)),
            None,
        )
        if primary_sponsor and primary_sponsor not in COMPANIES:
            primary_sponsor = primary_sponsor.lower()

        if not primary_sponsor:
            logger.warning(
                f"STEP2 | SKIP TRIAL | {nct_id} — no tracked company identified as sponsor"
            )
            continue

        if context.get("out_of_scope"):
            logger.info(
                f"STEP2 | SKIP TRIAL | {nct_id} — out of scope, company page only"
            )
            continue

        found_any = True
        nct_action = get_nct_action(nct_id)
        mark_nct_processed(nct_id, primary_sponsor, extracted.get("event_date"))
        logger.debug(f"STEP2 | {nct_id} → action={nct_action}, sponsor={primary_sponsor}")

        if trial_buffer is not None:
            trial_buffer.setdefault(primary_sponsor, []).append({
                "nct_id":    nct_id,
                "action":    nct_action,
                "extracted": extracted,
                "file_path": str(file_path),
            })
            logger.info(
                f"STEP2 | BUFFER TRIAL | {primary_sponsor}/{nct_id} ({nct_action}) — "
                f"queued for batch flush"
            )
        else:
            trial_page_path = f"trials/{primary_sponsor}.md"
            existing = next(
                (p for p in pages_to_update if p["path"] == trial_page_path), None
            )
            if existing is None:
                pages_to_update.append({
                    "path":    trial_page_path,
                    "type":    "trial",
                    "entity":  primary_sponsor,
                    "current": read_wiki_page(trial_page_path),
                    "nct_ids": [{"nct_id": nct_id, "action": nct_action}],
                })
            else:
                existing["nct_ids"].append({"nct_id": nct_id, "action": nct_action})
                logger.debug(f"STEP2 | Accumulated {nct_id} onto {trial_page_path}")

    if extracted.get("requires_new_event_page") and extracted.get("suggested_event_slug"):
        event_path = f"events/{extracted['suggested_event_slug']}.md"
        pages_to_update.append({
            "path":    event_path,
            "type":    "event",
            "entity":  extracted["suggested_event_slug"],
            "current": "",
        })
        logger.debug(f"STEP2 | Queued event: {event_path}")
        found_any = True

    return found_any


def build_trial_context(page: dict) -> str:
    """Build the trial-specific NCT instruction block for Step 3 prompts.
    Returns an empty string for all non-trial page types.
    """
    if page["type"] != "trial":
        return ""

    nct_details = page.get("nct_ids", [])
    new_ncts    = [n["nct_id"] for n in nct_details if n["action"] == "new"]
    update_ncts = [n["nct_id"] for n in nct_details if n["action"] == "update"]

    return f"""
Trial page instructions (determined by processing state — do not override):
- NCT IDs to ADD as new entries: {new_ncts if new_ncts else "none"}
- NCT IDs to UPDATE existing entries: {update_ncts if update_ncts else "none"}
- For ADD: create a new section under the correct status group (Active / Completed / Terminated)
  and add a new row to the summary table.
- For UPDATE: find the existing section by NCT ID and update status, result, and completion date.
  Do NOT add a duplicate section. Do NOT remove any other existing trial entries.
"""


def write_wiki_pages(
    pages_to_update: list,
    extracted: dict,
    file_path: Path,
    doc_type: str,
    system_prompt: str,
    template_caches: dict,
) -> list[str]:
    """Step 3 — write or update each wiki page identified in Step 2.
    All LLM calls fire in parallel (one thread per page), then results are
    written to disk in the main thread. Uses cached template if available.
    Returns the list of wiki page paths written.
    """
    import concurrent.futures as _cf

    if not pages_to_update:
        return []

    # build clinical block once — shared across all page prompts
    clinical_block = ""
    if extracted.get("clinical_findings"):
        cf = extracted["clinical_findings"]
        clinical_block = f"""
Clinical findings to include verbatim in the Clinical evidence section:
- Study design: {cf.get('study_design')}
- Sample size: {cf.get('sample_size')}
- Comparator: {cf.get('comparator')}
- Primary outcome: {cf.get('primary_outcome')}
- Primary result: {cf.get('primary_result')}
- Secondary results: {cf.get('secondary_results', [])}
- Safety: {cf.get('safety_note')}
- Conclusions (verbatim): {cf.get('conclusions_verbatim')}
- Journal: {cf.get('journal')}, {cf.get('publication_year')}
- Industry sponsored: {cf.get('industry_sponsored')}

These numbers MUST appear in the Clinical evidence section exactly as stated.
Do not paraphrase the primary result or conclusions.
"""

    # pre-build (prompt, config) for every page — no LLM involved yet
    tasks: list[tuple[dict, str, types.GenerateContentConfig]] = []
    for page in pages_to_update:
        trial_context = build_trial_context(page)
        prompt = f"""{trial_context}
{clinical_block}
Write the updated wiki page for: {page['path']}
Page type: {page['type']}
Entity: {page['entity']}

Extracted signals from the new document:
{json.dumps(extracted, indent=2)}

Current page content (empty if new page):
---
{page['current'] if page['current'] else '[NEW PAGE — write from scratch using the template above]'}
---

Source document: {file_path} (type: {doc_type})

Rules:
- Follow the page template exactly for this page type
- If the page exists: UPDATE it — preserve all existing content, only add or change what this document affects
- If new page: write complete page from scratch
- Timeline section is append-only — add new rows at the top, never delete existing rows
- End with a ## Sources section listing the raw file path
- For all_drugs_mentioned in the extracted signals: include ALL entries (both is_tracked=true and
  is_tracked=false) in the drugs frontmatter field. Use [[drug_name]] link syntax in body text for
  is_tracked=true drugs only. Write is_tracked=false drugs as plain text in the body.
- last_updated: always use the source document date, never a date from within the document body
- Write ONLY the markdown content — no preamble, no explanation
"""
        cache_name = template_caches.get(page["type"])
        if cache_name:
            config = types.GenerateContentConfig(
                cached_content=cache_name, temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        else:
            page_template = load_page_template(page["type"])
            config = types.GenerateContentConfig(
                system_instruction=system_prompt + "\n\n" + page_template,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        tasks.append((page, prompt, config))

    def _call_llm(task: tuple) -> tuple[dict, str, object, float]:
        page, prompt, config = task
        start = time.time()
        resp = client.models.generate_content(
            model=FLASH_MODEL, contents=prompt, config=config,
        )
        return page, resp.text, resp.usage_metadata, time.time() - start

    # fire all LLM calls in parallel — bounded by number of pages per file (≤5)
    with _cf.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        results = list(pool.map(_call_llm, tasks, timeout=600))

    # write to disk and record usage in main thread (avoids any shared-state races)
    updated_paths = []
    for page, content, usage, elapsed in results:
        logger.info(f"STEP3 | API call took {elapsed:.1f}s for {page['type']}")
        ledger.record(usage)
        write_wiki_page(page["path"], content)
        updated_paths.append(page["path"])
        logger.info(f"STEP3 | WROTE | {page['path']}")

    return updated_paths


# ── company "Recent events" table — rendered deterministically, never by the LLM ─

_EVENT_DOC_TYPE_LABELS = {"edgar_8k": "sec", "edgar_10q": "sec", "ctgov": "trial", "pubmed": "research"}
_SIGNAL_LABELS = {
    "bullish":            "Bullish",
    "moderately_bullish": "Moderately Bullish",
    "neutral":            "Neutral",
    "moderately_bearish": "Moderately Bearish",
    "bearish":            "Bearish",
}


def _build_event_row(signal: dict) -> dict | None:
    """Derive one canonical events-table row from a buffered company signal.
    Returns None when the signal doesn't carry enough to make a meaningful row
    (e.g. a non-terminal ctgov status update, or a doc type with no mapping)."""
    extracted = signal.get("extracted", {})
    doc_type  = signal.get("doc_type", "")
    event_type_label = _EVENT_DOC_TYPE_LABELS.get(doc_type)
    event_text = extracted.get("event_summary")
    if event_type_label is None or not event_text:
        return None

    pmid = extracted.get("pmid")
    source = f"PMID:{pmid}" if doc_type == "pubmed" and pmid else ""

    return {
        "date":      extracted.get("event_date") or "",
        "type":      event_type_label,
        "event":     event_text,
        "signal":    _SIGNAL_LABELS.get(extracted.get("sentiment"), ""),
        "source":    source,
        "file_path": signal.get("file_path", ""),
    }


def _strip_section_for_prompt(content: str, heading: str) -> str:
    """Remove a deterministic section from page content BEFORE it's shown to
    the LLM as "current page content" — not after. Confirmed in testing
    (semaglutide hit MAX_TOKENS repeatedly): telling the LLM "if you see this
    section, leave it exactly where it is" while still showing it a 170KB
    section pushes the model toward reproducing that section verbatim in its
    output to honor "leave it where it is", defeating the entire point of
    making the section deterministic. If the section never appears in what
    the LLM sees, there's nothing for it to preserve or reproduce."""
    return re.sub(rf"{re.escape(heading)}.*?(?=\n### |\n---|\Z)", "", content, flags=re.DOTALL)


def _render_events_table(rows: list[dict]) -> str:
    header = "### Recent events\n| Date | Type | Event | Signal | Source |\n|---|---|---|---|---|"
    if not rows:
        return header + "\n"
    sorted_rows = sorted(rows, key=lambda r: r.get("date") or "", reverse=True)
    body = "\n".join(
        f"| {r.get('date', '')} | {r.get('type', '')} | {r.get('event', '')} | "
        f"{r.get('signal', '')} | {r.get('source', '')} |"
        for r in sorted_rows
    )
    return f"{header}\n{body}\n"


def _splice_events_section(content: str, table_md: str) -> str:
    """Strip any '### Recent events' block the LLM wrote anyway (in case it
    ignored the instruction not to), then insert the canonical, Python-rendered
    table right before '### Sources' — or at the end if no Sources section."""
    content = re.sub(r"### Recent events.*?(?=\n### |\n---|\Z)", "", content, flags=re.DOTALL)
    sources_idx = content.find("### Sources")
    if sources_idx == -1:
        return content.rstrip() + "\n\n" + table_md + "\n"
    return content[:sources_idx] + table_md + "\n" + content[sources_idx:]


# ── company "Earnings intelligence" — append-only log, never reproduced by the LLM ─
# Same root fix as the events table above, applied to the section that actually
# grows unboundedly on company pages: one paragraph per financial filing (8-K,
# 10-Q), forever. Each new paragraph is its own standalone LLM call (see the
# "earnings_para" task type in flush_buffered_pages) — never asked to
# reproduce prior quarters, and no delimiter/count-matching needed since each
# call returns exactly one paragraph.

_EARNINGS_DOC_TYPES = {"edgar_8k", "edgar_10q"}


def _render_earnings_section(rows: list[dict]) -> str:
    header = "### Earnings intelligence"
    if not rows:
        return header + "\n"
    sorted_rows = sorted(rows, key=lambda r: r.get("date") or "", reverse=True)
    body = "\n\n".join(r.get("text", "") for r in sorted_rows if r.get("text"))
    return f"{header}\n{body}\n"


def _splice_earnings_section(content: str, section_md: str) -> str:
    """Strip any '### Earnings intelligence' block the LLM wrote anyway, then
    insert the canonical, Python-rendered section in its place — or before
    '### Pipeline' if the LLM omitted the heading, or at the end as a fallback."""
    content = re.sub(r"### Earnings intelligence.*?(?=\n### |\n---|\Z)", "", content, flags=re.DOTALL)
    pipeline_idx = content.find("### Pipeline")
    if pipeline_idx == -1:
        return content.rstrip() + "\n\n" + section_md + "\n"
    return content[:pipeline_idx] + section_md + "\n" + content[pipeline_idx:]


# ── drug "Timeline" table — rendered deterministically, never by the LLM ────
# Same append-forever problem as the company events table: one row per trial/
# event signal touching this drug, accumulating indefinitely. No LLM call is
# needed at all here (same as company events) — every field the table needs
# already lives in the extracted signal.

def _build_drug_timeline_row(signal: dict) -> dict | None:
    extracted = signal.get("extracted", {})
    event_text = extracted.get("event_summary")
    if not event_text:
        return None
    slug = extracted.get("suggested_event_slug")
    type_label = f"[[{slug}]]" if slug else _EVENT_DOC_TYPE_LABELS.get(signal.get("doc_type", ""), "")
    return {
        "date":      extracted.get("event_date") or "",
        "event":     event_text,
        "type":      type_label,
        "file_path": signal.get("file_path", ""),
    }


def _render_drug_timeline_table(rows: list[dict]) -> str:
    header = "### Timeline\n| Date | Event | Type |\n|---|---|---|"
    if not rows:
        return header + "\n"
    sorted_rows = sorted(rows, key=lambda r: r.get("date") or "", reverse=True)
    body = "\n".join(
        f"| {r.get('date', '')} | {r.get('event', '')} | {r.get('type', '')} |"
        for r in sorted_rows
    )
    return f"{header}\n{body}\n"


def _splice_drug_timeline_section(content: str, table_md: str) -> str:
    content = re.sub(r"### Timeline.*?(?=\n### |\n---|\Z)", "", content, flags=re.DOTALL)
    sources_idx = content.find("### Sources")
    if sources_idx == -1:
        return content.rstrip() + "\n\n" + table_md + "\n"
    return content[:sources_idx] + table_md + "\n" + content[sources_idx:]


# ── drug "Management sentiment" — append-only log, never reproduced by the LLM ─
# Confirmed in testing: semaglutide's page hit 170KB / MAX_TOKENS because this
# section accumulated one rolling paragraph stitched together from every
# earnings filing ever processed, reproduced in full on every flush — same
# growth pattern as company earnings intelligence. Unlike earnings/clinical-
# evidence, this needs NO extra LLM call at all: DrugMention.commentary
# ("one sentence from management or results") is already written during Step 1
# extraction for exactly this purpose — Python just has to find the right
# DrugMention and store it.

def _build_drug_sentiment_row(signal: dict, drug_slug: str) -> dict | None:
    extracted = signal.get("extracted", {})
    if signal.get("doc_type") not in _EARNINGS_DOC_TYPES or extracted.get("event_type") != "earnings_signal":
        return None
    mention = next(
        (d for d in extracted.get("all_drugs_mentioned", []) if d.get("name") == drug_slug and d.get("commentary")),
        None,
    )
    if not mention:
        return None
    return {
        "date":      extracted.get("event_date") or "",
        "text":      mention["commentary"],
        "file_path": signal.get("file_path", ""),
    }


def _render_drug_sentiment_section(rows: list[dict]) -> str:
    header = "### Management sentiment"
    if not rows:
        return header + "\n"
    sorted_rows = sorted(rows, key=lambda r: r.get("date") or "", reverse=True)
    body = "\n\n".join(r.get("text", "") for r in sorted_rows if r.get("text"))
    return f"{header}\n{body}\n"


def _splice_drug_sentiment_section(content: str, section_md: str) -> str:
    """Strip any '### Management sentiment' block the LLM wrote anyway, then
    insert the canonical section before '### Clinical evidence' (falling back
    to the same anchor chain as the clinical-evidence splice)."""
    content = re.sub(r"### Management sentiment.*?(?=\n### |\n---|\Z)", "", content, flags=re.DOTALL)
    for anchor in ("### Clinical evidence", "### Competitive position", "### Timeline", "### Sources"):
        idx = content.find(anchor)
        if idx != -1:
            return content[:idx] + section_md + "\n" + content[idx:]
    return content.rstrip() + "\n\n" + section_md + "\n"


# ── drug "Clinical evidence" — append-only log, never reproduced by the LLM ──
# Same fix as company earnings intelligence: one paragraph per pubmed/clinical
# finding, forever. The LLM is only ever asked for the NEW paragraph(s).

def _render_drug_clinical_evidence_section(rows: list[dict]) -> str:
    header = "### Clinical evidence"
    if not rows:
        return header + "\n"
    sorted_rows = sorted(rows, key=lambda r: r.get("date") or "", reverse=True)
    body = "\n\n".join(r.get("text", "") for r in sorted_rows if r.get("text"))
    return f"{header}\n{body}\n"


def _splice_drug_clinical_evidence_section(content: str, section_md: str) -> str:
    """Strip any '### Clinical evidence' block the LLM wrote anyway, then
    insert the canonical section before '### Competitive position' (falling
    back to '### Timeline', then '### Sources', then end of page)."""
    content = re.sub(r"### Clinical evidence.*?(?=\n### |\n---|\Z)", "", content, flags=re.DOTALL)
    for anchor in ("### Competitive position", "### Timeline", "### Sources"):
        idx = content.find(anchor)
        if idx != -1:
            return content[:idx] + section_md + "\n" + content[idx:]
    return content.rstrip() + "\n\n" + section_md + "\n"


# ── indication hub "Recent events" table — rendered deterministically, never by the LLM ─
# Same problem at the widest scope: one row per signal across EVERY company/
# drug in the indication, accumulating indefinitely (oncology-nsclc was 1334
# lines, the largest page in the wiki by a wide margin).

def _build_indication_event_row(signal: dict) -> dict | None:
    extracted = signal.get("extracted", {})
    event_text = extracted.get("event_summary")
    if not event_text:
        return None
    return {
        "date":      extracted.get("event_date") or "",
        "event":     event_text,
        "signal":    _SIGNAL_LABELS.get(extracted.get("sentiment"), ""),
        "file_path": signal.get("file_path", ""),
    }


def _render_indication_events_table(rows: list[dict]) -> str:
    header = "### Recent events\n| Date | Event | Signal |\n|---|---|---|"
    if not rows:
        return header + "\n"
    sorted_rows = sorted(rows, key=lambda r: r.get("date") or "", reverse=True)
    body = "\n".join(
        f"| {r.get('date', '')} | {r.get('event', '')} | {r.get('signal', '')} |"
        for r in sorted_rows
    )
    return f"{header}\n{body}\n"


def _splice_indication_events_section(content: str, table_md: str) -> str:
    content = re.sub(r"### Recent events.*?(?=\n### |\n---|\Z)", "", content, flags=re.DOTALL)
    anchor_idx = content.find("### Active trials")
    if anchor_idx == -1:
        return content.rstrip() + "\n\n" + table_md + "\n"
    return content[:anchor_idx] + table_md + "\n" + content[anchor_idx:]


# ── trial registry "Summary" table — rendered deterministically, never by the LLM ─
# Every field the table needs (NCT ID, title, phase, status, completion date,
# results) already lives verbatim in each trial's own YAML frontmatter block,
# parsed out by _parse_trial_page() below.


def _render_trial_summary_table(blocks: list[dict]) -> str:
    header = (
        "## Summary\n\n"
        "| NCT ID | Title | Phase | Status | Primary Completion | Results |\n"
        "|---|---|---|---|---|---|"
    )
    if not blocks:
        return header + "\n"
    sorted_blocks = sorted(blocks, key=lambda b: str(b.get("trial_id") or ""))
    rows = "\n".join(
        f"| [[{b.get('trial_id', '')}]] | {b.get('title', '')} | {b.get('phase', '')} | "
        f"{b.get('status', '')} | {b.get('primary_completion_date') or ''} | "
        f"{'Yes' if b.get('has_results') else 'No'} |"
        for b in sorted_blocks
    )
    return f"{header}\n{rows}\n"


# ── trial pages — per-trial block regeneration ───────────────────────────────
# The page-level rewrite (send the whole page, ask for the whole page back) is
# what blew the output-token cap for AstraZeneca (583 trials) and Sanofi (385
# trials) — response size scaled with total trial count, not with what this
# batch actually changed. Below: each trial's frontmatter+body is parsed out,
# kept verbatim if untouched this batch, and only regenerated (small, bounded
# LLM call) for trials with an actual signal this run. The page is then
# reassembled in Python, never echoed back by the LLM as a whole.

_TRIAL_BLOCK_DELIM = "===NEXT_TRIAL==="


def _strip_trial_structural_noise(body: str) -> str:
    """Remove stray '### NCT...' / '## Active|Completed|Terminated Trials'
    headings that legacy pages sometimes carry inside a trial's body text —
    grouping and trial headings are always re-derived/re-rendered fresh,
    never trusted from page layout."""
    body = re.sub(r"^### NCT\d+.*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"^## (Active|Completed|Terminated) Trials\s*$", "", body, flags=re.MULTILINE)
    return body


def _trial_group(meta: dict) -> str:
    """Status group is derived fresh from the trial's own `status` field every
    render — not from whatever section it happened to sit under on the page
    before. A trial that moved from active to completed self-corrects."""
    status = str(meta.get("status") or "").strip().lower()
    if status in ("terminated", "withdrawn"):
        return "Terminated Trials"
    if status == "completed":
        return "Completed Trials"
    return "Active Trials"


def _extract_sources_lines(block: str) -> list[str]:
    """Pull the '- `path`' bullet lines out of a trial block's '### Sources'
    section, so they can be carried forward when a trial is regenerated."""
    match = re.search(r"### Sources\s*\n((?:- .+\n?)*)", block)
    if not match:
        return []
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def _parse_trial_page(content: str) -> dict[str, dict]:
    """Parse an existing trials/{company}.md page into nct_id -> {meta, block}.
    `block` is the trial's frontmatter + body, kept byte-for-byte except for
    stray structural headings — untouched trials are carried forward exactly
    as they were, never round-tripped through the LLM."""
    chunks = re.split(r"^---$", content, flags=re.MULTILINE)
    blocks: dict[str, dict] = {}
    i = 1
    while i + 1 < len(chunks):
        yaml_text, body_text = chunks[i], chunks[i + 1]
        i += 2
        try:
            meta = yaml.safe_load(yaml_text.strip())
        except yaml.YAMLError:
            continue
        if not isinstance(meta, dict) or not meta.get("trial_id"):
            continue
        body_text = _strip_trial_structural_noise(body_text).strip()
        block = f"---\n{yaml_text.strip()}\n---\n\n{body_text}\n"
        blocks[str(meta["trial_id"])] = {"meta": meta, "block": block}

    # Legacy pages have mixed conventions (some trial blocks fenced with
    # ```yaml instead of bare --- delimiters) that this parser can't recover —
    # those blocks are silently dropped from `blocks` rather than raising, so
    # surface the gap loudly instead of letting content vanish unnoticed.
    raw_trial_id_count = len(re.findall(r"^\s*trial_id:\s*\S", content, flags=re.MULTILINE))
    if raw_trial_id_count > len(blocks):
        logger.warning(
            f"_parse_trial_page | found {raw_trial_id_count} 'trial_id:' occurrences "
            f"but only parsed {len(blocks)} valid blocks — {raw_trial_id_count - len(blocks)} "
            f"trial(s) are in an unrecognized format and will be dropped if this page is "
            f"rewritten without every affected NCT ID being regenerated this batch."
        )
    return blocks


def _count_distinct_ncts(text: str) -> int:
    """Count distinct NCT IDs anywhere in raw page text, independent of
    formatting convention. Used as a format-agnostic backstop: _parse_trial_page()
    only recovers trials written in the current `---`-delimited convention, but
    real pages exist in several legacy formats (yaml-fenced blocks, a flat
    markdown table with no per-trial detail at all — found in production
    AstraZeneca/Sanofi pages during testing). If a legacy-format page isn't
    fully recoverable by the parser, this catches it before a rewrite would
    silently drop the unparsed trials."""
    return len(set(re.findall(r"NCT\d{8}", text)))


def _render_trial_page(company_slug: str, blocks: dict[str, dict]) -> str:
    """Reassemble the full trials/{company}.md page from per-trial blocks —
    H1, the (already-deterministic) Summary table, then each block grouped by
    status. Output size here is purely a function of total trial count, with
    zero LLM involvement — the LLM never sees or produces the whole page."""
    aliases = COMPANIES.get(company_slug, {}).get("aliases", [])
    company_label = aliases[0] if aliases else company_slug
    metas = [b["meta"] for b in blocks.values()]
    summary = _render_trial_summary_table(metas)

    groups: dict[str, list[str]] = {"Active Trials": [], "Completed Trials": [], "Terminated Trials": []}
    for nct_id in sorted(blocks):
        b = blocks[nct_id]
        groups[_trial_group(b["meta"])].append(b["block"])

    sections = [f"# {company_label} Clinical Trials", summary]
    for group_name in ("Active Trials", "Completed Trials", "Terminated Trials"):
        if not groups[group_name]:
            continue
        sections.append(f"## {group_name}\n\n" + "\n".join(groups[group_name]))
    return "\n\n".join(sections) + "\n"


def flush_buffered_pages(
    company_buffer: dict,
    trial_buffer: dict,
    drug_buffer: dict,
    indication_buffer: dict,
    template_caches: dict,
) -> tuple[list[str], set[str]]:
    """Flush all buffered signals — one LLM call per entity covering all accumulated signals.
    Handles company, trial, drug, and indication pages in a single parallel pass.
    Call once after the per-file loop. Returns (succeeded_page_paths, failed_page_paths) —
    callers must use failed_page_paths to avoid marking a file as processed when the
    specific page(s) its signals fed into didn't actually get written (truncation,
    transient API error, etc.) even if other unrelated pages in the same batch succeeded.
    """
    import concurrent.futures as _cf
    from agents.state import update_index_py

    if not any([company_buffer, trial_buffer, drug_buffer, indication_buffer]):
        return [], set()

    system_prompt = build_system_prompt()
    tasks: list[tuple] = []

    # trial-only bookkeeping for the per-block (not per-page) regeneration —
    # keyed by page_path, consumed in the result loop below.
    trial_existing_blocks: dict[str, dict[str, dict]] = {}
    # ordered_ncts is now a FIFO queue of chunks, one entry per LLM call for
    # that page — chunking (see below) means several "trial" tasks/results can
    # share one page_path, processed/consumed in the same order they were created.
    trial_ordered_ncts:    dict[str, list[list[str]]] = {}
    trial_file_paths:      dict[str, dict[str, list[str]]] = {}
    trial_raw_current:     dict[str, str] = {}
    trial_chunks_remaining: dict[str, int] = {}

    def _signals_block(signals: list[dict]) -> str:
        return "\n\n".join(
            f"Signal {i + 1} (file: {s['file_path']}, type: {s['doc_type']}):\n"
            f"{json.dumps(s['extracted'], indent=2)}"
            for i, s in enumerate(signals)
        )

    def _chunk_list(items: list, size: int) -> list[list]:
        return [items[i:i + size] for i in range(0, len(items), size)]

    # Confirmed in production (AstraZeneca/Sanofi backlog catch-up): a company
    # or drug with hundreds of signals in one batch causes the SAME failure
    # the trial-page fix already solved, one level up — the main page-update
    # call embeds every signal's full JSON in one prompt, making the call slow
    # enough to get cancelled (499) or large enough to risk MAX_TOKENS on
    # output. >1 chunk triggers sequential multi-call processing for that
    # entity instead of one call asked to integrate everything at once.
    ENTITY_SIGNAL_CHUNK_SIZE = 20

    # Smaller chunks for very large batches — confirmed in production that a
    # 20-signal chunk can still be too slow/large for an entity already deep
    # into a huge batch (AstraZeneca: 558 signals, 28 chunks of 20 — chunk
    # 8 alone hit MAX_TOKENS). Smaller, more numerous calls are individually
    # faster and lighter, even though it doesn't change the total signal count.
    LARGE_BATCH_THRESHOLD = 100
    LARGE_BATCH_CHUNK_SIZE = 10

    def _entity_chunk_size(signal_count: int) -> int:
        return LARGE_BATCH_CHUNK_SIZE if signal_count > LARGE_BATCH_THRESHOLD else ENTITY_SIGNAL_CHUNK_SIZE

    # Smaller chunks alone don't fix AstraZeneca-style failures, because the
    # thing that actually grows across the sequential chain is the page body
    # itself (current page content carried from each chunk's output into the
    # next chunk's input + "preserve everything" instruction) — confirmed in
    # production: chunk 8/28 (not chunk 1) is what hit MAX_TOKENS, meaning the
    # accumulated body, not any single chunk's signal count, was the cause.
    # ~10K tokens of body content still leaves ample room under the 65536
    # output ceiling for that body to be reproduced plus genuinely new content
    # — if it's already bigger than that, finishing this chunk is unlikely to
    # succeed, so stop BEFORE attempting a call we can predict will fail
    # rather than discovering it after burning the API call.
    MAX_CURRENT_CHARS_FOR_CHUNKING = 40_000

    # Gemini 2.5 Flash's documented max — pages for companies with hundreds of
    # accumulated trials can get close to default output limits, so set this
    # explicitly rather than trusting an SDK default.
    MAX_OUTPUT_TOKENS = 65536

    def _make_config(page_type: str) -> types.GenerateContentConfig:
        cache_name = template_caches.get(page_type)
        if cache_name:
            return types.GenerateContentConfig(
                cached_content=cache_name, temperature=0.2, max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        page_template = load_page_template(page_type)
        return types.GenerateContentConfig(
            system_instruction=system_prompt + "\n\n" + page_template,
            temperature=0.2, max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def _chunk_note(chunk_index: int, total_chunks: int) -> str:
        if total_chunks <= 1:
            return ""
        return (
            f"\nNOTE: this is part {chunk_index + 1} of {total_chunks} for this batch — "
            f"earlier parts already integrated earlier signals from this same batch into "
            f"the \"current page content\" below. Integrate ONLY the signals listed here.\n"
        )

    def _build_company_prompt(page_path: str, slug: str, current: str, chunk_signals: list[dict],
                               chunk_index: int, total_chunks: int) -> str:
        return f"""Update the company wiki page for: {page_path}
Page type: company
Entity: {slug}
{_chunk_note(chunk_index, total_chunks)}
{len(chunk_signals)} document(s) processed in this batch run.

Current page content (empty if new page):
---
{current if current else '[NEW PAGE — write from scratch using the template above]'}
---

Signals to integrate (ordered as processed):
{_signals_block(chunk_signals)}

Rules:
- Integrate ALL signals above into the company page
- Follow the company template exactly
- Preserve all existing content — only add or update what these documents affect
- Do NOT write a "### Recent events" section or table — it is generated
  programmatically from canonical signal data after this call, not by you.
- Do NOT write a "### Earnings intelligence" section — it is generated
  programmatically from an append-only log after this call, not by you.
  Write everything else on the page as normal.
- For all_drugs_mentioned: drugs with is_tracked=true get [[drug_name]] links; is_tracked=false as plain text
- last_updated: use the most recent event_date from the signals above
- Write ONLY the markdown content — no preamble, no explanation
"""

    def _build_drug_prompt(page_path: str, drug_slug: str, current: str, chunk_signals: list[dict],
                            chunk_index: int, total_chunks: int) -> str:
        return f"""Update the drug wiki page for: {page_path}
Page type: drug
Entity: {drug_slug}
{_chunk_note(chunk_index, total_chunks)}
{len(chunk_signals)} document(s) processed in this batch run.

Current page content (empty if new page):
---
{current if current else '[NEW PAGE — write from scratch using the template above]'}
---

Signals to integrate (ordered as processed):
{_signals_block(chunk_signals)}

Rules:
- Integrate ALL signals above into the drug page
- Follow the drug template exactly
- Preserve all existing content — only add or update what these documents affect
- Do NOT write a "### Timeline" section or table — it is generated
  programmatically from canonical signal data after this call, not by you.
- Do NOT write a "### Management sentiment" section — it is generated
  programmatically from canonical signal data after this call, not by you.
- Do NOT write a "### Clinical evidence" section — it is generated
  programmatically from an append-only log after this call, not by you.
  Write everything else on the page as normal.
- End with a ## Sources section listing all raw file paths
- Write ONLY the markdown content — no preamble, no explanation
"""

    def _build_indication_prompt(page_path: str, ind_slug: str, current: str, chunk_signals: list[dict],
                                  chunk_index: int, total_chunks: int) -> str:
        return f"""Update the indication hub page for: {page_path}
Page type: indication_hub
Entity: {ind_slug}
{_chunk_note(chunk_index, total_chunks)}
{len(chunk_signals)} document(s) processed in this batch run.

Current page content (empty if new page):
---
{current if current else '[NEW PAGE — write from scratch using the template above]'}
---

Signals to integrate (ordered as processed):
{_signals_block(chunk_signals)}

Rules:
- Integrate ALL signals above into the indication hub page
- Follow the indication_hub template exactly
- Preserve all existing content — only add or update what these documents affect
- Do NOT write a "### Recent events" section or table — it is generated
  programmatically from canonical signal data after this call, not by you.
  Write everything else on the page as normal.
- End with a ## Sources section listing all raw file paths
- Write ONLY the markdown content — no preamble, no explanation
"""

    # ── company ────────────────────────────────────────────────────────────────
    for slug, signals in company_buffer.items():
        page_path = f"companies/{slug}.md"
        current   = read_wiki_page(page_path)
        # Strip deterministic sections from what the LLM SEES, not just from
        # what it's told to avoid writing — see _strip_section_for_prompt.
        for heading in ("### Recent events", "### Earnings intelligence"):
            current = _strip_section_for_prompt(current, heading)

        # doc_type alone over-matches — most 8-K/6-K filings are FDA approvals,
        # share-sale notices, shareholder-threshold crossings, etc., not earnings
        # reports (confirmed in testing: AstraZeneca had 51 edgar_8k signals in
        # one batch, only 2 were actual earnings calls). event_type narrows to
        # what the extraction step itself already classified as guidance/mgmt
        # commentary, so the ask isn't sent for filings with nothing to summarize.
        earnings_signals = [
            s for s in signals
            if s.get("doc_type") in _EARNINGS_DOC_TYPES and s["extracted"].get("event_type") == "earnings_signal"
        ]
        # One standalone, narrow call per finding instead of "give me exactly N
        # delimited paragraphs in this one call" — confirmed in testing that the
        # multi-paragraph ask doesn't reliably return the right count. A 1:1
        # call has nothing to miscount: the whole response IS the paragraph.
        for s in earnings_signals:
            para_prompt = f"""Write ONE 1-3 sentence paragraph summarizing this financial filing's
key guidance/financial commentary, in the style of company earnings intelligence.

Filing data:
{json.dumps(s["extracted"], indent=2)}

Output ONLY the paragraph text — no heading, no preamble, no markdown fences."""
            tasks.append(("earnings_para", slug, page_path, para_prompt, _make_config("company"), [s]))

        signal_chunks = _chunk_list(signals, _entity_chunk_size(len(signals)))
        if len(signal_chunks) <= 1:
            prompt = _build_company_prompt(page_path, slug, current, signals, 0, 1)
            tasks.append(("company", slug, page_path, prompt, _make_config("company"), signals))
        else:
            # Too many signals for one call (confirmed in production: AstraZeneca/
            # Sanofi with hundreds of signals hit 499 CANCELLED) — process
            # sequentially, each chunk building on the previous chunk's output.
            payload = {"current": current, "chunks": signal_chunks, "builder": _build_company_prompt}
            tasks.append(("company_chunked", slug, page_path, payload, _make_config("company"), signals))

    # ── trial ──────────────────────────────────────────────────────────────────
    # Only the trials touched THIS batch are sent to/from the LLM — the rest of
    # the page (potentially hundreds of trials) is carried forward verbatim in
    # Python via _render_trial_page(). This is what actually fixes the
    # AstraZeneca/Sanofi output-token failures: response size now scales with
    # batch size, never with total trial-page history.
    # A company with a deep backlog can have 100+ trials touched in a single
    # batch (confirmed in testing: AstraZeneca hit MAX_TOKENS at 106 trials in
    # one call). Capping each LLM call to a small chunk keeps every individual
    # response well under the output ceiling regardless of how large the batch
    # or backlog is — chunks fire as separate parallel tasks, merged into the
    # same page before a single write at the end.
    TRIAL_CHUNK_SIZE = 15

    for sponsor, entries in trial_buffer.items():
        page_path = f"trials/{sponsor}.md"
        current   = read_wiki_page(page_path)
        existing_blocks = _parse_trial_page(current)
        trial_existing_blocks[page_path] = existing_blocks
        trial_raw_current[page_path] = current

        # group entries by nct_id — a trial can appear more than once in a
        # batch (e.g. multiple files referencing it); later entries win for
        # action/extracted, all of their file paths are kept for Sources.
        nct_groups: dict[str, dict] = {}
        ordered_ncts: list[str] = []
        for e in entries:
            nct_id = e["nct_id"]
            if nct_id not in nct_groups:
                ordered_ncts.append(nct_id)
                nct_groups[nct_id] = {"action": e["action"], "file_paths": [], "extracted": []}
            nct_groups[nct_id]["action"] = e["action"]
            nct_groups[nct_id]["file_paths"].append(e["file_path"])
            nct_groups[nct_id]["extracted"].append(e["extracted"])

        trial_file_paths[page_path] = {n: g["file_paths"] for n, g in nct_groups.items()}

        nct_chunks = [
            ordered_ncts[i:i + TRIAL_CHUNK_SIZE]
            for i in range(0, len(ordered_ncts), TRIAL_CHUNK_SIZE)
        ]
        trial_ordered_ncts[page_path] = list(nct_chunks)  # FIFO queue, consumed in post-processing
        trial_chunks_remaining[page_path] = len(nct_chunks)

        for chunk_ncts in nct_chunks:
            trial_prompts = []
            for i, nct_id in enumerate(chunk_ncts):
                group = nct_groups[nct_id]
                old_block = existing_blocks.get(nct_id, {}).get("block", "")
                context_block = (
                    f"Existing block for this trial (update it, don't just copy it verbatim):\n{old_block}"
                    if group["action"] == "update" and old_block
                    else "This is a new trial — write its block from scratch using the template."
                )
                signals_text = "\n\n".join(
                    f"Signal: {json.dumps(ext, indent=2)}" for ext in group["extracted"]
                )
                trial_prompts.append(
                    f"Trial {i + 1}: {nct_id} — action: {group['action']}\n"
                    f"{context_block}\n\n{signals_text}"
                )

            prompt = f"""Generate ONLY the per-trial frontmatter+body blocks for the {len(chunk_ncts)}
trial(s) below — nothing else on the page. Do NOT write the page title, the
Summary table, "## Active/Completed/Terminated Trials" group headings, or any
other trial. Do NOT write a "### Sources" section — it is generated
programmatically from the source file path(s) after this call, not by you.

For each trial, output exactly one block in this format:
---
{{yaml frontmatter per the trial template}}
---

## {{Trial title}}

**Phase:** {{N}} | **Status:** {{status}} | **Sponsor:** [[{sponsor}]]

### Design
{{...}}

### Primary endpoint
{{...}}

### Results summary
{{...}}

Separate consecutive trial blocks with a line containing exactly:
{_TRIAL_BLOCK_DELIM}

Output the {len(chunk_ncts)} blocks in this exact order: {chunk_ncts}

{chr(10).join(trial_prompts)}

Follow the trial template exactly for frontmatter fields.
Write ONLY the blocks and the delimiter — no preamble, no explanation.
"""
            tasks.append(("trial", sponsor, page_path, prompt, _make_config("trial"), entries))

    # ── drug ───────────────────────────────────────────────────────────────────
    for drug_slug, signals in drug_buffer.items():
        page_path = f"drugs/{drug_slug}.md"
        current   = read_wiki_page(page_path)
        # Strip deterministic sections from what the LLM SEES — confirmed in
        # testing this is what actually caused semaglutide's repeated
        # MAX_TOKENS (its 170KB Clinical evidence section was still being
        # shown as "current page content" even though the LLM was told not
        # to write it; "leave it exactly where it is" pushed it to reproduce
        # the whole thing in the response instead). See _strip_section_for_prompt.
        for heading in ("### Timeline", "### Management sentiment", "### Clinical evidence"):
            current = _strip_section_for_prompt(current, heading)

        clinical_signals = [s for s in signals if _clinical_finding_matches_drug(s["extracted"], drug_slug)]
        # Same fix as earnings: one standalone call per finding instead of one
        # call asked for N delimited paragraphs at once — eliminates the count
        # mismatch by construction (response IS the paragraph, nothing to count).
        for s in clinical_signals:
            para_prompt = f"""Write ONE paragraph summarizing this study's key finding, effect
size, and design, for the Clinical evidence section of a drug wiki page.

Finding data:
{json.dumps(s["extracted"]["clinical_findings"], indent=2)}

Output ONLY the paragraph text — no heading, no preamble, no markdown fences."""
            tasks.append(("clinical_para", drug_slug, page_path, para_prompt, _make_config("drug"), [s]))

        signal_chunks = _chunk_list(signals, _entity_chunk_size(len(signals)))
        if len(signal_chunks) <= 1:
            prompt = _build_drug_prompt(page_path, drug_slug, current, signals, 0, 1)
            tasks.append(("drug", drug_slug, page_path, prompt, _make_config("drug"), signals))
        else:
            # Same fix as company pages — too many signals for one call
            # (confirmed in production: dapagliflozin hit MAX_TOKENS).
            payload = {"current": current, "chunks": signal_chunks, "builder": _build_drug_prompt}
            tasks.append(("drug_chunked", drug_slug, page_path, payload, _make_config("drug"), signals))

    # ── indication ─────────────────────────────────────────────────────────────
    # "Recent events" is the unbounded section here (one row per signal across
    # every company/drug in the indication) — rendered deterministically from
    # canonical signal data, same as company/drug events. No LLM call needed
    # for it at all; the rest of the page (small, fixed-size tables) is still
    # LLM-maintained since it isn't what grows unboundedly.
    for ind_slug, signals in indication_buffer.items():
        page_path = f"indications/{ind_slug}/_index.md"
        current   = read_wiki_page(page_path)
        current = _strip_section_for_prompt(current, "### Recent events")

        signal_chunks = _chunk_list(signals, _entity_chunk_size(len(signals)))
        if len(signal_chunks) <= 1:
            prompt = _build_indication_prompt(page_path, ind_slug, current, signals, 0, 1)
            tasks.append(("indication_hub", ind_slug, page_path, prompt, _make_config("indication_hub"), signals))
        else:
            # Same fix as company/drug pages — indication hubs aggregate
            # signals across every company/drug in that indication, so they're
            # exposed to the identical too-many-signals-in-one-call problem
            # (confirmed in production: oncology-nsclc hit MAX_TOKENS).
            payload = {"current": current, "chunks": signal_chunks, "builder": _build_indication_prompt}
            tasks.append(("indication_hub_chunked", ind_slug, page_path, payload, _make_config("indication_hub"), signals))

    # Flush bursts every buffered page-write call at once. At 20-24 concurrent
    # calls (confirmed in testing — 3x 429 RESOURCE_EXHAUSTED on a 24-task
    # flush), bursts can exceed Vertex AI's per-minute request/token quota.
    # Lower concurrency reduces how often that ceiling gets hit; retry-with-
    # backoff recovers the rest within the same run instead of deferring
    # every quota hit to the next pipeline run.
    FLUSH_MAX_WORKERS = 8
    MAX_429_RETRIES = 4

    def _generate_once(prompt: str, config, label: str) -> tuple[str | None, object | None]:
        """One LLM call with 429 retry/backoff and finish_reason validation.
        Returns (text, usage) on success, (None, usage_or_None) on failure —
        caller decides what to do next (single-call tasks give up; chunked
        tasks abort the whole entity, same conservative policy as trial chunks)."""
        attempt = 0
        while True:
            try:
                resp = client.models.generate_content(model=FLASH_MODEL, contents=prompt, config=config)
                finish_reason = resp.candidates[0].finish_reason if resp.candidates else None
                # STOP is the only "completed normally" reason. Anything else
                # (MAX_TOKENS, SAFETY, RECITATION, OTHER...) means resp.text is a
                # partial/incomplete document — never write it.
                if finish_reason is not None and not str(finish_reason).endswith("STOP"):
                    logger.error(
                        f"FLUSH | INCOMPLETE | {label} — finish_reason={finish_reason}, "
                        f"discarding partial content rather than writing a broken page."
                    )
                    return None, resp.usage_metadata
                return resp.text, resp.usage_metadata
            except genai_errors.ClientError as exc:
                if exc.code == 429 and attempt < MAX_429_RETRIES:
                    attempt += 1
                    delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                    logger.warning(f"FLUSH | 429 RATE LIMITED | {label} — retry {attempt}/{MAX_429_RETRIES} in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                logger.error(f"FLUSH | ERROR | {label}: {exc}")
                return None, None
            except Exception as exc:
                logger.error(f"FLUSH | ERROR | {label}: {exc}")
                return None, None

    def _call_llm_chunked(base_type: str, entity: str, page_path: str, payload: dict, config, signals) -> tuple | None:
        """Sequentially integrate each signal chunk into the page, each call
        building on the previous call's output — unlike trial chunks (which
        are independent units merged afterward), company/drug body sections
        are one shared narrative that every chunk must build on consistently."""
        current = payload["current"]
        chunks  = payload["chunks"]
        builder = payload["builder"]
        start = time.time()
        for i, chunk_signals in enumerate(chunks):
            if len(current) > MAX_CURRENT_CHARS_FOR_CHUNKING:
                logger.error(
                    f"FLUSH | PAGE TOO LARGE TO CONTINUE CHUNKING | {base_type} {entity} — "
                    f"current page content reached {len(current)} chars after {i}/{len(chunks)} "
                    f"chunks, stopping before attempting a call likely to hit MAX_TOKENS. "
                    f"Aborting this entity; all contributing files retry next run."
                )
                return None
            prompt = builder(page_path, entity, current, chunk_signals, i, len(chunks))
            text, usage = _generate_once(prompt, config, f"{base_type} {entity} (part {i + 1}/{len(chunks)})")
            if usage:
                ledger.record(usage)
            if text is None:
                return None
            current = _strip_fenced_code_block(text)
            if base_type == "company":
                strip_headings = ("### Recent events", "### Earnings intelligence")
            elif base_type == "indication_hub":
                strip_headings = ("### Recent events",)
            else:
                strip_headings = ("### Timeline", "### Management sentiment", "### Clinical evidence")
            for heading in strip_headings:
                current = _strip_section_for_prompt(current, heading)
        return base_type, entity, page_path, current, None, time.time() - start, signals

    def _call_llm(task: tuple) -> tuple | None:
        page_type, entity, page_path, prompt_or_payload, config, signals = task
        if page_type.endswith("_chunked"):
            return _call_llm_chunked(page_type.removesuffix("_chunked"), entity, page_path, prompt_or_payload, config, signals)
        start = time.time()
        text, usage = _generate_once(prompt_or_payload, config, f"{page_type} {entity}")
        if text is None:
            return None
        return page_type, entity, page_path, text, usage, time.time() - start, signals

    logger.info(
        f"FLUSH | Firing {len(tasks)} page writes (max {FLUSH_MAX_WORKERS} concurrent) — "
        f"company={len(company_buffer)}, trial={len(trial_buffer)}, "
        f"drug={len(drug_buffer)}, indication={len(indication_buffer)}"
    )
    # No pool.map timeout — the HTTP client already enforces 300s per call.
    # A global timeout here kills results from tasks that already finished.
    with _cf.ThreadPoolExecutor(max_workers=min(len(tasks), FLUSH_MAX_WORKERS)) as pool:
        results = list(pool.map(_call_llm, tasks))

    # pool.map preserves input order, so zipping tasks with results tells us
    # exactly which page_path failed — not just "something in this batch failed".
    failed_paths = {task[2] for task, result in zip(tasks, results) if result is None}

    # First pass: handle the standalone per-finding paragraph tasks
    # (earnings_para, clinical_para) — these just append one row each to the
    # canonical CSV logs. Must run BEFORE the main pass below, since the
    # company/drug page splice step reads those logs fresh and needs this
    # batch's new paragraphs already persisted.
    from agents.wiki_gcs import append_company_earnings, append_drug_clinical_evidence
    for result in results:
        if result is None:
            continue
        page_type, entity, page_path, content, usage, elapsed, signals = result
        if page_type not in ("earnings_para", "clinical_para"):
            continue
        ledger.record(usage)
        signal = signals[0]
        para = _strip_fenced_code_block(content).strip()
        if not para:
            logger.warning(f"FLUSH | EMPTY {page_type.upper()} | {entity} — file {signal['file_path']}, skipping")
            continue
        row = {"date": signal["extracted"].get("event_date") or "", "text": para, "file_path": signal["file_path"]}
        if page_type == "earnings_para":
            append_company_earnings(entity, [row])
        else:
            append_drug_clinical_evidence(entity, [row])

    updated_paths   = []
    pages_for_index = []
    for task, result in zip(tasks, results):
        if result is None:
            # A failed trial chunk must still pop the FIFO queue and decrement
            # the remaining-chunks counter — otherwise later, successful
            # chunks for the same page would desync and merge into the wrong
            # chunk's NCT list. The page_path is already in failed_paths via
            # the comprehension above, so it won't be written even if this
            # was the last chunk to complete.
            if task[0] == "trial":
                page_path = task[2]
                chunk_queue = trial_ordered_ncts.get(page_path, [])
                if chunk_queue:
                    chunk_queue.pop(0)
                trial_chunks_remaining[page_path] = trial_chunks_remaining.get(page_path, 1) - 1
            continue
        page_type, entity, page_path, content, usage, elapsed, signals = result
        if page_type in ("earnings_para", "clinical_para"):
            continue  # already handled in the first pass above
        n = len(signals)
        logger.info(f"FLUSH | {page_type.upper()} | {entity} — {n} signal(s), API took {elapsed:.1f}s")
        ledger.record(usage)
        if page_type == "company":
            from agents.wiki_gcs import append_company_events, read_company_events, read_company_earnings
            new_rows = [row for row in (_build_event_row(s) for s in signals) if row]
            if new_rows:
                append_company_events(entity, new_rows)
            content = _splice_events_section(content, _render_events_table(read_company_events(entity)))
            content = _splice_earnings_section(content, _render_earnings_section(read_company_earnings(entity)))
        elif page_type == "trial":
            blocks       = trial_existing_blocks.get(page_path, {})
            chunk_queue  = trial_ordered_ncts.get(page_path, [])
            ordered_ncts = chunk_queue.pop(0) if chunk_queue else []
            file_paths   = trial_file_paths.get(page_path, {})
            new_blocks_text = content.split(_TRIAL_BLOCK_DELIM)
            if len(new_blocks_text) != len(ordered_ncts):
                logger.warning(
                    f"FLUSH | TRIAL BLOCK COUNT MISMATCH | {entity} — expected "
                    f"{len(ordered_ncts)} blocks, got {len(new_blocks_text)}. "
                    f"Unmatched trials keep their previous content."
                )
            for nct_id, raw in zip(ordered_ncts, new_blocks_text):
                raw = _strip_fenced_code_block(raw).strip()
                sub_chunks = re.split(r"^---$", raw, flags=re.MULTILINE)
                if len(sub_chunks) < 3:
                    logger.warning(f"FLUSH | TRIAL BLOCK MALFORMED | {entity}/{nct_id} — skipping")
                    continue
                try:
                    meta = yaml.safe_load(sub_chunks[1].strip())
                except yaml.YAMLError:
                    logger.warning(f"FLUSH | TRIAL BLOCK YAML INVALID | {entity}/{nct_id} — skipping")
                    continue
                if not isinstance(meta, dict) or not meta.get("trial_id"):
                    logger.warning(f"FLUSH | TRIAL BLOCK NO trial_id | {entity}/{nct_id} — skipping")
                    continue
                try:
                    validated = TrialFrontmatter.model_validate(meta)
                except ValidationError as e:
                    logger.warning(f"FLUSH | TRIAL BLOCK SCHEMA INVALID | {entity}/{nct_id} — {e} — skipping")
                    continue
                # Write the *validated* (and therefore normalized — see
                # TrialFrontmatter._normalize_phase) model back out, not the
                # LLM's raw YAML text verbatim — otherwise field-level fixes
                # like phase-separator canonicalization only ever affect
                # pass/fail of validation, never the actual bytes on disk.
                meta = validated.model_dump(exclude_none=False)
                yaml_text = yaml.safe_dump(
                    meta, sort_keys=False, default_flow_style=None, allow_unicode=True, width=1000,
                ).strip()
                body = _strip_trial_structural_noise(sub_chunks[2])
                body = re.sub(r"### Sources.*", "", body, flags=re.DOTALL).strip()

                old_block = blocks.get(nct_id, {}).get("block", "")
                old_sources = _extract_sources_lines(old_block)
                new_source_lines = [f"- `{p}`" for p in file_paths.get(nct_id, [])]
                source_lines = new_source_lines + [l for l in old_sources if l not in new_source_lines]
                sources_block = "### Sources\n" + ("\n".join(source_lines) if source_lines else "")

                full_block = f"---\n{yaml_text}\n---\n\n{body}\n\n{sources_block}\n"
                blocks[nct_id] = {"meta": meta, "block": full_block}

            # Several chunks can share this page_path — only render/guard/write
            # once the LAST chunk for this page has been merged in. Earlier
            # chunks just update the shared `blocks` dict and move on.
            trial_chunks_remaining[page_path] = trial_chunks_remaining.get(page_path, 1) - 1
            if trial_chunks_remaining[page_path] > 0:
                continue

            content = _render_trial_page(entity, blocks)

            # Format-agnostic backstop: _parse_trial_page() only recovers the
            # current `---`-delimited convention. A legacy-format page (found
            # in testing: a flat markdown table with zero parseable blocks)
            # would otherwise have every trial it couldn't parse silently
            # dropped the moment any single trial on that page gets touched.
            # Never write a page with fewer distinct NCT IDs than it started
            # with — fail loud instead, same as a MAX_TOKENS/incomplete response.
            old_nct_count = _count_distinct_ncts(trial_raw_current.get(page_path, ""))
            new_nct_count = _count_distinct_ncts(content)
            if new_nct_count < old_nct_count:
                logger.error(
                    f"FLUSH | TRIAL PAGE WOULD LOSE TRIALS | {entity} — old page had "
                    f"{old_nct_count} distinct NCT IDs, rewrite would only have "
                    f"{new_nct_count}. Refusing to write; likely an unrecognized legacy "
                    f"page format. Skipping this page for this batch."
                )
                failed_paths.add(page_path)
                continue
        elif page_type == "drug":
            from agents.wiki_gcs import (
                append_drug_timeline, read_drug_timeline, read_drug_clinical_evidence,
                append_drug_management_sentiment, read_drug_management_sentiment,
            )
            new_timeline_rows = [row for row in (_build_drug_timeline_row(s) for s in signals) if row]
            if new_timeline_rows:
                append_drug_timeline(entity, new_timeline_rows)

            new_sentiment_rows = [
                row for row in (_build_drug_sentiment_row(s, entity) for s in signals) if row
            ]
            if new_sentiment_rows:
                append_drug_management_sentiment(entity, new_sentiment_rows)

            content = _splice_drug_timeline_section(content, _render_drug_timeline_table(read_drug_timeline(entity)))
            content = _splice_drug_sentiment_section(
                content, _render_drug_sentiment_section(read_drug_management_sentiment(entity)),
            )
            content = _splice_drug_clinical_evidence_section(
                content, _render_drug_clinical_evidence_section(read_drug_clinical_evidence(entity)),
            )
        elif page_type == "indication_hub":
            from agents.wiki_gcs import append_indication_events, read_indication_events
            new_rows = [row for row in (_build_indication_event_row(s) for s in signals) if row]
            if new_rows:
                append_indication_events(entity, new_rows)
            content = _splice_indication_events_section(
                content, _render_indication_events_table(read_indication_events(entity)),
            )

        # Whole-page frontmatter schema validation for the three page types
        # that have no per-block fallback (trial pages are already validated
        # per-NCT above and post-render-guarded against losing trials). A
        # malformed/missing frontmatter block here means the LLM produced a
        # broken page — skip the write entirely (keep old content) rather
        # than ship something the API can't parse, same fail-loud philosophy
        # as the trial-page-would-lose-trials guard above.
        _FRONTMATTER_MODELS = {
            "company":        CompanyFrontmatter,
            "drug":           DrugFrontmatter,
            "indication_hub": IndicationHubFrontmatter,
        }
        if page_type in _FRONTMATTER_MODELS:
            fm = _extract_frontmatter(content)
            if fm is None:
                logger.error(f"FLUSH | NO FRONTMATTER | {entity} ({page_type}) — refusing to write")
                failed_paths.add(page_path)
                continue
            try:
                _FRONTMATTER_MODELS[page_type].model_validate(fm)
            except ValidationError as e:
                logger.error(f"FLUSH | FRONTMATTER SCHEMA INVALID | {entity} ({page_type}) — {e} — refusing to write")
                failed_paths.add(page_path)
                continue

        try:
            write_wiki_page(page_path, content)
        except Exception:
            logger.error(f"FLUSH | WRITE FAILED | {page_path}", exc_info=True)
            failed_paths.add(page_path)
            continue
        updated_paths.append(page_path)
        logger.info(f"FLUSH | WROTE | {page_path}")
        all_drugs = (
            signals[-1]["extracted"].get("all_drugs_mentioned", [])
            if isinstance(signals[-1], dict) and "extracted" in signals[-1]
            else []
        )
        pages_for_index.append({
            "path":                page_path,
            "type":                page_type,
            "entity":              entity,
            "all_drugs_mentioned": all_drugs,
        })

    if pages_for_index:
        update_index_py(pages_for_index)

    return updated_paths, failed_paths


def compile_document(
    file_path: Path,
    context: dict,
    extraction_caches: dict | None = None,
    template_caches: dict | None = None,
    company_buffer: dict | None = None,
    trial_buffer: dict | None = None,
    drug_buffer: dict | None = None,
    indication_buffer: dict | None = None,
) -> list[str]:
    """3-step compiler chain: extract → validate → write.

    When batch buffers are provided, entity pages are accumulated across files
    and written once at end-of-batch via flush_buffered_pages(). Only event
    pages (unique per slug) are written per-file.

    Returns list of wiki page paths written in this call (event pages only in batch mode).
    """
    from agents.state import update_index_py

    extraction_caches = extraction_caches or {}
    template_caches   = template_caches   or {}

    doc_type  = context["doc_type"]
    file_name = file_path.name

    logger.info(
        f"COMPILE | START | {file_name} | doc_type={doc_type} | "
        f"company={context.get('company')} | drug={context.get('drug')}"
    )

    # pre-process document
    raw_content = preprocess_document(file_path, doc_type)

    # ctgov pre-filter — skip irrelevant trials before any LLM call
    if doc_type == "ctgov":
        try:
            _parsed_ctgov = json.loads(raw_content)
        except (json.JSONDecodeError, AttributeError):
            _parsed_ctgov = {}
        _passes, _skip_reason = _ctgov_passes_filter(_parsed_ctgov)
        if not _passes:
            logger.info(f"COMPILE | PRE-FILTER SKIP | {file_name} — {_skip_reason}")
            return []

    # build system prompt once — used for Step 3 (and Step 1 for non-ctgov types)
    system_prompt = build_system_prompt()

    # ── STEP 1: Extract ───────────────────────────────────────────────────────
    if doc_type == "ctgov":
        extracted = extract_ctgov_python(_parsed_ctgov)
        logger.info(f"STEP1 | PYTHON | {file_name} — LLM call skipped")
        pubmed_results = _parsed_ctgov.get("pubmed_results")
        if pubmed_results:
            findings = _extract_clinical_findings(pubmed_results)
            if findings:
                # subject_drug is set here, deterministically, from the ctgov
                # trial's own resolved drug list — not asked of the LLM, since
                # this path already knows unambiguously which trial/drug this
                # finding is about (unlike a multi-drug press release).
                tracked_drugs = [d["name"] for d in extracted.get("all_drugs_mentioned", []) if d.get("is_tracked")]
                findings["subject_drug"] = tracked_drugs[0] if tracked_drugs else None
                extracted["clinical_findings"] = findings
                logger.info(f"STEP1 | CLINICAL_FINDINGS | {file_name} — extracted from pubmed_results")
    else:
        cache_name = extraction_caches.get(doc_type)
        try:
            extracted = compile_document_step1(
                file_path, context, raw_content, doc_type, system_prompt, cache_name,
            )
        except ValueError as e:
            logger.error(f"STEP1 | FAILED | {file_name}: {e}")
            raise

    # pubmed files never trigger event pages
    if doc_type == "pubmed":
        extracted["requires_new_event_page"] = False
        extracted["suggested_event_slug"]    = None
        extracted["event_type"]              = None
        extracted["event_summary"]           = None
        # pmid is read in Python, not asked of the LLM — it's already in raw_content
        # (preprocess_pubmed includes it) and must survive into the signal verbatim.
        try:
            extracted["pmid"] = json.loads(raw_content).get("pmid")
        except (json.JSONDecodeError, AttributeError):
            extracted["pmid"] = None

    # ctgov events only for terminal status trials
    if doc_type == "ctgov":
        _TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "WITHDRAWN"}
        try:
            _ctgov_status = json.loads(raw_content).get("overall_status", "")
        except (json.JSONDecodeError, AttributeError):
            _ctgov_status = ""
        if _ctgov_status.upper() not in _TERMINAL_STATUSES:
            extracted["requires_new_event_page"] = False
            extracted["suggested_event_slug"]    = None
            extracted["event_type"]              = None
            extracted["event_summary"]           = None
            logger.info(
                f"STEP2 | SKIP EVENT | {file_name} — overall_status={_ctgov_status!r} is not terminal"
            )

    # ── STEP 2: Collect pages ─────────────────────────────────────────────────
    pages_to_update = []
    found_entities = collect_entity_pages(
        extracted, pages_to_update, context, file_path,
        company_buffer, drug_buffer, indication_buffer,
    )
    found_trials = collect_trial_and_event_pages(
        extracted, pages_to_update, file_path, context, trial_buffer,
    )

    if not pages_to_update:
        if found_entities or found_trials:
            logger.info(f"COMPILE | DONE | {file_name} — all signals buffered, no per-file writes")
        else:
            logger.info(f"COMPILE | DONE | {file_name} — no tracked entities, nothing written")
        return []

    # ── STEP 3: Write event pages (only per-file pages remain) ───────────────
    updated_paths = write_wiki_pages(
        pages_to_update, extracted, file_path, doc_type, system_prompt, template_caches,
    )

    if updated_paths:
        update_index_py(pages_to_update)

    logger.info(
        f"COMPILE | DONE | {file_name} → "
        f"{len(updated_paths)} pages written: {updated_paths}"
    )
    return updated_paths