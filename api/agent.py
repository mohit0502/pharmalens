"""
api/agent.py
PharmaLens Q&A agent — Gemini 2.5 Flash + wiki/stock tools.

Runs an agentic tool loop (max MAX_TOOL_CALLS iterations) and yields
server-sent event dicts that the FastAPI route streams to the client:

  {"type": "tool_call",   "name": str, "input": dict}
  {"type": "tool_result", "name": str, "content": str}   # first 300 chars
  {"type": "text",        "content": str}
  {"type": "done",        "full_text": str}
"""

import asyncio
import json
from typing import AsyncGenerator

from dotenv import load_dotenv
from google import genai
from google.genai import types

from . import news
from .tools import (
    read_wiki_page, list_wiki_pages, get_stock_price, get_stock_history,
    search_wiki, search_company_wiki, parse_company_trials, resolve_ticker,
)

load_dotenv()

client = genai.Client(http_options=types.HttpOptions(timeout=60_000))
FLASH_MODEL = "gemini-2.5-flash"
MAX_TOOL_CALLS = 10

SYSTEM_PROMPT = """You are PharmaLens, a pharmaceutical intelligence assistant backed by a structured wiki.

Wiki layout:
- drugs/<drug>.md               — per-drug pages (mechanism, trials, sentiment)
- companies/<company>.md        — company pipeline, earnings intelligence, recent events
- indications/<slug>/_index.md  — therapeutic area overview (drugs, companies, trials, events)
- trials/<company>.md           — clinical trial roster per company
- events/<slug>.md              — individual corporate events (earnings, approvals, filings)

Instructions:
1. Always look up relevant wiki pages before answering. Start with read_wiki_page for known paths.
2. When you don't know the exact path (e.g. looking for an earnings event, NCT number, or specific drug mention), use search_wiki first — it returns matching file paths and snippets so you can then read_wiki_page the right file.
   - If the term is a drug/product name that isn't in the wiki's drug pages (read_wiki_page on drugs/<name>.md returns "Page not found") AND you know which company it belongs to (from context, or because search_wiki's snippets mention the company), use search_company_wiki(company_slug, query) instead of repeating search_wiki — it scans only that company's own page, trials, drugs, and events, so it stays fast and still surfaces real coverage (e.g. earnings calls, event log entries) even for a drug too new or niche to have its own tracked page.
3. For any question about a company's clinical trials (which trials, which indications, dates, status, results), use get_company_trials instead of read_wiki_page on trials/<company>.md — that file can hold 50+ trials and gets cut off well before recent ones, while get_company_trials returns the full structured list pre-sorted newest-first.
4. Quote specific numbers, dates, and drug names from the wiki — do not hallucinate.
5. All stock/news tools take a company_slug (e.g. 'eli-lilly', 'novo-nordisk') — you never need to know or ask for a ticker symbol.
6. For "why did X happen" or "why did the stock move" questions, or anything else time-sensitive, gather both data and recent news before answering (e.g. get_stock_history for the price trajectory plus get_company_news for the catalysts behind it) — static wiki pages describe the pipeline and fundamentals, not what happened this week.
7. Be concise. One short paragraph per topic; bullet points for lists.
"""

# ── Gemini tool declarations ──────────────────────────────────────────────────

TOOL_DECLARATIONS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="read_wiki_page",
            description=(
                "Read a single wiki page by its path relative to the wiki directory. "
                "Example paths: 'indications/glp1-obesity/_index.md', "
                "'companies/novo-nordisk.md', 'drugs/semaglutide.md'"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "page_path": types.Schema(
                        type=types.Type.STRING,
                        description="Path relative to wiki/, e.g. 'drugs/tirzepatide.md'",
                    )
                },
                required=["page_path"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_wiki_pages",
            description=(
                "List all .md pages under a wiki sub-directory. "
                "Useful to discover what pages exist before reading them. "
                "Example prefixes: 'companies', 'drugs', 'indications', 'events', 'trials'"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "prefix": types.Schema(
                        type=types.Type.STRING,
                        description="Sub-directory prefix, e.g. 'companies'. Leave empty to list all.",
                    )
                },
                required=[],
            ),
        ),
        types.FunctionDeclaration(
            name="get_stock_price",
            description=(
                "Get a company's current stock price, change, and % change vs. the "
                "previous close — a single live snapshot, not a trend. Use "
                "get_stock_history instead if the question is about movement over "
                "time (e.g. 'how has it moved this month')."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "company_slug": types.Schema(
                        type=types.Type.STRING,
                        description="Company slug, e.g. 'eli-lilly', 'novo-nordisk'",
                    )
                },
                required=["company_slug"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_stock_history",
            description=(
                "Get a company's price trajectory over a period — start/end price, "
                "% change, period high/low, plus the underlying candles. Use this "
                "for any 'how/why has the stock moved' question; pair it with "
                "get_company_news to explain *why* it moved, since this tool only "
                "has the price data, not the catalysts behind it."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "company_slug": types.Schema(
                        type=types.Type.STRING,
                        description="Company slug, e.g. 'eli-lilly', 'novo-nordisk'",
                    ),
                    "period": types.Schema(
                        type=types.Type.STRING,
                        description="One of '1d', '5d', '1mo', '1y'. Defaults to '1mo'.",
                    ),
                },
                required=["company_slug"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_company_news",
            description=(
                "Get recent news for a company from the last 7 days, combining BioSpace "
                "and Yahoo Finance — deals, trial readouts, FDA actions, earnings reactions, "
                "analyst notes. Use this for anything time-sensitive: 'what's happening with "
                "X', 'why did the stock move', 'any recent news'. The static wiki pages "
                "describe pipeline/fundamentals and won't have this week's events."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "company_slug": types.Schema(
                        type=types.Type.STRING,
                        description="Company slug, e.g. 'eli-lilly', 'novo-nordisk'",
                    )
                },
                required=["company_slug"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_company_trials",
            description=(
                "Get the full structured list of clinical trials for a company — trial ID, "
                "title, phase, status, completion date, indications, drugs, and results if "
                "published. Sorted newest completion date first. Use this for any question "
                "about which trials a company is running, in what indications, or trial dates "
                "— it is far more reliable than reading trials/<company>.md directly, which "
                "gets truncated before recent trials for companies with many trials."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "company_slug": types.Schema(
                        type=types.Type.STRING,
                        description="Company slug, e.g. 'roche', 'eli-lilly', 'novo-nordisk'",
                    )
                },
                required=["company_slug"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_wiki",
            description=(
                "Full-text search across all wiki pages. Returns up to 20 matches "
                "with the file path and a short snippet around the matching line. "
                "Use this when you need to find a specific event, NCT number, drug name, "
                "or any term without knowing the exact page path. "
                "Optionally scope the search with a prefix (e.g. 'events', 'companies')."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Search term or phrase, e.g. 'eli-lilly earnings 2026-04-30'",
                    ),
                    "prefix": types.Schema(
                        type=types.Type.STRING,
                        description="Optional sub-directory to search within, e.g. 'events'. Leave empty to search all.",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_company_wiki",
            description=(
                "Full-text search scoped to one company's own pages — its company page, "
                "trial roster, drug pages, and event log — instead of the entire wiki. "
                "Use this for a drug/term that isn't found via read_wiki_page (no dedicated "
                "drug page exists, e.g. a niche or newly-launched product) when you already "
                "know which company it belongs to. Much faster than search_wiki for sparse "
                "terms since it's bounded to a handful of files, and still finds real coverage "
                "in that company's earnings commentary or event history."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "company_slug": types.Schema(
                        type=types.Type.STRING,
                        description="Company slug, e.g. 'vertex', 'eli-lilly'",
                    ),
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Search term, e.g. 'casgevy'",
                    ),
                },
                required=["company_slug", "query"],
            ),
        ),
    ]
)

# ── tool dispatch ─────────────────────────────────────────────────────────────

def _dispatch(name: str, args: dict) -> str:
    if name == "read_wiki_page":
        return read_wiki_page(args.get("page_path", ""))
    if name == "list_wiki_pages":
        result = list_wiki_pages(args.get("prefix", ""))
        return json.dumps(result)
    if name == "get_stock_price":
        slug = args.get("company_slug", "")
        ticker = resolve_ticker(slug)
        if not ticker:
            return json.dumps({"error": f"Unknown company slug: {slug}"})
        return json.dumps(get_stock_price(ticker))
    if name == "get_stock_history":
        result = get_stock_history(args.get("company_slug", ""), args.get("period", "1mo"))
        return json.dumps(result)
    if name == "get_company_news":
        result = news.get_company_news(args.get("company_slug", ""))
        return json.dumps(result)
    if name == "search_wiki":
        result = search_wiki(args.get("query", ""), args.get("prefix", ""))
        return json.dumps(result)
    if name == "search_company_wiki":
        result = search_company_wiki(args.get("company_slug", ""), args.get("query", ""))
        return json.dumps(result)
    if name == "get_company_trials":
        trials = parse_company_trials(args.get("company_slug", ""))
        # Drop internal-only fields the LLM doesn't need; keep the rest compact.
        trimmed = [
            {k: v for k, v in t.items() if k not in ("phase_display", "is_active")}
            for t in trials
        ]
        return json.dumps(trimmed)
    return f"Unknown tool: {name}"


# ── system-prompt context cache ────────────────────────────────────────────────
# Same client.caches.create() pattern as agents/orchestrator.py:create_cache —
# one cache shared by every request on this process, not per-user-session
# (Gemini caching is content-scoped, not session-scoped). Saves resending the
# ~600-token system prompt on every turn of every conversation. Built lazily
# on first use rather than at import time so a transient Gemini error on
# module load doesn't take down the whole API process.
_SYSTEM_CACHE_TTL = "3600s"
_system_cache_name: str | None = None
_system_cache_checked = False


def _get_system_cache_name() -> str | None:
    """Returns the cache name to pass as `cached_content`, or None to fall back
    to passing system_instruction directly. None is also the permanent outcome
    if creation fails once — e.g. the prompt is below Gemini's minimum token
    count for caching — so we don't retry a doomed call on every request."""
    global _system_cache_name, _system_cache_checked
    if _system_cache_checked:
        return _system_cache_name
    _system_cache_checked = True
    try:
        cache = client.caches.create(
            model=FLASH_MODEL,
            config=types.CreateCachedContentConfig(
                system_instruction=SYSTEM_PROMPT,
                ttl=_SYSTEM_CACHE_TTL,
            ),
        )
        _system_cache_name = cache.name
    except Exception:
        _system_cache_name = None
    return _system_cache_name


# ── agent loop ────────────────────────────────────────────────────────────────

async def run_agent(
    question: str,
    indication: str | None = None,
    company: str | None = None,
    article: str | None = None,
    history: list[dict] | None = None,
) -> AsyncGenerator[dict, None]:
    """Async generator that runs the agentic loop and yields SSE event dicts.

    `history` is prior {question, answer} turns from the same browser-side
    chat (see api/main.py:AskRequest) — replayed into `contents` so follow-up
    questions have the earlier exchange as context. The backend itself holds
    no session state between HTTP requests."""

    # Build user message with optional context hint
    context_lines = []
    if article:
        # Article text is static and already fetched for display by the time the
        # user asks a question — inject it directly instead of giving the model a
        # tool to fetch it, since there's nothing dynamic to decide here.
        try:
            art = news.get_article(article)
            context_lines.append(
                f"The user is reading this article:\nTitle: {art['title']}\n\n{art['body_text'][:6000]}"
            )
        except Exception:
            pass
    if indication:
        # Deterministically inject the indication hub page instead of relying
        # on the model to choose to call read_wiki_page for it — it's a known,
        # fixed path given this context, so there's no decision to make.
        # (Skipping read_wiki_page here was the AbbVie-earnings failure mode:
        # the model went straight to search_wiki with a verbose query instead
        # of just reading the page it already knew the path to.)
        page = read_wiki_page(f"indications/{indication}/_index.md")
        if not page.startswith("Page not found"):
            context_lines.append(
                f"User is currently viewing indication '{indication}'. "
                f"Here is its wiki page (indications/{indication}/_index.md) — "
                f"already fetched, no need to read_wiki_page it again unless "
                f"you need a different page:\n{page[:10000]}"
            )
        else:
            context_lines.append(f"User is currently viewing indication: {indication}")
    if company:
        page = read_wiki_page(f"companies/{company}.md")
        if not page.startswith("Page not found"):
            context_lines.append(
                f"User is currently viewing company '{company}'. "
                f"Here is its wiki page (companies/{company}.md) — already "
                f"fetched, no need to read_wiki_page it again unless you need "
                f"a different page (e.g. drugs/<drug>.md, trials/{company}.md, "
                f"or get_company_trials/get_company_news for more detail):\n{page[:10000]}"
            )
        else:
            context_lines.append(f"User is currently viewing company: {company}")

    user_text = ("\n".join(context_lines) + "\n\n" if context_lines else "") + question

    # Replay prior turns first so the model has the earlier exchange as
    # context — tool calls from previous turns aren't replayed, just the
    # final question/answer text, which is all the frontend retains anyway.
    contents: list[types.Content] = []
    for turn in (history or []):
        prior_q, prior_a = turn.get("question", ""), turn.get("answer", "")
        if not prior_q or not prior_a:
            continue
        contents.append(types.Content(role="user", parts=[types.Part(text=prior_q)]))
        contents.append(types.Content(role="model", parts=[types.Part(text=prior_a)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    cache_name = _get_system_cache_name()
    config_kwargs = dict(
        tools=[TOOL_DECLARATIONS],
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_budget=2048),
    )
    if cache_name:
        config_kwargs["cached_content"] = cache_name
    else:
        config_kwargs["system_instruction"] = SYSTEM_PROMPT

    full_text_parts: list[str] = []

    try:
        for _ in range(MAX_TOOL_CALLS):
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=FLASH_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                ),
                timeout=60,
            )

            candidate = response.candidates[0]
            function_calls: list = []
            text_parts: list[str] = []

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    function_calls.append(part.function_call)
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            if text_parts:
                text = "".join(text_parts)
                full_text_parts.append(text)
                yield {"type": "text", "content": text}

            if not function_calls:
                break

            contents.append(candidate.content)

            tool_result_parts: list[types.Part] = []
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}

                yield {"type": "tool_call", "name": fn_name, "input": fn_args}

                try:
                    # _dispatch does blocking I/O (GCS, yfinance, requests) — run it
                    # off the event loop so a slow tool call doesn't freeze every
                    # other concurrent request on this worker, and cap it so a stuck
                    # call fails the request instead of hanging the SSE stream forever.
                    raw = await asyncio.wait_for(
                        asyncio.to_thread(_dispatch, fn_name, fn_args), timeout=45
                    )
                except asyncio.TimeoutError:
                    raw = f"Error: {fn_name} timed out after 45s"
                except Exception as exc:
                    raw = f"Error: {exc}"

                # 8000 chars was silently dropping recent trials from large company
                # trial rosters (e.g. roche.md at 46K chars, 34 trials — only the
                # first 3, oldest, survived). Gemini 2.5 Flash has a 1M token context,
                # so this can afford to be generous; get_company_trials sidesteps the
                # problem structurally for trial data specifically.
                if len(raw) > 30000:
                    raw = raw[:30000] + "\n...[truncated]"

                yield {"type": "tool_result", "name": fn_name, "content": raw[:300]}

                tool_result_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn_name,
                            response={"output": raw},
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=tool_result_parts))

    except asyncio.TimeoutError:
        yield {"type": "text", "content": "\n\n_Request timed out after 60 seconds. Try a more specific question._"}

    yield {"type": "done", "full_text": "".join(full_text_parts)}
