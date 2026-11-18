"""Shared state, helpers and components for the Research Studio pages.

The central object is a research *case* (stored in st.session_state.dossiers): a thread of
questions and answers plus everything needed to audit it: the plan versions that drove it,
the evidence the agent retrieved, every claim with the passage that supports it and a
verdict, the user's review decisions, and a run trace. Chat is just how you work on a case.

A session keeps one stable id (its thread_id) for its whole life. Follow-ups never start a new
session: they're answered from the session's structured evidence by default, and only search
when the user picks "Research further". Each new answer is a new message and bumps the version.
"""

import io
import os
import re
import threading
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

import persistence
from langchain_core.tools import Tool
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from agent import (
    MODEL_NAME,
    PROMPT_VERSION,
    Answer,
    Route,
    RunContext,
    Source,
    TokenCounter,
    answer_direct,
    answer_from_session,
    check_claims,
    route_request,
    typed_request,
    build_agent_executor,
    get_llm,
    make_plan,
    plan_to_instructions,
    research_further_instructions,
    run_research,
)
from checks import (
    LOOKUP_TOOLS,
    SOURCE_TOOLS,
    apply_quote_checks,
    domain,
    evidence_text as _evidence_text,
    finalize as _finalize,
    needs_review,
    quality_checks,
    source_key as _key,
    step_name,
)
from evidence import PACK_ITEMS, build_pack, compress_step, keywords
from tools import make_documents_tool, make_search_tool, make_wiki_tool, parse_domains

RESEARCH_PAGE = "app_pages/research.py"
APPROVALS_PAGE = "app_pages/approvals.py"

MODES = ["Review plan", "Auto-research"]
MODE_CAPTIONS = [
    "Approve the plan before research starts",
    "Plan and research without stopping",
]

FURTHER_MAX_LOOKUPS = 2  # search budget for "Research further"

# Follow-ups default to the current research; searching is always the user's choice.
FOLLOWUP_MODES = ["Use current research", "Research further", "New research"]
FOLLOWUP_HELP = ("Use current research: answer from what this session already found (no searching). "
                 f"Research further: up to {FURTHER_MAX_LOOKUPS} new searches in this session. "
                 "New research: start a separate research case.")

TYPE_LABELS = {"code": "Code", "explanation": "Explanation", "research": "Research",
               "comparison": "Comparison", "fact_check": "Fact check", "clarification": "Clarification"}
TYPE_COLORS = {"code": "violet", "explanation": "gray", "research": "blue", "comparison": "blue",
               "fact_check": "orange"}
CONTEXT_LABELS = {"answer_from_session": "from current research", "research_further": "needs a few new facts",
                  "new_research": "new topic"}
VERDICT_LABELS = {"supported": ("Supported", "green"), "partially_supported": ("Partially supported", "orange"),
                  "unsupported": ("Unsupported", "red"), "uncertain": ("Uncertain", "gray")}

DEPTHS = {
    "Quick": {"steps": "1 to 2", "max_iterations": 6, "max_seconds": 60,
              "instruction": "Research depth: quick. Make 1 to 2 lookups and keep the answer short."},
    "Standard": {"steps": "2 to 4", "max_iterations": 10, "max_seconds": 150,
                 "instruction": "Research depth: standard. Make 2 to 4 lookups from different sources "
                                "and cross-check the key facts."},
    "Deep": {"steps": "4 to 6", "max_iterations": 20, "max_seconds": 300,
             "instruction": "Research depth: deep. Make 4 to 8 lookups across different sources, "
                            "cross-check every important fact, note disagreements, and use the summary "
                            "for longer context."},
}

SOURCE_OPTIONS = ["Web", "Wikipedia", "Documents"]  # sidebar source chips
STEP_VERBS = {
    "duckduckgo_search": "Searching the web",
    "wikipedia": "Reading Wikipedia",
    "search_documents": "Reading your documents",
    "request_save_file": "Preparing a file to save",
}
STATUS_COLORS = {
    # Blue is reserved for the one primary action per view; in-progress states stay neutral.
    "Needs your input": "orange",
    "Awaiting approval": "orange",
    "Drafting plan": "gray",
    "Researching": "gray",
    "Needs approval": "orange",
    "Needs review": "orange",
    "Complete": "green",
    "Failed": "red",
}

VERDICT_COLORS = {"Supported": "green", "Partially supported": "orange", "Unsupported": "red",
                  "Insufficient evidence": "gray"}

SUGGESTIONS = {
    "Solid-state batteries": "How do solid-state batteries work?",
    "Renewable-energy trends": "What are the latest trends in renewable energy?",
    "History of AI": "Give me a short history of artificial intelligence.",
    "James Webb Space Telescope": "What is the James Webb Space Telescope and what has it discovered?",
}


# ---------- State ----------

def _fresh_research_state() -> dict:
    return {"active": None, "pending_plan": None, "pending_clarify": None, "pending_decision": None}


def _restore_from_disk():
    """Once per browser session: take the session id from the URL (or make one) and reload
    any cases saved under it, so a refresh brings everything back."""
    if "sid" not in st.session_state:
        sid = st.query_params.get("sid")
        if not persistence.valid_sid(sid):
            sid = uuid.uuid4().hex
        st.session_state.sid = sid
        saved = persistence.load(sid)
        if saved:
            for k, v in saved.items():
                st.session_state[k] = v
    if st.query_params.get("sid") != st.session_state.sid:  # page switches drop query params
        st.query_params["sid"] = st.session_state.sid


def save_to_disk():
    """Called at the end of every run. Failures never break the app; the case just isn't saved."""
    if "sid" not in st.session_state:
        return
    try:
        persistence.save(st.session_state.sid, {k: st.session_state[k] for k in persistence.PERSISTED_KEYS
                                                if k in st.session_state})
    except Exception as e:  # noqa: BLE001 - e.g. a read-only disk
        st.toast(f"Couldn't save this session ({type(e).__name__}). It will be lost on refresh.",
                 icon=":material/warning:")


def init_state():
    _restore_from_disk()
    defaults = {
        "dossiers": [],  # research sessions (cases)
        "runs": [],  # every model run (new research and follow-ups), for Runs & quality
        **_fresh_research_state(),
        "pending_saves": [],  # file writes awaiting approval
        "queued_prompt": None,  # prompt queued by a button
        "queued_mode": None,  # "New research" when a button must start a separate session
        # Saves live in this visitor's session only. On a shared server (e.g. Streamlit
        # Cloud) a disk folder would be visible to every visitor and wiped on restart.
        "saved_files": [],
        "docs": {},  # uploaded documents: key -> {"name", "text"}
        "usage": {"input": 0, "output": 0},  # tokens across every model call
        "followup_mode": FOLLOWUP_MODES[0],  # "Use current research": searching is always a choice
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.get("mode") not in (None, *MODES):  # "Ask first" from an older version
        st.session_state.mode = "Auto-research"
    _migrate_old_sessions()


def _as_answer(c):
    """Bring any stored answer up to the current Answer class (after an app update)."""
    if c is None or type(c) is Answer:
        return c
    if hasattr(c, "direct_answer"):  # an Answer from before a module reload
        return Answer.model_validate(c.model_dump())
    label = (getattr(c, "answer_type", "") or "").lower()  # the old research-report format
    rtype = "code" if "code" in label else "comparison" if "compar" in label else "research"
    srcs = [Source(id=str(i), url=s, title=domain(s) if s.startswith("http") else s)
            for i, s in enumerate(getattr(c, "sources", []) or [], 1)]
    text = c.answer + (f"\n\n{c.summary}" if getattr(c, "summary", "") else "")
    return Answer(response_type=rtype, title=getattr(c, "topic", ""), direct_answer=text,
                  details=[f.text for f in getattr(c, "details", [])], sources=srcs,
                  follow_ups=list(getattr(c, "next_steps", []) or []))


_OLD_ROUTES = {"New research": "new_research", "Answer from research": "answer_from_session",
               "Research further": "research_further", "Direct answer": "direct"}


def _migrate_old_sessions():
    """Convert sessions stored by earlier versions, so an open browser tab survives an app update."""
    for d in st.session_state.dossiers:
        if "turns" not in d:
            r = d.get("result")
            c = r.response if r else None
            d["turns"] = [{"question": d.get("query", ""), "route": "New research", "card": c,
                           "error": d.get("error"), "raw": r.raw_text if r else "", "time": d.get("created", ""),
                           "seconds": d.get("seconds", 0), "searches": len(d.get("evidence", [])),
                           "new_sources": 0, "checks": d.get("checks", {}), "fresh": False}]
            d.setdefault("updated", d.get("created", ""))
        d.setdefault("evidence", [])
        for t in d["turns"]:
            t["card"] = _as_answer(t["card"])
            t.setdefault("context_mode", _OLD_ROUTES.get(t.get("route"), "new_research"))
            t.setdefault("claims", [])
        if d.get("sources") and isinstance(d["sources"][0], str):
            d["sources"] = [{"id": f"S{i}", "title": domain(s) if s.startswith("http") else s,
                             "url": s if s.startswith("http") else None,
                             "domain": domain(s) if s.startswith("http") else s}
                            for i, s in enumerate(d["sources"], 1)]
        d.setdefault("sources", [])
        if "evidence_items" not in d:
            d["evidence_items"], d["retrieved"] = [], []
            for s in d["evidence"]:
                compress_step(s.tool, s.tool_input, s.output, d.get("query", ""), d["retrieved"], d["evidence_items"])
        d.setdefault("thread_id", d["id"])
        d.setdefault("version", sum(t["card"] is not None for t in d["turns"]))
        d.setdefault("search_budget", FURTHER_MAX_LOOKUPS)


def mode() -> str:
    return st.session_state.get("mode") or "Auto-research"


def depth() -> str:
    return st.session_state.get("depth") or "Standard"


def enabled_sources() -> list[str]:
    """Sources picked in the sidebar. Documents only count once something is uploaded."""
    picked = st.session_state.get("sources_pick", ["Web", "Wikipedia"]) or []
    return [x for x in SOURCE_OPTIONS if x in picked and (x != "Documents" or st.session_state.docs)]


def source_rules() -> dict:
    """Allowlist and denylist from Settings. Enforced in the search tools themselves."""
    return {"allow": parse_domains(st.session_state.get("rule_allow", "")),
            "deny": parse_domains(st.session_state.get("rule_deny", ""))}


def rules_summary(rules: dict | None = None) -> str:
    r = rules or source_rules()
    parts = []
    if r["allow"]:
        parts.append("only " + ", ".join(r["allow"]))
    if r["deny"]:
        parts.append("never " + ", ".join(r["deny"]))
    return "; ".join(parts)


def rules_instruction() -> str:
    r = source_rules()
    lines = []
    if r["allow"]:
        lines.append(f"Only use and cite sources from these domains: {', '.join(r['allow'])}.")
    if r["deny"]:
        lines.append(f"Never use or cite sources from: {', '.join(r['deny'])}.")
    return ("\n\nSource rules set by the user:\n" + "\n".join(lines)) if lines else ""


@st.cache_resource(show_spinner="Starting the research agent...")
def cached_llm():
    return get_llm()


def has_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def now() -> str:
    return datetime.now().strftime("%H:%M")


# ---------- Usage & cost ----------

def add_usage(counter: TokenCounter):
    st.session_state.usage["input"] += counter.input_tokens
    st.session_state.usage["output"] += counter.output_tokens


def estimate_cost(input_tokens: int, output_tokens: int) -> float | None:
    """USD estimate from the per-million-token prices set in Settings, or None if unset."""
    p_in = st.session_state.get("price_in") or 0.0
    p_out = st.session_state.get("price_out") or 0.0
    if not p_in and not p_out:
        return None
    return input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out


def log_run(kind: str, query: str, session_id: str | None, seconds: float, counter: TokenCounter,
            searches: int, status: str, checks: dict | None = None, error: str | None = None,
            context_chars: int = 0):
    st.session_state.runs.append({
        "time": now(), "kind": kind, "query": query, "session": session_id, "seconds": seconds,
        "tokens_in": counter.input_tokens, "tokens_out": counter.output_tokens, "searches": searches,
        "status": status, "checks": checks or {}, "error": error, "depth": depth(),
        "model": MODEL_NAME, "prompt_version": PROMPT_VERSION, "rules": rules_summary() or "none",
        "context_chars": context_chars,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


# ---------- Uploaded documents ----------

def ingest_uploads(files):
    """Parse newly uploaded files; drop ones the user removed."""
    files = files or []
    keep = {}
    for f in files:
        key = f"{f.name}:{f.size}"
        if key in st.session_state.docs:
            keep[key] = st.session_state.docs[key]
            continue
        try:
            if f.name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(f.getvalue())).pages)
            else:
                text = f.getvalue().decode("utf-8", errors="replace")
        except Exception as e:
            st.toast(f"Couldn't read {f.name}: {e}", icon=":material/error:")
            continue
        if text.strip():
            keep[key] = {"name": f.name, "text": text[:300_000]}
        else:
            st.toast(f"{f.name} has no readable text (scanned PDF?)", icon=":material/warning:")
    st.session_state.docs = keep


# ---------- Helpers ----------

def format_source(source: str) -> str:
    source = source.strip()
    if source.startswith(("http://", "https://")):
        return f"[{domain(source)}]({source})"
    return source


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w.\- ]", "_", os.path.basename(name.strip())) or "Research_output"
    return name if name.lower().endswith(".txt") else f"{name}.txt"


def card(d: dict) -> Answer | None:
    """The most recent successful answer in a session."""
    for t in reversed(d["turns"]):
        if t["card"] is not None:
            return t["card"]
    return None


def dossier_status(d: dict) -> str:
    if card(d) is None:
        return "Failed"
    if any(r.get("dossier") == d["id"] for r in st.session_state.pending_saves):
        return "Needs approval"
    if review_queue(d):
        return "Needs review"
    return "Complete"


# ---------- Claims ----------

def all_claims(d: dict) -> list[tuple[int, int, dict]]:
    """(turn index, claim index, claim) for every checked claim in a case."""
    return [(n, k, c) for n, t in enumerate(d["turns"]) for k, c in enumerate(t.get("claims") or [])]


def review_queue(d: dict) -> list[tuple[int, int, dict]]:
    return [x for x in all_claims(d) if needs_review(x[2])]


def conflicts(d: dict) -> list[tuple[int, int, dict]]:
    return [x for x in all_claims(d) if x[2]["conflict_quote"]]


def review_claim(idx: int, n: int, k: int, decision: str | None):
    st.session_state.dossiers[idx]["turns"][n]["claims"][k]["review"] = decision


def claims_summary(claims: list[dict]) -> str:
    if not claims:
        return ""
    ok = sum(c["verdict"] == "Supported" for c in claims)
    text = f"{ok}/{len(claims)} claims supported"
    flagged = sum(needs_review(c) for c in claims)
    if flagged:
        text += f" · {flagged} to review"
    return text


def verify_claims(c: Answer, steps: list, parent, counter: TokenCounter) -> list[dict]:
    """Check each claim in an answer against the retrieved text, inside `parent` (a chat bubble).

    The checker quotes its evidence; the code then confirms each quote really is in the text
    the tools returned. A quote that isn't there can't support anything, so the claim is
    downgraded rather than trusted. Code answers make no factual claims and are skipped.
    """
    if c.response_type == "code" or not c.sources:
        return []
    evidence = _evidence_text(steps)
    text = "\n\n".join([c.direct_answer] + list(c.details)
                       + [f"{r.option} — {r.aspect}: {r.value}" for r in (c.comparison or [])])
    urls = [s.url or s.title for s in c.sources]
    with parent:
        status = st.status(":shimmer[Checking each claim against the sources...]", type="compact")
        if not evidence.strip():
            status.update(label="No retrieved text to check claims against", state="complete")
            return []
        try:
            checked = check_claims(cached_llm(), text, urls, evidence, callbacks=[counter])
        except Exception as e:
            status.update(label=f"Couldn't check claims ({type(e).__name__})", state="error")
            return []
    claims = apply_quote_checks(checked, evidence, urls)
    ok = sum(x["verdict"] == "Supported" for x in claims)
    status.update(label=f"Checked {_plural(len(claims), 'claim')} · {ok} supported", state="complete")
    return claims


def source_count(d: dict) -> int:
    return len(d["sources"])


def _card_markdown(c: Answer) -> str:
    parts = [f"*{TYPE_LABELS.get(c.response_type, c.response_type)}* · {c.title}", c.direct_answer]
    if c.verdict:
        parts.append(f"**Verdict:** {VERDICT_LABELS[c.verdict][0]}")
    if c.code:
        parts.append((f"`{c.code.filename}`\n\n" if c.code.filename else "")
                     + f"```{c.code.language.lower()}\n{c.code.code}\n```")
        if c.code.run_instructions:
            parts.append("Run:\n```bash\n" + "\n".join(c.code.run_instructions) + "\n```")
        if c.code.expected_output:
            parts.append(f"Expected output:\n```text\n{c.code.expected_output}\n```")
        if c.code.explanation:
            parts.append("\n".join(f"- {x}" for x in c.code.explanation))
    if c.comparison:
        parts.append(_comparison_table(c))
    if c.details:
        parts.append("\n".join(f"- {x}" for x in c.details))
    if c.sources:
        parts.append("Sources:\n" + "\n".join(
            f"{s.id}. {s.title} — {s.url or 'uploaded document'}" + (f" — “{s.quote}”" if s.quote else "")
            for s in c.sources))
    return "\n\n".join(parts)


def report_markdown(d: dict) -> str:
    """Audit report: every answer with its claims, quoted evidence and verdicts, then the plan
    versions, the lookups made and the run trace."""
    claims, queue, confl = all_claims(d), review_queue(d), conflicts(d)
    ok = sum(c["verdict"] == "Supported" for _, _, c in claims)
    parts = [
        f"# {d['title']}",
        f"**Research case** · {dossier_status(d)} · v{sum(t['card'] is not None for t in d['turns'])} · "
        f"exported {datetime.now():%Y-%m-%d %H:%M}  \n"
        f"Model `{MODEL_NAME}` · prompt version `{PROMPT_VERSION}` · depth {d.get('depth', 'Standard')}  \n"
        f"Claims: {ok}/{len(claims)} supported · {len(queue)} awaiting review · {len(confl)} with conflicting evidence",
    ]
    for i, t in enumerate(d["turns"], 1):
        parts.append(f"## {i}. {t['question']}")
        parts.append(f"_Asked {t.get('timestamp', t['time'])} · {turn_status(t)}"
                     + (f" · source rules: {t['rules']}" if t.get("rules") else "") + "_")
        parts.append(_card_markdown(t["card"]) if t["card"] else f"_{t['error'] or 'No answer.'}_")
        if t.get("claims"):
            rows = ["| # | Claim | Verdict | Evidence (quoted) | Source | Review |", "|---|---|---|---|---|---|"]
            for k, c in enumerate(t["claims"], 1):
                verdict = c["verdict"] + (" · conflicting evidence" if c["conflict_quote"] else "")
                rows.append("| " + " | ".join(_cell(x) for x in [
                    k, c["claim"], verdict, f"“{c['quote']}”" if c["quote"] else "–",
                    c["source"] or "–", c["review"] or ("pending" if needs_review(c) else "–")]) + " |")
            parts.append("**Claims**\n\n" + "\n".join(rows))
            for c in t["claims"]:
                if c["conflict_quote"]:
                    parts.append(f"> **Conflict:** “{c['claim']}” is contradicted by: “{c['conflict_quote']}” "
                                 f"({c['conflict_source'] or 'retrieved text'})")
    parts.append("---\n\n## Appendix A. Plan versions")
    if d.get("plans"):
        for p in d["plans"]:
            steps = "\n".join(f"{i}. {s['question']} ({', '.join(s['sources'])}; {s.get('priority', 'Required')}; "
                              f"{s.get('origin', 'drafted by agent')})" + (f" Why: {s['why']}" if s.get("why") else "")
                              for i, s in enumerate(p["steps"], 1))
            parts.append(f"**v{p['version']}**, approved {p['time']} for “{p['question']}”\n\n{steps}")
    else:
        parts.append("_No plan: researched in Auto-research mode._")
    parts.append("## Appendix B. Lookups")
    parts.append("\n".join(f"- {step_name(s.tool)}: {s.tool_input}" for s in d["evidence"]) or "_None._")
    parts.append("## Appendix C. Run trace\n\n" + _trace_table(d))
    parts.append("## How to read this\n\n"
                 "- **Supported**: the quoted passage fully backs the claim.\n"
                 "- **Partially supported**: it backs part of the claim, or a weaker version.\n"
                 "- **Unsupported**: the evidence addresses the point but doesn't back the claim.\n"
                 "- **Insufficient evidence**: nothing retrieved addresses it.\n\n"
                 "Every quote was checked word for word against the text the research tools returned; "
                 "claims whose quote couldn't be found there were downgraded. This makes each material "
                 "claim traceable to inspectable evidence. It is not a guarantee that the claims are true.")
    return "\n\n".join(parts) + "\n"


def scope_estimate(steps: list[dict]) -> str:
    lookups = sum(len(s["sources"]) for s in steps)
    secs = 10 + 8 * lookups
    when = "under a minute" if secs < 60 else f"about {secs // 60}–{secs // 60 + 1} minutes"
    return f"{lookups} lookup{'s' if lookups != 1 else ''} · {when}"


def _plural(n: int, word: str) -> str:
    if n == 1:
        return f"{n} {word}"
    return f"{n} {word}{'es' if word.endswith(('ch', 'sh', 's', 'x')) else 's'}"


# ---------- Evidence pack ----------

def session_pack(d: dict, question: str) -> str:
    """Compact, structured context for a follow-up: the original question, the current answer
    (and code), and the strongest evidence items for this follow-up. Never raw pages or the
    whole chat history."""
    c = card(d)
    current = (c.direct_answer + ("\n" + "\n".join(f"- {x}" for x in c.details) if c.details else "")) if c else ""
    return build_pack(d["query"], current, c.code.code if c and c.code else None,
                      [t["question"] for t in d["turns"]], d["evidence_items"], question)


def _add_evidence(d: dict, steps: list, question: str = ""):
    """Keep raw lookups for the audit trail and quote checks, and compress each into evidence items."""
    seen = {(s.tool, s.tool_input) for s in d["evidence"]}
    for s in steps:
        if s.output.startswith("Search budget"):
            continue
        if s.tool in LOOKUP_TOOLS and (s.tool, s.tool_input) not in seen:
            d["evidence"].append(s)
            seen.add((s.tool, s.tool_input))
            compress_step(s.tool, s.tool_input, s.output, question or d["query"],
                          d.setdefault("retrieved", []), d.setdefault("evidence_items", []))


# ---------- Automatic quality checks ----------

# ---------- Callbacks ----------

def queue_prompt(prompt: str, followup: str | None = None):
    st.session_state.queued_prompt = prompt
    st.session_state.queued_mode = followup


def prefill_composer(text: str):
    st.session_state.composer = text


def pick_suggestion():
    choice = st.session_state.get("suggestion")
    if choice:
        queue_prompt(SUGGESTIONS[choice])
    st.session_state.suggestion = None


def pick_related(key: str):
    choice = st.session_state.get(key)
    if choice:
        queue_prompt(choice)
    st.session_state[key] = None


def new_research():
    for k, v in _fresh_research_state().items():
        st.session_state[k] = v


def open_report(idx: int):
    new_research()
    st.session_state.active = idx


def clear_history():
    new_research()
    st.session_state.dossiers = []
    st.session_state.runs = []
    st.session_state.pending_saves = []
    st.session_state.usage = {"input": 0, "output": 0}


def cancel_plan():
    st.session_state.pending_plan = None


def cancel_clarify():
    st.session_state.pending_clarify = None


def add_plan_step():
    plan = st.session_state.pending_plan
    plan["steps"].append({"sid": uuid.uuid4().hex[:8], "question": "", "sources": [plan["allowed"][0]],
                          "expected": "", "why": "", "origin": "you", "priority": "Required"})


def move_plan_step(sid: str, delta: int):
    steps = st.session_state.pending_plan["steps"]
    i = next(k for k, s in enumerate(steps) if s["sid"] == sid)
    j = i + delta
    if 0 <= j < len(steps):
        steps[i], steps[j] = steps[j], steps[i]


def remove_plan_step(sid: str):
    plan = st.session_state.pending_plan
    plan["steps"] = [s for s in plan["steps"] if s["sid"] != sid]


def new_research_job(query: str, idx: int | None = None, route: dict | None = None) -> dict:
    """A full-research job, following the sidebar's research mode.

    `idx` is the open thread the answer is appended to; None starts a new session.
    """
    job = {
        "Review plan": {"kind": "plan", "query": query, "context": ""},
        "Auto-research": {"kind": "research", "query": query, "steps": None, "context": ""},
    }[mode()]
    job["idx"] = idx
    job["route"] = route
    return job


def decide(choice: str):
    """Resolve an ask-before-searching message.

    "further": research further in this session (bounded search). "anyway": answer from the
    current research. "new": start a separate research case for this question.
    """
    dec = st.session_state.pending_decision
    st.session_state.pending_decision = None
    if not dec:
        return
    if choice == "further":
        st.session_state.job = {"kind": "further", "idx": dec["idx"], "question": dec["question"],
                                "route": dec.get("route")}
    elif choice == "anyway":
        st.session_state.job = {"kind": "context", "idx": dec["idx"], "question": dec["question"], "force": True,
                                "route": dec.get("route")}
    else:
        st.session_state.active = None
        st.session_state.job = {"kind": "start", "query": dec["question"]}


def _store_file(name: str, data: str) -> str:
    """Add a file to this session's library; returns the (possibly de-duplicated) name."""
    taken = {f["name"] for f in st.session_state.saved_files}
    stem, ext = os.path.splitext(name)
    n = 2
    while name in taken:
        name = f"{stem} ({n}){ext}"
        n += 1
    st.session_state.saved_files.append(
        {"name": name, "data": data, "when": datetime.now().strftime("%b %d, %H:%M")}
    )
    return name


def approve_save(req_id: str):
    req = next((r for r in st.session_state.pending_saves if r["id"] == req_id), None)
    if not req:
        return
    name = _store_file(safe_filename(st.session_state.get(f"save_name_{req_id}", req["filename"])), req["data"])
    st.session_state.pending_saves.remove(req)
    st.toast(f"Saved {name} to your library", icon=":material/check_circle:")


def skip_save(req_id: str):
    st.session_state.pending_saves = [r for r in st.session_state.pending_saves if r["id"] != req_id]
    st.toast("Skipped. Nothing was saved.")


def save_report(idx: int):
    d = st.session_state.dossiers[idx]
    name = _store_file(safe_filename(d["title"][:50].replace(" ", "_")), report_markdown(d))
    st.toast(f"Saved {name} to your library", icon=":material/check_circle:")


def save_turn(idx: int, n: int):
    d = st.session_state.dossiers[idx]
    t = d["turns"][n]
    name = _store_file(safe_filename((t["card"].title or d["title"])[:50].replace(" ", "_")),
                       f"# {t['question']}\n\n{_card_markdown(t['card'])}\n")
    st.toast(f"Saved {name} to your library", icon=":material/check_circle:")


# ---------- Live progress ----------

class LiveProgress(BaseCallbackHandler):
    """Turns tool calls into a one-line progress strip, with details on demand."""

    run_inline = True

    def __init__(self, status, finishing: str = "Drafting the answer"):
        self.status = status
        self.finishing = finishing
        self.ctx = get_script_run_ctx()
        self.done: dict[str, int] = {}
        self.current = None

    def _attach(self):
        if get_script_run_ctx() is None and self.ctx is not None:
            add_script_run_ctx(threading.current_thread(), self.ctx)

    def _label(self, doing: str) -> str:
        done = " · ".join(f"✓ {n}" + (f" ×{c}" if c > 1 else "") for n, c in self.done.items())
        return f"{done} · :shimmer[{doing}...]" if done else f":shimmer[{doing}...]"

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._attach()
        name = (serialized or {}).get("name") or kwargs.get("name", "tool")
        self.status.update(label=self._label(STEP_VERBS.get(name, f"Running {step_name(name)}")))
        with self.status:
            self.current = (name, st.status(f"{step_name(name)}: {str(input_str)[:80]}", type="step"))

    def on_tool_end(self, output, **kwargs):
        self._attach()
        if self.current is None:
            return
        name, step = self.current
        text = str(getattr(output, "content", output))
        with step:
            st.caption(text[:300] + ("…" if len(text) > 300 else ""))
        step.update(state="error" if " failed (" in text[:60] else "complete")
        self.done[step_name(name)] = self.done.get(step_name(name), 0) + 1
        self.status.update(label=self._label(self.finishing))
        self.current = None

    def on_tool_error(self, error, **kwargs):
        self._attach()
        if self.current is not None:
            self.current[1].update(state="error")
            self.current = None


def make_request_save_tool(pending: list) -> Tool:
    """Side-effect guardrail: the agent can only request a save; the user approves it."""

    def request_save(text: str) -> str:
        pending.append({"id": uuid.uuid4().hex[:8], "data": text, "filename": "Research_output.txt"})
        return "Save request sent to the user for approval. Do not call this again; continue with your answer."

    return Tool(
        name="request_save_file",
        func=request_save,
        description="Save research text to a file. Only use when the user asks to save something. "
        "The user must approve before anything is written. Input: the full text to save.",
    )


def _budgeted(tools: list, budget: int) -> list:
    """Cap total lookups across tools. Gemini can request several searches in one step,
    so limiting agent steps alone doesn't cap searches."""
    used = {"n": 0}

    def wrap(t: Tool) -> Tool:
        def run(query: str) -> str:
            if used["n"] >= budget:
                return f"Search budget of {budget} reached. Answer now using the evidence you have."
            used["n"] += 1
            return t.func(query)
        return Tool(name=t.name, func=run, description=t.description)

    return [wrap(t) for t in tools]


def _tools(pending: list, lookup_budget: int | None = None) -> list:
    sources = enabled_sources()
    rules = source_rules()
    tools = []
    if "Web" in sources:
        tools.append(make_search_tool(rules["allow"], rules["deny"]))
    if "Wikipedia" in sources and (wiki := make_wiki_tool(rules["allow"], rules["deny"])):
        tools.append(wiki)
    if "Documents" in sources:
        tools.append(make_documents_tool(list(st.session_state.docs.values())))
    if lookup_budget is not None:
        tools = _budgeted(tools, lookup_budget)
    tools.append(make_request_save_tool(pending))
    return tools


def _attach_saves(pending: list, d: dict, question: str):
    for req in pending:
        req["dossier"] = d["id"]
        req["question"] = question
    st.session_state.pending_saves.extend(pending)


# ---------- Run stages: new research ----------

def _working_header(badge: str, title: str):
    """The question as a chat message, with what's happening to it."""
    _user_message(title)
    st.badge(badge, color=STATUS_COLORS[badge])


def draft_plan(query: str, context: str, slot):
    counter = TokenCounter()
    allowed = enabled_sources()
    with slot.container():
        _working_header("Drafting plan", query)
        with st.status(":shimmer[Drafting a research plan...]", type="compact"):
            steps = make_plan(cached_llm(), query + context, allowed,
                              n_steps=DEPTHS[depth()]["steps"], callbacks=[counter])
    add_usage(counter)
    for step in steps:
        step.update(sid=uuid.uuid4().hex[:8], origin="agent", priority="Required")
    st.session_state.pending_plan = {
        "id": uuid.uuid4().hex[:8], "query": query, "context": context, "steps": steps, "allowed": allowed,
    }


def _new_turn(question: str, route: str, c: Answer | None, seconds: float, searches: int = 0,
              new_sources: int = 0, error: str | None = None, raw: str = "", checks: dict | None = None,
              claims: list | None = None, plan_version: int | None = None, context_mode: str = "new_research",
              route_info: dict | None = None, context_chars: int = 0, lookups: list | None = None,
              partial: bool = False) -> dict:
    return {"lookups": lookups or [], "partial": partial,
            "question": question, "route": route, "card": c, "error": error, "raw": raw, "time": now(),
            "timestamp": datetime.now().isoformat(timespec="seconds"), "seconds": seconds,
            "searches": searches, "new_sources": new_sources, "checks": checks or {}, "fresh": c is not None,
            "claims": claims or [], "plan_version": plan_version, "rules": rules_summary(),
            "context_mode": context_mode, "route_info": route_info, "context_chars": context_chars}


def _claim_checks(checks: dict, claims: list[dict]) -> dict:
    if claims:
        checks["claims"] = (sum(c["verdict"] == "Supported" for c in claims), len(claims))
    return checks


def _record_plan(d: dict, steps: list[dict] | None, question: str) -> int | None:
    """Keep each approved plan as a numbered version on the case. Returns its version."""
    if not steps:
        return None
    d.setdefault("plans", [])
    version = len(d["plans"]) + 1
    d["plans"].append({"version": version, "question": question, "time": now(), "steps": steps})
    return version


def _add_sources(d: dict, c: Answer) -> int:
    """Add an answer's cited sources to the session (only new ones). Returns how many were new."""
    known = {_key(s["url"] or s["title"]) for s in d["sources"]}
    added = 0
    for s in c.sources:
        key = _key(s.url or s.title)
        if key and key not in known:
            d["sources"].append({"id": f"S{len(d['sources']) + 1}", "title": s.title, "url": s.url,
                                 "domain": s.domain or s.title})
            known.add(key)
            added += 1
    return added


def _run_context(d_id: str, route: Route | None, budget: int | None) -> RunContext:
    return RunContext(thread_id=d_id, response_type=route.response_type if route else "research",
                      context_mode=route.context_mode if route else "new_research", depth=depth(),
                      sources=enabled_sources(), source_rules=rules_summary() or "none", search_budget=budget)


def _route(route) -> Route | None:
    if route is None or isinstance(route, Route):
        return route
    return Route.model_validate(route)


def execute(query: str, steps: list[dict] | None, slot, context: str = "", idx: int | None = None,
            show_question: bool = True, route=None):
    """Run full research with live progress and store the answer as a turn.

    With `idx`, the answer is appended to that session (below the earlier messages) instead of
    starting a new one. `show_question=False` when the question bubble is already on screen.
    """
    route = _route(route)
    rtype = route.response_type if route and route.response_type != "clarification" else "research"
    sources = enabled_sources()
    cfg = DEPTHS[depth()]
    pending: list = []
    counter = TokenCounter()
    result, error = None, None
    d_id = st.session_state.dossiers[idx]["id"] if idx is not None else uuid.uuid4().hex[:8]

    instructions = context + plan_to_instructions(steps) + rules_instruction()
    if cfg["instruction"]:
        instructions += f"\n\n{cfg['instruction']}"

    start = time.perf_counter()
    with slot.container():
        if show_question:
            _user_message(query)
        # When continuing a follow-up, `slot` is already that answer's assistant bubble.
        bubble = st.chat_message("assistant") if show_question else st.container()
        with bubble:
            st.caption(" · ".join(filter(None, [
                TYPE_LABELS.get(rtype), f"Following your {len(steps)}-step plan" if steps else None,
                f"{depth()} depth", ", ".join(sources), f"Rules: {rules_summary()}" if rules_summary() else None,
            ])))
            live = st.status(":shimmer[Planning searches...]", type="compact")
        scroll_to_bottom()
        executor = build_agent_executor(cached_llm(), _tools(pending), max_iterations=cfg["max_iterations"],
                                        max_execution_time=cfg["max_seconds"])
        try:
            result = run_research(executor, typed_request(query, rtype) + instructions, None,
                                  callbacks=[LiveProgress(live), counter],
                                  context=_run_context(d_id, route, None))
            if result.hit_limit:
                error = (f"Stopped at the {depth()} depth limit before finishing. "
                         "Try again with a deeper setting or a narrower question.")
            elif result.response is None:
                error = "The agent replied, but not in the expected format."
            live.update(label="Research stopped" if error else "Research complete",
                        state="error" if error else "complete")
        except Exception as e:
            error = str(e)
            live.update(label="Research failed", state="error")

    lookups = [s for s in (result.steps if result else []) if s.tool in LOOKUP_TOOLS]
    earlier = st.session_state.dossiers[idx]["evidence"] if idx is not None else []
    response, partial = None, False
    if result and result.response and not error:
        response = _finalize(result.response, None, earlier + lookups, "new_research", len(lookups))
    elif result and result.hit_limit and lookups:
        # Stopped at the depth limit, but lookups came back: answer from what was found
        # instead of throwing the work away. The turn is marked partial.
        response = _partial_answer(query, rtype, lookups, bubble, counter)
        if response:
            response = _finalize(response, None, earlier + lookups, "new_research", len(lookups))
            partial, error = True, None
    claims = verify_claims(response, earlier + lookups, bubble, counter) if response else []
    seconds = time.perf_counter() - start
    add_usage(counter)
    checks = _claim_checks(quality_checks(response, lookups, steps), claims)
    status = "Failed" if response is None else "Partial" if partial else "Complete"
    route_info = route.model_dump() if route else None
    activity = [(step_name(s.tool), s.tool_input[:80], " failed (" in s.output[:60]) for s in lookups]

    if idx is not None:  # continue the open session: a new message below the earlier ones
        d = st.session_state.dossiers[idx]
        _add_evidence(d, lookups, query)
        n_sources = _add_sources(d, response) if response else 0
        d["turns"].append(_new_turn(query, "New research", response, seconds, len(lookups), n_sources, error,
                                    result.raw_text if result else "", checks, claims,
                                    _record_plan(d, steps, query), "new_research", route_info,
                                    lookups=activity, partial=partial))
        if response:
            d["version"] += 1
        d["updated"] = now()
        _attach_saves(pending, d, query)
        log_run("New research", query, d["id"], seconds, counter, len(lookups), status, checks, error)
        return

    d = {
        "id": d_id,
        "thread_id": d_id,  # stable for the life of the session; every run is traced under it
        "query": query,  # the original question
        "title": (response.title if response else "") or query[:60],
        "created": now(),
        "updated": now(),
        "mode": mode(),
        "depth": depth(),
        "enabled_sources": sources,
        "sources": [],  # sources cited anywhere in this session (new ones are added, never repeated)
        "evidence": [],  # raw lookups: the audit trail, and what quotes are checked against
        "evidence_items": [],  # compressed, citable evidence: what follow-ups are answered from
        "retrieved": [],  # every source retrieved, cited or not
        "turns": [],  # the messages: each question with its answer
        "plans": [],  # every approved plan, as numbered versions
        "version": 1 if response else 0,
        "search_budget": FURTHER_MAX_LOOKUPS,
    }
    _add_evidence(d, lookups, query)
    n_sources = _add_sources(d, response) if response else 0
    d["turns"].append(_new_turn(query, "New research", response, seconds, len(lookups), n_sources, error,
                                result.raw_text if result else "", checks, claims,
                                _record_plan(d, steps, query), "new_research", route_info,
                                lookups=activity, partial=partial))
    st.session_state.dossiers.append(d)
    st.session_state.active = len(st.session_state.dossiers) - 1
    _attach_saves(pending, d, query)
    log_run("New research", query, d["id"], seconds, counter, len(lookups), status, checks, error)


def _partial_answer(query: str, rtype: str, lookups: list, bubble, counter: TokenCounter):
    """Write an answer from the lookups a stopped run already made. No new searches."""
    registry, items = [], []
    for s in lookups:
        compress_step(s.tool, s.tool_input, s.output, query, registry, items)
    if not items:
        return None
    pack = build_pack(query, "", None, [], items, query)
    with bubble:
        with st.status(":shimmer[Writing a partial answer from what was found...]", type="compact") as status:
            try:
                resp = answer_from_session(cached_llm(), pack, query, rtype, True, callbacks=[counter])
            except Exception:
                resp = None
            status.update(label="Partial answer written" if resp else "Couldn't write a partial answer",
                          state="complete" if resp else "error")
    return resp


def answer_directly(query: str, route, slot):
    """A first question that needs no research (code, a plain explanation): one model call, no tools."""
    route = _route(route)
    counter = TokenCounter()
    start = time.perf_counter()
    resp, error = None, None
    with slot.container():
        _user_message(query)
        with st.chat_message("assistant"):
            with st.status(f":shimmer[Writing the {TYPE_LABELS[route.response_type].lower()}...]",
                           type="compact") as status:
                try:
                    resp = answer_direct(cached_llm(), query, route.response_type, callbacks=[counter])
                    if resp is None:
                        error = "The model returned no answer."
                except Exception as e:
                    error = str(e)
                status.update(label="Done" if not error else "Couldn't answer",
                              state="error" if error else "complete")
    seconds = time.perf_counter() - start
    add_usage(counter)
    response = _finalize(resp, set(), [], "direct", 0) if resp else None
    d_id = uuid.uuid4().hex[:8]
    d = {"id": d_id, "thread_id": d_id, "query": query, "title": (response.title if response else "") or query[:60],
         "created": now(), "updated": now(), "mode": mode(), "depth": depth(), "enabled_sources": enabled_sources(),
         "sources": [], "evidence": [], "evidence_items": [], "retrieved": [], "turns": [], "plans": [],
         "version": 1 if response else 0, "search_budget": FURTHER_MAX_LOOKUPS}
    d["turns"].append(_new_turn(query, "Direct answer", response, seconds, error=error, context_mode="direct",
                                route_info=route.model_dump(), checks={"format": response is not None}))
    st.session_state.dossiers.append(d)
    st.session_state.active = len(st.session_state.dossiers) - 1
    log_run("Direct answer", query, d_id, seconds, counter, 0, "Failed" if error else "Complete",
            {"format": response is not None}, error)


# ---------- Run stages: follow-ups in the same session ----------

def _append_answer(idx: int, question: str, new_card: Answer, new_steps: list, route: str,
                   seconds: float, counter: TokenCounter, bubble, context_mode: str = "answer_from_session",
                   route_info: dict | None = None, context_chars: int = 0):
    """Check the answer's claims, then add it to the same session as a new message (version + 1)."""
    d = st.session_state.dossiers[idx]
    _add_evidence(d, new_steps, question)
    searches = len([s for s in new_steps if s.tool in LOOKUP_TOOLS and not s.output.startswith("Search budget")])
    new_sources = _add_sources(d, new_card)
    vc = TokenCounter()
    claims = verify_claims(new_card, d["evidence"], bubble, vc)
    add_usage(vc)
    counter.input_tokens += vc.input_tokens
    counter.output_tokens += vc.output_tokens
    checks = _claim_checks(quality_checks(new_card, d["evidence"], None), claims)
    d["turns"].append(_new_turn(question, route, new_card, seconds, searches, new_sources, checks=checks,
                                claims=claims, context_mode=context_mode, route_info=route_info,
                                context_chars=context_chars))
    d["version"] += 1
    d["updated"] = now()
    log_run(route, question, d["id"], seconds, counter, searches, "Complete", checks, context_chars=context_chars)


def _append_failure(idx: int, question: str, route: str, error: str, seconds: float):
    d = st.session_state.dossiers[idx]
    d["turns"].append(_new_turn(question, route, None, seconds, error=error))
    d["updated"] = now()


def _user_message(question: str):
    with st.chat_message("user"):
        st.markdown(question)


def _ask_before_searching(idx: int, question: str, kind: str, missing: str, route: Route | None = None):
    """Stop and let the user decide; the question and the choice appear as messages in the thread."""
    st.session_state.pending_decision = {"idx": idx, "question": question, "kind": kind,
                                         "missing": (missing or "").strip(),
                                         "route": route.model_dump() if route else None}


def session_summary(d: dict) -> str:
    c = card(d)
    return (f"{d['title']} (original question: {d['query']}; latest answer type: "
            f"{c.response_type if c else 'none'}; {len(d['evidence_items'])} evidence items)")


def run_followup(idx: int, question: str, fmode: str, slot):
    """Route a follow-up, then act according to the follow-up mode the user picked.

    "Use current research": answer from the session's evidence. Never searches; if new facts are
    needed, say so and offer "Research further".
    "Research further": up to FURTHER_MAX_LOOKUPS new searches in this session; a new topic is
    confirmed before starting a separate case.
    """
    d = st.session_state.dossiers[idx]
    counter = TokenCounter()
    with slot.container():
        _user_message(question)
        bubble = st.chat_message("assistant")  # one bubble for this answer, whatever it turns into
        with bubble:
            status = st.status(":shimmer[Reading the request...]", type="compact")
            route = route_request(cached_llm(), question, session_summary(d), callbacks=[counter])
            status.update(label=f"{TYPE_LABELS.get(route.response_type, 'Clarification')} · "
                                f"{CONTEXT_LABELS[route.context_mode]} · {route.reason}", state="complete")
        scroll_to_bottom()
    add_usage(counter)

    if route.response_type == "clarification":
        _ask_before_searching(idx, question, "clarify", route.reason, route)
    elif fmode == FOLLOWUP_MODES[0]:  # Use current research
        if route.needs_tools and route.context_mode != "answer_from_session":
            _ask_before_searching(idx, question, "needs_tools", route.reason, route)
        else:
            followup_from_session(idx, question, route, bubble)
    elif route.context_mode == "new_research":  # Research further, but it's a new topic
        _ask_before_searching(idx, question, "new_topic", route.reason, route)
    else:
        followup_research_further(idx, question, bubble, show_question=False, route=route)


def followup_from_session(idx: int, question: str, route, slot, force: bool = False, show_question: bool = False):
    """Answer from the session's structured evidence: one model call, no tools, 0 new searches."""
    route = _route(route) or Route(response_type="research", context_mode="answer_from_session",
                                   needs_tools=False, reason="")
    rtype = route.response_type if route.response_type != "clarification" else "explanation"
    d = st.session_state.dossiers[idx]
    counter = TokenCounter()
    start = time.perf_counter()
    pack = session_pack(d, question)
    resp, error = None, None
    with slot.container():
        if show_question:
            _user_message(question)
            bubble = st.chat_message("assistant")
        else:
            bubble = st.container()
        with bubble:
            with st.status(":shimmer[Answering from the current research...]", type="compact") as status:
                try:
                    resp = answer_from_session(cached_llm(), pack, question, rtype, force, callbacks=[counter])
                    if resp is None:
                        error = "The model returned no answer."
                except Exception as e:
                    error = str(e)
                status.update(label="Answered from the current research" if not error else "Couldn't answer",
                              state="error" if error else "complete")
        scroll_to_bottom()
    add_usage(counter)
    seconds = time.perf_counter() - start
    if error:
        _append_failure(idx, question, "Answer from research", f"This follow-up couldn't be answered: {error}", seconds)
        log_run("Answer from research", question, d["id"], seconds, counter, 0, "Failed", error=error,
                context_chars=len(pack))
        return
    if not force and not resp.evidence_sufficient:
        log_run("Answer from research", question, d["id"], seconds, counter, 0, "Needs more research",
                context_chars=len(pack))
        _ask_before_searching(idx, question, "insufficient", resp.missing, route)
        return
    allowed = {_key(i["url"] or i["title"]) for i in d["evidence_items"]} | \
              {_key(s["url"] or s["title"]) for s in d["sources"]}
    new_card = _finalize(resp, allowed, d["evidence"], "answer_from_session", 0)
    _append_answer(idx, question, new_card, [], "Answer from research", seconds, counter, bubble,
                   "answer_from_session", route.model_dump(), len(pack))


def followup_research_further(idx: int, question: str, slot, show_question: bool = True, route=None):
    """Fetch only the missing evidence (at most FURTHER_MAX_LOOKUPS searches), then answer as a
    new message in the same session."""
    route = _route(route)
    rtype = route.response_type if route and route.response_type != "clarification" else "research"
    d = st.session_state.dossiers[idx]
    counter = TokenCounter()
    pending: list = []
    start = time.perf_counter()
    result, error, new_card = None, None, None
    pack = session_pack(d, question)
    with slot.container():
        if show_question:
            _user_message(question)
        bubble = st.chat_message("assistant") if show_question else st.container()
        live = bubble.status(f":shimmer[Researching further (up to {FURTHER_MAX_LOOKUPS} searches)...]",
                             type="compact")
        scroll_to_bottom()
        executor = build_agent_executor(cached_llm(), _tools(pending, lookup_budget=FURTHER_MAX_LOOKUPS),
                                        max_iterations=FURTHER_MAX_LOOKUPS + 2, max_execution_time=90)
        try:
            result = run_research(
                executor, research_further_instructions(pack, question, rtype, FURTHER_MAX_LOOKUPS)
                + rules_instruction(),
                None, callbacks=[LiveProgress(live, "Writing the answer"), counter],
                context=_run_context(d["id"], route, FURTHER_MAX_LOOKUPS),
            )
            new_card = result.response
            if new_card is None:
                # Budget ran out before a final answer: keep what was found, answer from the evidence.
                _add_evidence(d, result.steps, question)
                live.update(label=":shimmer[Writing the answer from what was found...]")
                resp = answer_from_session(cached_llm(), session_pack(d, question), question, rtype, True,
                                           callbacks=[counter])
                new_card = resp
            if new_card is None:
                error = result.error or "No answer was produced."
            live.update(label="Research failed" if error else "Done", state="error" if error else "complete")
        except Exception as e:
            error = str(e)
            live.update(label="Research failed", state="error")
    add_usage(counter)
    seconds = time.perf_counter() - start
    steps = result.steps if result else []
    _attach_saves(pending, d, question)
    if error or new_card is None:
        _add_evidence(d, steps, question)
        _append_failure(idx, question, "Research further", f"Researching further didn't finish: {error}", seconds)
        log_run("Research further", question, d["id"], seconds, counter,
                len([s for s in steps if s.tool in LOOKUP_TOOLS]), "Failed", error=error, context_chars=len(pack))
        return
    lookups = [s for s in steps if s.tool in LOOKUP_TOOLS and not s.output.startswith("Search budget")]
    new_card = _finalize(new_card, None, d["evidence"] + lookups, "research_further", len(lookups))
    _append_answer(idx, question, new_card, steps, "Research further", seconds, counter, bubble,
                   "research_further", route.model_dump() if route else None, len(pack))


# ---------- Components ----------

COMPOSER_PLACEHOLDER = "Ask a research question..."


def composer(disabled: bool, placeholder: str = COMPOSER_PLACEHOLDER):
    return st.chat_input(placeholder, key="composer", submit_mode="disable",
                         disabled=disabled, max_chars=2000)


def render_empty_state(disabled: bool):
    """Welcome screen with the chat bar centered on the page. Returns a submitted prompt."""
    st.space(96)
    st.title("What should I research?", text_alignment="center")
    st.caption("Every claim is checked against the passage it came from, and anything unsupported is "
               "flagged for your review.", text_alignment="center")
    # Inside a container, chat_input renders inline (centered here) instead of pinned to the bottom.
    with st.container():
        typed = composer(disabled)
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.pills("Starting points", list(SUGGESTIONS), key="suggestion",
                 label_visibility="collapsed", on_change=pick_suggestion)
    return typed


def render_clarify_card():
    """Returns (query, context) once the user continues."""
    c = st.session_state.pending_clarify
    _working_header("Needs your input", c["query"])
    st.caption("A quick question or two so the research fits what you need. Leave any blank to let the agent decide.")
    answers = []
    with st.container(border=True):
        for i, q in enumerate(c["questions"]):
            answers.append(st.text_input(q, key=f"clar_{c['id']}_{i}", persist_state="session"))
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        go = st.button("Continue", type="primary", key=f"clar_go_{c['id']}")
        skip = st.button("Skip questions", type="tertiary", key=f"clar_skip_{c['id']}")
        st.space("stretch")
        st.button("Cancel", type="tertiary", on_click=cancel_clarify, key=f"clar_cancel_{c['id']}")
    if go or skip:
        pairs = [(q, a.strip()) for q, a in zip(c["questions"], answers) if a and a.strip()] if go else []
        context = ("\n\nThe user clarified:\n" + "\n".join(f"- {q} {a}" for q, a in pairs)) if pairs else ""
        return c["query"], context
    return None


def render_plan_card():
    """Returns (query, steps, context) once the user approves the plan."""
    plan = st.session_state.pending_plan
    _working_header("Awaiting approval", plan["query"])
    st.caption("Here's how I'd research this. Reorder, edit, add or remove steps, then start. "
               "The approved plan is kept with the case.")

    current = []
    last = len(plan["steps"])
    for n, s in enumerate(plan["steps"], 1):
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"**Step {n}**")
                st.space("stretch")
                st.button(":material/arrow_upward:", type="tertiary", key=f"up_{s['sid']}", help="Move up",
                          on_click=move_plan_step, args=(s["sid"], -1), disabled=n == 1)
                st.button(":material/arrow_downward:", type="tertiary", key=f"down_{s['sid']}", help="Move down",
                          on_click=move_plan_step, args=(s["sid"], 1), disabled=n == last)
                st.button(":material/close:", type="tertiary", key=f"del_{s['sid']}",
                          help=f"Remove step {n}", on_click=remove_plan_step, args=(s["sid"],))
            if s.get("why"):
                st.caption(f"Why: {s['why']}")
            q = st.text_input(f"Step {n} question", value=s["question"], key=f"q_{s['sid']}",
                              label_visibility="collapsed", placeholder="What should this step find out?",
                              persist_state="session")
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                srcs = st.segmented_control(f"Step {n} sources", plan["allowed"], selection_mode="multi",
                                            default=s["sources"], key=f"src_{s['sid']}",
                                            persist_state="session", label_visibility="collapsed")
            if not srcs:
                st.caption(":orange[Pick at least one source for this step.]")
            elif s.get("expected"):
                st.caption(f"Looking for: {s['expected']}")
            question = (q or "").strip()
            origin = s.get("origin", "agent")
            if origin == "agent" and question != s["question"]:
                origin = "edited by you"
            current.append({"question": question, "sources": list(srcs or []),
                            "why": s.get("why", ""), "expected": s.get("expected", ""),
                            "origin": {"agent": "drafted by agent", "you": "added by you"}.get(origin, origin)})

    steps = [s for s in current if s["question"]]
    missing_source = any(not s["sources"] for s in steps)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        approved = st.button("Approve and run", type="primary", disabled=not steps or missing_source,
                             key=f"approve_{plan['id']}")
        st.button("Add step", type="tertiary", icon=":material/add:", on_click=add_plan_step,
                  key=f"add_{plan['id']}")
        st.space("stretch")
        st.caption(scope_estimate(steps) if steps else "Add at least one step")
        st.button("Cancel", type="tertiary", on_click=cancel_plan, key=f"cancel_{plan['id']}")
    return (plan["query"], steps, plan.get("context", "")) if approved else None


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sup(i: int, sources: list) -> str:
    """A small superscript citation, linked to its source when it has a URL."""
    mark = str(i).translate(_SUPERSCRIPT)
    if 1 <= i <= len(sources):
        url = sources[i - 1].url if hasattr(sources[i - 1], "url") else sources[i - 1]
        if url and str(url).startswith("http"):
            return f"[{mark}]({url})"
    return mark


_THIN = " "  # narrow no-break space: ¹ ² never reads as "12" or wraps apart


def _citations(ids: list[int], sources: list[str]) -> str:
    return _THIN.join(_sup(i, sources) for i in ids if 1 <= i <= len(sources))


def _inline_citations(text: str, sources: list[str]) -> str:
    """Turn [1] / [2][3] / [2, 3] markers in prose into superscript links, leaving code blocks alone."""
    def replace(m: re.Match) -> str:
        ids = [int(x) for x in re.findall(r"\d{1,2}", m.group(0))]
        return _citations(ids, sources)

    parts = text.split("```")
    for k in range(0, len(parts), 2):  # even parts are outside code fences
        parts[k] = re.sub(r"(?:\[\d{1,2}(?:\s*,\s*\d{1,2})*\](?!\())+", replace, parts[k])
    return "```".join(parts)


def _reveal(text: str):
    """Word-by-word reveal. (st.write_stream imports pandas, which this machine blocks.)"""
    slot = st.empty()
    shown = ""
    for word in text.split(" "):
        shown += word + " "
        slot.markdown(shown + "▌")
        time.sleep(0.02)
    slot.markdown(text)


def turn_status(t: dict) -> str:
    """One line under each answer saying how it was produced."""
    mode_ = t.get("context_mode", "new_research")
    if mode_ == "direct":
        base = "Answered directly · no research needed"
    elif mode_ == "answer_from_session":
        base = "Answered from existing research · 0 new searches"
    elif mode_ == "research_further":
        base = f"Researched further · {_plural(t['searches'], 'new search')}"
        if t["new_sources"]:
            base += f" · {_plural(t['new_sources'], 'new source')}"
    else:
        base = ("Partial answer · stopped at the depth limit" if t.get("partial") else "Researched") \
            + f" · {_plural(t['searches'], 'lookup')}"
        if t.get("card") and t["card"].sources:
            base += f" · {_plural(len(t['card'].sources), 'source')}"
    if t.get("seconds") and mode_ in ("new_research", "research_further"):
        base += f" · {t['seconds']:.0f}s"
    return f"{base} · {t['time']}"


def _comparison_table(c: Answer, cite: bool = False) -> str:
    rows = c.comparison or []
    options = list(dict.fromkeys(r.option for r in rows))
    aspects = list(dict.fromkeys(r.aspect for r in rows))
    cell = {(r.aspect, r.option): r.value for r in rows}

    def fmt(v: str) -> str:
        v = _inline_citations(v, c.sources) if cite else v
        return v.replace("|", "\\|").replace("\n", " ")

    lines = ["| | " + " | ".join(fmt(o) for o in options) + " |", "|---" * (len(options) + 1) + "|"]
    for a in aspects:
        lines.append(f"| **{fmt(a)}** | " + " | ".join(fmt(cell.get((a, o), "–")) for o in options) + " |")
    return "\n".join(lines)


def scroll_to_bottom():
    """Keep the newest message in view, like a chat app. Runs in the page, after it renders."""
    st.html(
        "<script>setTimeout(() => {"
        "const m = window.parent.document.querySelector('section.stMain, [data-testid=\"stMain\"]')"
        " || document.scrollingElement;"
        "m.scrollTo({top: m.scrollHeight, behavior: 'smooth'});"
        "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});"
        "}, 150);</script>",
        unsafe_allow_javascript=True, width="content",
    )


def _render_turn(idx: int, n: int, t: dict, last: bool, locked: bool):
    """One exchange: the user's question, then the assistant's answer below it."""
    _user_message(t["question"])
    with st.chat_message("assistant"):
        _render_answer(idx, n, t, last, locked)


def _render_answer(idx: int, n: int, t: dict, last: bool, locked: bool):
    """One standard card for every answer type: header, direct answer (code, verdict or
    comparison as needed), details, evidence, status, claims, and next steps."""
    c = t["card"]
    if c is None:
        _render_failure(idx, n, t, last, locked)
        return
    if t.get("partial"):
        st.warning("Research stopped at the depth limit. This answer uses only what was found before it "
                   "stopped. Try **Deep** depth or a narrower question for a complete answer.",
                   icon=":material/hourglass_bottom:")

    # Header: [type] · language or topic          [Copy] [Save]
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(TYPE_LABELS.get(c.response_type, "Answer"), color=TYPE_COLORS.get(c.response_type, "gray"))
        if c.code:
            st.caption(c.code.language)
        elif n > 0:  # the first answer's title is already the case title above
            st.caption(c.title)
        st.space("stretch")
        with st.popover("Copy", type="tertiary", icon=":material/content_copy:", disabled=locked):
            st.code(_card_markdown(c), language="markdown", wrap_lines=True, height=220)
        st.button("Save", type="tertiary", icon=":material/bookmark_add:", key=f"save_turn_{idx}_{n}",
                  on_click=save_turn, args=(idx, n), disabled=locked)

    claims = t.get("claims") or []
    researched = bool(c.sources) and c.response_type in ("research", "comparison", "fact_check")
    if researched:
        # Analyst layout: the answer on the left, the evidence behind it on the right.
        left, right = st.columns([3, 2], gap="large")
        with left:
            _render_answer_body(idx, n, t, c, claims, last, locked)
        with right:
            _render_evidence_panel(idx, n, c, claims, locked)
    else:
        _render_answer_body(idx, n, t, c, claims, last, locked)
        if claims:
            with st.expander(f"Claims · {claims_summary(claims)}", icon=":material/fact_check:"):
                for k, cl in enumerate(claims):
                    _render_claim(idx, n, k, cl, locked, prefix="a")

    # Next steps: only under the newest answer
    dec = st.session_state.pending_decision
    if last and c.follow_ups and not (dec and dec["idx"] == idx):
        key = f"related_{idx}_{n}"
        st.pills("Next steps", c.follow_ups[:3], key=key, on_change=pick_related, args=(key,), disabled=locked)


_HL_COLORS = {"Supported": "green", "Partially supported": "orange", "Unsupported": "red",
              "Insufficient evidence": "gray"}
_CITE = re.compile(r"\s*(?:\[\d{1,2}(?:\s*,\s*\d{1,2})*\](?!\())+")


def _claim_color(cl: dict) -> str:
    if cl["review"] == "accepted":
        return "green"
    if cl["review"] == "rejected":
        return "red"
    return "red" if cl["conflict_quote"] and cl["verdict"] == "Supported" else _HL_COLORS[cl["verdict"]]


def _highlight_claims(text: str, claims: list[dict]) -> tuple[str, int]:
    """Colour each sentence of the answer by the verdict of the claim it makes.

    The checker restates claims in its own words, so sentences are matched to claims by keyword
    overlap. Citation markers are moved just after the highlight. Returns (markdown, highlights).
    """
    if not claims or "```" in text:
        return text, 0
    kws = [(keywords(cl["claim"]), cl) for cl in claims]
    used, out_lines = 0, []
    for line in text.split("\n"):
        if not line.strip() or line.lstrip().startswith(("|", "#", ">")):
            out_lines.append(line)
            continue
        prefix = re.match(r"^\s*(?:[-*]|\d+\.)\s+", line)
        lead, rest = (prefix.group(0), line[prefix.end():]) if prefix else ("", line)
        parts = []
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*\"“(])", rest):
            words = keywords(sent)
            best, score = None, 0.0
            for ck, cl in kws:
                if ck:
                    s = len(words & ck) / len(ck)
                    if s > score:
                        best, score = cl, s
            body = _CITE.sub("", sent).strip()
            if best is None or score < 0.5 or not body or "[" in body or "]" in body:
                parts.append(sent)
                continue
            markers = "".join(m.strip() for m in _CITE.findall(sent))
            parts.append(f":{_claim_color(best)}-background[{body}]" + (f" {markers}" if markers else ""))
            used += 1
        out_lines.append(lead + " ".join(parts))
    return "\n".join(out_lines), used


def _confidence_bar(claims: list[dict], height: int = 8) -> str:
    """A stacked bar of claim verdicts, coloured like the highlights."""
    palette = {"green": "#4ADE80", "orange": "#FBBF24", "red": "#F87171", "gray": "#5B636B"}
    counts: dict[str, int] = {}
    for cl in claims:
        counts[_claim_color(cl)] = counts.get(_claim_color(cl), 0) + 1
    segs = "".join(
        f'<div style="flex:{counts[k]};background:{palette[k]};" title="{counts[k]}"></div>'
        for k in ("green", "orange", "red", "gray") if counts.get(k))
    return (f'<div style="display:flex;gap:3px;height:{height}px;border-radius:{height}px;overflow:hidden;'
            f'margin:4px 0 2px 0;">{segs}</div>')


def _verdict_counts(claims: list[dict]) -> str:
    names = {"green": "supported", "orange": "partial", "red": "unsupported or conflicting", "gray": "no evidence"}
    counts: dict[str, int] = {}
    for cl in claims:
        counts[_claim_color(cl)] = counts.get(_claim_color(cl), 0) + 1
    return " · ".join(f":{k}[●] {counts[k]} {names[k]}" for k in names if counts.get(k))


def _render_answer_body(idx: int, n: int, t: dict, c: Answer, claims: list[dict], last: bool, locked: bool):
    # Direct answer, with each claim highlighted by its verdict
    highlighted, n_hl = _highlight_claims(c.direct_answer, claims)
    answer = _inline_citations(highlighted, c.sources)
    if n_hl:
        st.markdown(answer)
    elif t.pop("fresh", False) and last and not locked and "```" not in answer and "|" not in answer:
        _reveal(answer)
    else:
        st.markdown(answer)
    if c.verdict:
        label, color = VERDICT_LABELS[c.verdict]
        st.markdown(f":{color}-badge[Verdict: {label}]")
    if c.code:
        if c.code.filename:
            st.caption(c.code.filename)
        st.code(c.code.code, language=c.code.language.lower().replace("c++", "cpp").replace("c#", "csharp"))
        if c.code.run_instructions:
            st.caption("Run")
            st.code("\n".join(c.code.run_instructions), language="bash")
        if c.code.expected_output:
            st.caption("Expected output")
            st.code(c.code.expected_output, language="text")
        if c.code.explanation:
            st.markdown("\n".join(f"- {_md(x)}" for x in c.code.explanation[:4]))
    if c.comparison:
        st.markdown(_comparison_table(c, cite=True))

    # Details: only for answers that are research
    if c.details and c.response_type in ("research", "comparison", "fact_check"):
        st.caption("Details")
        st.markdown("\n".join(f"- {_inline_citations(_highlight_claims(x, claims)[0], c.sources)}"
                              for x in c.details))

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(turn_status(t))
        _render_activity(t)
    grounded = t["checks"].get("grounded")
    if last and grounded and grounded[0] < grounded[1]:
        st.caption(f":orange[{grounded[1] - grounded[0]} of {grounded[1]} sources weren't in the results "
                   "the agent read. Treat them with care.]")


def _render_evidence_panel(idx: int, n: int, c: Answer, claims: list[dict], locked: bool):
    """The evidence behind an answer: verdict bar, then each source with its quote and the
    claims it backs, then the claims themselves with review controls."""
    st.caption("Evidence")
    lines = []
    for i, s in enumerate(c.sources, 1):
        backed = [cl for cl in claims if cl["source"] == (s.url or s.title)]
        dots = "".join(f":{_claim_color(cl)}[●]" for cl in backed)
        title = f"[{_md(s.title or s.domain)}]({s.url})" if s.url else _md(s.title)
        line = f"{_sup(i, c.sources)} {title} {dots}".rstrip()
        if s.quote:
            line += f"  \n:gray[*“{_md(s.quote)}”*]"
        lines.append(line)
    st.markdown("\n\n".join(lines))
    if claims:
        with st.expander(f"Claims · {claims_summary(claims)}"):
            for k, cl in enumerate(claims):
                _render_claim(idx, n, k, cl, locked, prefix="a")


def _render_activity(t: dict):
    """Research activity, collapsed: every lookup this answer made, with its outcome."""
    lookups = t.get("lookups") or []
    if not lookups:
        return
    failed = sum(1 for *_, bad in lookups if bad)
    label = "Activity" + (f" · {failed} failed" if failed else "")
    with st.popover(label, type="tertiary", icon=":material/manage_search:"):
        st.markdown("  \n".join(f"{':red[✗]' if bad else ':green[✓]'} **{name}:** {_md(query)}"
                                for name, query, bad in lookups))


def _render_failure(idx: int, n: int, t: dict, last: bool, locked: bool):
    """A failed answer: what went wrong, what was kept, and what to do next."""
    with st.container(border=True):
        st.markdown(":red[:material/error:] **This answer didn't finish**")
        st.caption(t["error"] or "No answer was produced.")
        if t.get("searches"):
            st.caption(f"Kept: {_plural(t['searches'], 'lookup')} saved to this case's evidence, "
                       "so a retry or a follow-up can reuse them.")
        _render_activity(t)
        if t.get("raw"):
            with st.expander("Raw reply"):
                st.markdown(t["raw"])
        if last:
            with st.container(horizontal=True, gap="small"):
                st.button("Retry", icon=":material/refresh:", on_click=queue_prompt,
                          args=(t["question"],), key=f"retry_{idx}_{n}", disabled=locked)
                st.caption("Or change sources or depth in the sidebar first.")


def _md(text: str) -> str:
    """Quoted source text shown as Markdown: keep $ from turning into maths."""
    return text.replace("$", "\\$")


def _render_claim(idx: int, n: int, k: int, c: dict, locked: bool, prefix: str, review: bool = True):
    """One claim: verdict, the quoted passage behind it, any conflicting passage, and review."""
    claim = f"~~{_md(c['claim'])}~~" if c["review"] == "rejected" else _md(c["claim"])
    st.markdown(f":{VERDICT_COLORS[c['verdict']]}-badge[{c['verdict']}] {claim}")
    if c["quote"]:
        st.markdown(f"> “{_md(c['quote'])}”  \n> — {format_source(c['source']) if c['source'] else 'retrieved text'}")
    if c["conflict_quote"]:
        st.markdown(f":red-badge[Conflicting evidence] “{_md(c['conflict_quote'])}” — "
                    f"{format_source(c['conflict_source']) if c['conflict_source'] else 'retrieved text'}")
    if c["reason"]:
        st.caption(c["reason"])
    if not review or not (needs_review(c) or c["review"]):
        return
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if c["review"] is None:
            st.button("Accept", type="tertiary", icon=":material/check:", key=f"{prefix}_acc_{idx}_{n}_{k}",
                      on_click=review_claim, args=(idx, n, k, "accepted"), disabled=locked,
                      help="Keep this claim: you've checked it yourself")
            st.button("Reject", type="tertiary", icon=":material/close:", key=f"{prefix}_rej_{idx}_{n}_{k}",
                      on_click=review_claim, args=(idx, n, k, "rejected"), disabled=locked,
                      help="Mark this claim as wrong; it stays visible, struck through")
        else:
            st.caption(f"You {c['review']} this claim.")
            st.button("Undo", type="tertiary", key=f"{prefix}_undo_{idx}_{n}_{k}",
                      on_click=review_claim, args=(idx, n, k, None), disabled=locked)


def render_session(idx: int, locked: bool = False):
    """The research session: every question with its answer, newest at the bottom.

    `locked` disables every control while a follow-up is running.
    """
    d = st.session_state.dossiers[idx]
    c = card(d)
    render_case_header(idx, d, c, locked)

    # Every message, oldest first. Earlier ones are only ever re-drawn, never replaced.
    fresh = d["turns"][-1].get("fresh", False)
    for n, t in enumerate(d["turns"]):
        _render_turn(idx, n, t, last=n == len(d["turns"]) - 1, locked=locked)

    dec = st.session_state.pending_decision
    if dec and dec["idx"] == idx:
        _render_decision(dec, locked)

    if fresh or locked:  # a new message just arrived, or one is being written below
        scroll_to_bottom()


def _render_decision(dec: dict, locked: bool):
    """A follow-up that needs the user's decision, shown as the next messages in the thread."""
    _user_message(dec["question"])
    with st.chat_message("assistant"):
        kind = dec.get("kind")
        if kind == "clarify":
            st.markdown("Could you say a bit more about what you'd like? "
                        + (f"_{dec['missing']}_" if dec.get("missing") else ""))
            return
        if kind == "new_topic":
            st.markdown("This looks like a new topic, so it needs its own research rather than a few "
                        "searches here.")
            go = ("Start new research", "new", ":material/add:")
            alt = (f"Research here anyway (up to {FURTHER_MAX_LOOKUPS} searches)", "further")
        else:  # needs_tools / insufficient
            st.info("This needs information that isn't in the current research. Switch to "
                    f"**Research further** to add up to {FURTHER_MAX_LOOKUPS} sources.",
                    icon=":material/search_off:")
            go = (f"Research further (up to {FURTHER_MAX_LOOKUPS} searches)", "further", ":material/travel_explore:")
            alt = ("Answer from what we have", "anyway")
        if dec.get("missing"):
            # "insufficient" carries what the evidence lacks; the others carry the router's reason.
            reason = re.sub(r"^(Rule|Model):\s*", "", dec["missing"])
            st.caption(("Missing: " if kind == "insufficient" else "Why: ") + reason[:1].upper() + reason[1:])
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.button(go[0], icon=go[2], on_click=decide, args=(go[1],), key="dec_go", disabled=locked)
            st.button(alt[0], type="tertiary", on_click=decide, args=(alt[1],), key="dec_alt", disabled=locked)


def render_case_header(idx: int, d: dict, c, locked: bool):
    """Status line for the session, plus the case file: review queue, claims, conflicts, plan,
    evidence and run trace."""
    status = dossier_status(d)
    claims, queue, confl = all_claims(d), review_queue(d), conflicts(d)
    # Case-first header, kept quiet: title and status, one line of counts, one thin verdict bar.
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"#### {_md(d['title'])}")
        st.badge(status, color=STATUS_COLORS[status])
    if claims:
        st.html(_confidence_bar([cl for _, _, cl in claims], height=4))
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        counts = [_plural(len(d["sources"]), "source"), f"v{d.get('version', 1)}"]
        if claims:
            counts.insert(1, _verdict_counts([cl for _, _, cl in claims]))
        if queue:
            counts.insert(-1, f"**{len(queue)} to review**")
        st.caption(" · ".join(counts))
        st.space("stretch")
        if c is not None:
            with st.popover("Export", type="tertiary", icon=":material/download:", disabled=locked):
                st.caption("Audit report: every answer with its claims, quoted evidence, sources, "
                           "plan versions and run trace.")
                st.download_button("Download audit report (.md)", data=report_markdown(d), type="tertiary",
                                   file_name=f"{d['title'][:40].strip().replace(' ', '_') or 'research'}_audit.md",
                                   mime="text/markdown", key=f"dl_{idx}", on_click="ignore")
                st.button("Save to library", type="tertiary", on_click=save_report, args=(idx,), key=f"save_{idx}")

    if not (claims or d["evidence"] or d.get("plans")):
        return
    with st.expander("Case file" + (f" · {len(queue)} to review" if queue else ""), icon=":material/folder_open:"):
        tabs = st.tabs([f"Review · {len(queue)}", f"Claims · {len(claims)}", f"Conflicts · {len(confl)}",
                        "Plan", f"Evidence · {len(d.get('evidence_items', []))}", "Trace"])
        with tabs[0]:
            if not queue:
                st.caption("Nothing to review. Claims that aren't fully supported, or that conflict with "
                           "another source, wait here for you to accept or reject.")
            for n, k, cl in queue:
                st.caption(f"From: {d['turns'][n]['question']}")
                _render_claim(idx, n, k, cl, locked, prefix="q")
                st.divider()
        with tabs[1]:
            if not claims:
                st.caption("No claims checked yet.")
            for n, t in enumerate(d["turns"]):
                if t.get("claims"):
                    st.markdown(f"**{t['question']}**")
                    st.caption(claims_summary(t["claims"]))
                    for k, cl in enumerate(t["claims"]):
                        _render_claim(idx, n, k, cl, locked, prefix="l", review=False)
        with tabs[2]:
            if not confl:
                st.caption("No conflicting evidence found. When one source contradicts a claim, both "
                           "passages are shown here side by side.")
            for n, k, cl in confl:
                st.markdown(f"**{_md(cl['claim'])}**")
                left, right = st.columns(2)
                with left:
                    st.caption("Supporting")
                    st.markdown(f"“{_md(cl['quote'])}”  \n{format_source(cl['source'])}" if cl["quote"]
                                else "_No supporting passage found._")
                with right:
                    st.caption("Contradicting")
                    st.markdown(f"“{_md(cl['conflict_quote'])}”  \n{format_source(cl['conflict_source'])}")
        with tabs[3]:
            _render_plans(d)
        with tabs[4]:
            items = d.get("evidence_items", [])
            if not items:
                st.caption("No evidence gathered yet.")
            else:
                st.caption(f"{len(items)} evidence items from {_plural(len(d['evidence']), 'lookup')}. Follow-ups "
                           f"are answered from the {PACK_ITEMS} most relevant, not from raw pages.")
            for i in sorted(items, key=lambda x: x["confidence"], reverse=True):
                src = f"[{i['domain']}]({i['url']})" if i["url"] else _md(i["title"])
                st.markdown(f"**{_md(i['title'])}** · {src} · relevance {i['confidence']:.0%}  \n"
                            f"“{_md(i['quote'])}”")
            if d["evidence"]:
                with st.expander("Raw lookups (audit trail)"):
                    for s in d["evidence"]:
                        with st.expander(f"{step_name(s.tool)}: {s.tool_input[:80]}", type="step"):
                            st.text(s.output[:2000] + ("…" if len(s.output) > 2000 else ""))
        with tabs[5]:
            st.caption(f"Session (thread) id `{d.get('thread_id', d['id'])}` · every run below shares it.")
            st.markdown(_trace_table(d))


def _render_plans(d: dict):
    plans = d.get("plans") or []
    if not plans:
        st.caption("No plan for this case: it ran in Auto-research mode. Choose **Review plan** in "
                   "Settings to approve and edit a plan before research starts; it's kept here.")
        return
    for p in reversed(plans):
        used = [t["question"] for t in d["turns"] if t.get("plan_version") == p["version"]]
        st.markdown(f"**Plan v{p['version']}** · approved {p['time']}" + (f" · used for “{used[0]}”" if used else ""))
        for i, s in enumerate(p["steps"], 1):
            st.markdown(f"{i}. {s['question']}  \n:gray[{', '.join(s['sources'])} · {s.get('priority', 'Required')}"
                        f" · {s.get('origin', 'drafted by agent')}]")
            if s.get("why"):
                st.caption(f"Why: {s['why']}")


def _cell(x) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _trace_table(d: dict) -> str:
    runs = [r for r in st.session_state.runs if r.get("session") == d["id"]]
    if not runs:
        return "_No runs recorded for this case._"
    rows = ["| Time | Step | Status | Lookups | Context sent | Tokens | Seconds | Model | Prompt | Source rules |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        rows.append("| " + " | ".join(_cell(x) for x in [
            r.get("timestamp", r["time"]), r["kind"], r["status"], r["searches"],
            f"{r.get('context_chars', 0):,} chars" if r.get("context_chars") else "–",
            f"{r['tokens_in'] + r['tokens_out']:,}", f"{r['seconds']:.0f}", r.get("model", "–"),
            r.get("prompt_version", "–"), r.get("rules", "–")]) + " |")
    return "\n".join(rows)


def render_approval_notice(disabled: bool = False):
    n = len(st.session_state.pending_saves)
    if not n:
        return
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.markdown(f":orange[:material/pending_actions:] **{n} file{' is' if n == 1 else 's are'} waiting for your approval**")
        st.space("stretch")
        if st.button("Review", key="review_approvals", disabled=disabled):
            st.switch_page(APPROVALS_PAGE)


def render_saved_files():
    files = st.session_state.saved_files
    if not files:
        st.caption("Nothing saved yet. Use **Save** on a report, or approve a save request from the agent.")
        return
    st.caption("Saved files last for this session. Download anything you want to keep.")
    for i in reversed(range(len(files))):
        f = files[i]
        with st.expander(f"{f['name']} · {len(f['data'].encode()) / 1024:.1f} KB · {f['when']}"):
            st.text(f["data"][:3000] + ("…" if len(f["data"]) > 3000 else ""))
            st.download_button("Download", data=f["data"], file_name=f["name"], mime="text/plain",
                               type="tertiary", key=f"dlfile_{i}", on_click="ignore")
