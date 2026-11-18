"""Shared state, helpers and components for the Research Studio pages.

A research *session* (stored in st.session_state.dossiers) starts with one question and
keeps one answer card. Follow-ups update that card as a new version instead of starting
over. Each session keeps the tool results it has gathered as an evidence pack, so most
follow-ups are answered without searching again.
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
from langchain_core.tools import Tool
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from agent import (
    AgentResult,
    ResearchResponse,
    TokenCounter,
    answer_from_evidence,
    build_agent_executor,
    get_llm,
    make_clarifying_questions,
    make_plan,
    plan_to_instructions,
    research_further_instructions,
    run_research,
)
from tools import make_documents_tool, search_tool, wiki_tool

RESEARCH_PAGE = "app_pages/research.py"
APPROVALS_PAGE = "app_pages/approvals.py"

MODES = ["Ask first", "Review plan", "Auto-research"]
MODE_CAPTIONS = [
    "Clarify the question, then review the plan",
    "Approve the plan before research starts",
    "Plan and research without stopping",
]

FOLLOWUP_MODES = ["Answer from this research", "Research further", "New research"]
FURTHER_MAX_LOOKUPS = 2  # tool budget for a "Research further" follow-up
EVIDENCE_PACK_CHARS = 6000  # cap on stored snippets sent with each follow-up

DEPTHS = {
    "Quick": {"steps": "1 to 2", "max_iterations": 6, "max_seconds": 60,
              "instruction": "Research depth: quick. Use at most 2 lookups. Give 2 to 3 details."},
    "Standard": {"steps": "2 to 4", "max_iterations": 10, "max_seconds": 150, "instruction": ""},
    "Deep": {"steps": "4 to 6", "max_iterations": 20, "max_seconds": 300,
             "instruction": "Research depth: deep. Cross-check important facts across several lookups. "
                            "Give 4 to 6 details and use the summary for longer context."},
}
DEPTH_CAPTIONS = ["Fewer lookups, short answer", "Balanced", "More lookups, fuller answer"]

# Source label -> (sidebar checkbox key, tool name)
SOURCE_KEYS = {"Web": "src_web", "Wikipedia": "src_wiki", "Documents": "src_docs"}
SOURCE_TOOLS = {"Web": "duckduckgo_search", "Wikipedia": "wikipedia", "Documents": "search_documents"}

STEP_NAMES = {
    "duckduckgo_search": "Web search",
    "wikipedia": "Wikipedia",
    "search_documents": "Your documents",
    "request_save_file": "Save request",
}
STEP_VERBS = {
    "duckduckgo_search": "Searching the web",
    "wikipedia": "Reading Wikipedia",
    "search_documents": "Reading your documents",
    "request_save_file": "Preparing a file to save",
}
LOOKUP_TOOLS = {"duckduckgo_search", "wikipedia", "search_documents"}
STATUS_COLORS = {
    # Blue is reserved for the one primary action per view; in-progress states stay neutral.
    "Needs your input": "orange",
    "Awaiting approval": "orange",
    "Drafting plan": "gray",
    "Researching": "gray",
    "Needs approval": "orange",
    "Complete": "green",
    "Failed": "red",
}

SUGGESTIONS = {
    "Solid-state batteries": "How do solid-state batteries work?",
    "Renewable-energy trends": "What are the latest trends in renewable energy?",
    "History of AI": "Give me a short history of artificial intelligence.",
    "James Webb Space Telescope": "What is the James Webb Space Telescope and what has it discovered?",
}


# ---------- State ----------

def _fresh_research_state() -> dict:
    return {"active": None, "pending_plan": None, "pending_clarify": None, "pending_decision": None}


def init_state():
    defaults = {
        "dossiers": [],  # research sessions
        "runs": [],  # every model run (new research and follow-ups), for Runs & quality
        **_fresh_research_state(),
        "pending_saves": [],  # file writes awaiting approval
        "queued_prompt": None,  # prompt queued by a button
        "queued_mode": None,  # follow-up mode forced by that button, if any
        # Saves live in this visitor's session only. On a shared server (e.g. Streamlit
        # Cloud) a disk folder would be visible to every visitor and wiped on restart.
        "saved_files": [],
        "docs": {},  # uploaded documents: key -> {"name", "text"}
        "usage": {"input": 0, "output": 0},  # tokens across every model call
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def mode() -> str:
    return st.session_state.get("mode") or "Review plan"


def depth() -> str:
    return st.session_state.get("depth") or "Standard"


def followup_mode() -> str:
    return st.session_state.get("followup_mode") or FOLLOWUP_MODES[0]


def enabled_sources() -> list[str]:
    out = []
    if st.session_state.get("src_web", True):
        out.append("Web")
    if st.session_state.get("src_wiki", True):
        out.append("Wikipedia")
    if st.session_state.get("src_docs", False) and st.session_state.docs:
        out.append("Documents")
    return out


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
            searches: int, status: str, checks: dict | None = None, error: str | None = None):
    st.session_state.runs.append({
        "time": now(), "kind": kind, "query": query, "session": session_id, "seconds": seconds,
        "tokens_in": counter.input_tokens, "tokens_out": counter.output_tokens, "searches": searches,
        "status": status, "checks": checks or {}, "error": error, "depth": depth(),
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

def step_name(tool: str) -> str:
    return STEP_NAMES.get(tool, tool.replace("_", " ").capitalize())


def domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.") or url


def format_source(source: str) -> str:
    source = source.strip()
    if source.startswith(("http://", "https://")):
        return f"[{domain(source)}]({source})"
    return source


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w.\- ]", "_", os.path.basename(name.strip())) or "Research_output"
    return name if name.lower().endswith(".txt") else f"{name}.txt"


def card(d: dict) -> ResearchResponse | None:
    r = d.get("result")
    return r.response if r else None


def dossier_status(d: dict) -> str:
    if d.get("error") and card(d) is None:
        return "Failed"
    if any(r.get("dossier") == d["id"] for r in st.session_state.pending_saves):
        return "Needs approval"
    return "Complete"


def source_count(d: dict) -> int:
    c = card(d)
    return len(c.sources) if c else 0


def report_markdown(result: AgentResult) -> str:
    r = result.response
    if r is None:
        return result.raw_text
    parts = [f"# {r.topic}", f"*{r.answer_type}*", r.answer]
    if r.details or r.summary:
        details = "\n".join(f"- {f.text}" + (f" [{', '.join(map(str, f.source_ids))}]" if f.source_ids else "")
                            for f in r.details)
        parts.append("## Details\n" + "\n\n".join(p for p in (details, r.summary) if p))
    if r.sources:
        parts.append("## Evidence\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(r.sources, 1)))
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

def evidence_pack(d: dict) -> str:
    """Compact context for follow-ups: the current card plus trimmed tool snippets.

    Sends the original question, the current answer, its details and numbered sources,
    and the newest tool results up to EVIDENCE_PACK_CHARS. Never the whole chat or raw pages.
    """
    r = card(d)
    parts = [f"ORIGINAL QUESTION: {d['query']}"]
    followups = [v["question"] for v in d["versions"][1:]]
    if followups:
        parts.append("EARLIER FOLLOW-UPS:\n" + "\n".join(f"- {q}" for q in followups[-5:]))
    if r:
        parts.append(f"CURRENT ANSWER ({r.answer_type}):\n{r.answer}")
        if r.details:
            parts.append("CURRENT DETAILS:\n" + "\n".join(f"- {f.text} (sources {f.source_ids})" for f in r.details))
        if r.summary:
            parts.append(f"CURRENT CONTEXT:\n{r.summary[:1500]}")
        if r.sources:
            parts.append("SOURCES:\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(r.sources, 1)))
    snippets, budget = [], EVIDENCE_PACK_CHARS
    for step in reversed(d["evidence"]):  # newest first
        snip = f"({step_name(step.tool)}: {step.tool_input[:80]})\n{step.output[:700]}"
        if len(snip) > budget:
            break
        snippets.append(snip)
        budget -= len(snip)
    if snippets:
        parts.append("EVIDENCE SNIPPETS:\n" + "\n\n".join(snippets))
    return "\n\n".join(parts)


def _add_evidence(d: dict, steps: list):
    seen = {(s.tool, s.tool_input) for s in d["evidence"]}
    for s in steps:
        if s.output.startswith("Search budget"):
            continue
        if s.tool in LOOKUP_TOOLS and (s.tool, s.tool_input) not in seen:
            d["evidence"].append(s)
            seen.add((s.tool, s.tool_input))


# ---------- Automatic quality checks ----------

def _is_grounded(source: str, outputs: str, used_tools: set[str]) -> bool:
    src = source.strip().lower().rstrip("/")
    if not src:
        return False
    if src.startswith("http"):
        dom = domain(src)
        if src in outputs or (dom and dom in outputs):
            return True
        return "wikipedia.org" in dom and "wikipedia" in used_tools  # wiki tool output has no URLs
    if f"[document: {src}]" in outputs or src in outputs:
        return True
    return src.startswith("wikipedia") and "wikipedia" in used_tools


def quality_checks(response: ResearchResponse | None, steps: list, plan_steps: list[dict] | None) -> dict:
    used = {s.tool for s in steps}
    outputs = "\n".join(s.output for s in steps).lower()
    grounded = None
    if response and response.sources:
        hits = sum(_is_grounded(s, outputs, used) for s in response.sources)
        grounded = (hits, len(response.sources))
    plan = None
    if plan_steps:
        needed = {SOURCE_TOOLS[x] for s in plan_steps for x in s["sources"]}
        plan = needed <= used
    return {"format": response is not None, "grounded": grounded, "plan": plan}


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
    plan["steps"].append({"sid": uuid.uuid4().hex[:8], "question": "", "sources": [plan["allowed"][0]], "expected": ""})


def remove_plan_step(sid: str):
    plan = st.session_state.pending_plan
    plan["steps"] = [s for s in plan["steps"] if s["sid"] != sid]


def new_research_job(query: str) -> dict:
    """A job that starts a separate research session, following the sidebar's research mode."""
    return {
        "Ask first": {"kind": "clarify", "query": query},
        "Review plan": {"kind": "plan", "query": query, "context": ""},
        "Auto-research": {"kind": "research", "query": query, "steps": None, "context": ""},
    }[mode()]


def decide(choice: str):
    """Resolve the 'evidence isn't enough' card."""
    dec = st.session_state.pending_decision
    st.session_state.pending_decision = None
    if not dec:
        return
    if choice == "further":
        st.session_state.job = {"kind": "further", "idx": dec["idx"], "question": dec["question"]}
    elif choice == "anyway":
        st.session_state.job = {"kind": "context", "idx": dec["idx"], "question": dec["question"], "force": True}
    else:
        st.session_state.active = None
        st.session_state.job = new_research_job(dec["question"])


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
    name = _store_file(safe_filename(d["title"][:50].replace(" ", "_")), report_markdown(d["result"]))
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
    tools = []
    if "Web" in sources:
        tools.append(search_tool)
    if "Wikipedia" in sources:
        tools.append(wiki_tool)
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
    st.badge(badge, color=STATUS_COLORS[badge])
    st.title(title)


def draft_clarify(query: str, slot) -> list[str]:
    counter = TokenCounter()
    with slot.container():
        _working_header("Drafting plan", query)
        with st.status(":shimmer[Checking whether anything needs clarifying...]", type="compact"):
            questions = make_clarifying_questions(cached_llm(), query, callbacks=[counter])
    add_usage(counter)
    return questions


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
        step["sid"] = uuid.uuid4().hex[:8]
    st.session_state.pending_plan = {
        "id": uuid.uuid4().hex[:8], "query": query, "context": context, "steps": steps, "allowed": allowed,
    }


def execute(query: str, steps: list[dict] | None, slot, context: str = ""):
    """Start a new research session: run the agent with live progress, then store the card."""
    sources = enabled_sources()
    cfg = DEPTHS[depth()]
    pending: list = []
    counter = TokenCounter()
    result, error = None, None

    instructions = context + plan_to_instructions(steps)
    if cfg["instruction"]:
        instructions += f"\n\n{cfg['instruction']}"

    start = time.perf_counter()
    with slot.container():
        _working_header("Researching", query)
        st.caption(" · ".join(filter(None, [
            f"Following your {len(steps)}-step plan" if steps else None, f"{depth()} depth", ", ".join(sources),
        ])))
        live = st.status(":shimmer[Planning searches...]", type="compact")
        executor = build_agent_executor(cached_llm(), _tools(pending), max_iterations=cfg["max_iterations"],
                                        max_execution_time=cfg["max_seconds"])
        try:
            # A new session starts fresh: no earlier conversation is sent.
            result = run_research(executor, query + instructions, None, callbacks=[LiveProgress(live), counter])
            if result.hit_limit:
                error = (f"Stopped at the {depth()} depth limit before finishing. "
                         "Try again with a deeper setting or a narrower question.")
            live.update(label="Research stopped" if error else "Research complete",
                        state="error" if error else "complete")
        except Exception as e:
            error = str(e)
            live.update(label="Research failed", state="error")

    seconds = time.perf_counter() - start
    add_usage(counter)
    response = result.response if result and not error else None
    lookups = [s for s in (result.steps if result else []) if s.tool in LOOKUP_TOOLS]
    checks = quality_checks(response, lookups, steps)
    d = {
        "id": uuid.uuid4().hex[:8],
        "query": query,
        "title": response.topic if response else query[:60],
        "result": AgentResult(response, result.raw_text if result else "", lookups) if result else None,
        "error": error,
        "seconds": seconds,
        "created": now(),
        "updated": now(),
        "fresh": True,
        "mode": mode(),
        "depth": depth(),
        "sources": sources,
        "checks": checks,
        "evidence": [],
        "versions": [{"n": 1, "question": query, "route": "New research", "change": "Started new research session",
                      "searches": len(lookups), "new_sources": len(response.sources) if response else 0,
                      "time": now()}],
        "last_note": None,
    }
    _add_evidence(d, lookups)
    st.session_state.dossiers.append(d)
    st.session_state.active = len(st.session_state.dossiers) - 1
    _attach_saves(pending, d, query)
    log_run("New research", query, d["id"], seconds, counter, len(lookups),
            "Failed" if error or response is None else "Complete", checks, error)


# ---------- Run stages: follow-ups in the same session ----------

def _apply_update(idx: int, question: str, new_card: ResearchResponse, new_steps: list, route: str,
                  seconds: float, counter: TokenCounter):
    """Store an updated card as the session's next version."""
    d = st.session_state.dossiers[idx]
    old = card(d)
    _add_evidence(d, new_steps)
    searches = len([s for s in new_steps if s.tool in LOOKUP_TOOLS and not s.output.startswith("Search budget")])
    new_sources = max(0, len(new_card.sources) - (len(old.sources) if old else 0))
    d["versions"].append({
        "n": len(d["versions"]) + 1, "question": question, "route": route,
        "change": new_card.change_summary.strip() or "Answered the follow-up",
        "searches": searches, "new_sources": new_sources, "time": now(),
    })
    d["result"] = AgentResult(new_card, "", list(d["evidence"]))
    d["updated"] = now()
    d["fresh"] = True
    d["last_note"] = None
    d["checks"] = quality_checks(new_card, d["evidence"], None)
    log_run(route, question, d["id"], seconds, counter, searches, "Complete", d["checks"])


def _strip_followup_fields(resp) -> ResearchResponse:
    return ResearchResponse.model_validate(
        resp.model_dump(exclude={"evidence_sufficient", "different_topic", "missing"})
    )


def _no_new_sources(new_card: ResearchResponse, old: ResearchResponse | None) -> ResearchResponse:
    """Answering from existing research runs no tools, so it can't legitimately add sources.
    Drop any the model appended and the citations that point at them."""
    limit = len(old.sources) if old else 0
    if len(new_card.sources) <= limit:
        return new_card
    return new_card.model_copy(update={
        "sources": new_card.sources[:limit],
        "evidence": [e for e in new_card.evidence if 1 <= e.source_id <= limit],
        "details": [f.model_copy(update={"source_ids": [i for i in f.source_ids if 1 <= i <= limit]})
                    for f in new_card.details],
    })


def followup_from_evidence(idx: int, question: str, slot, force: bool = False):
    """Answer a follow-up from the session's evidence pack, with no tool calls."""
    d = st.session_state.dossiers[idx]
    counter = TokenCounter()
    start = time.perf_counter()
    resp, error = None, None
    with slot.container():
        with st.status(":shimmer[Answering from this research...]", type="compact") as status:
            try:
                resp = answer_from_evidence(cached_llm(), evidence_pack(d), question, force, callbacks=[counter])
                if resp is None:
                    error = "The model returned no answer."
            except Exception as e:
                error = str(e)
            status.update(label="Done", state="error" if error else "complete")
    add_usage(counter)
    seconds = time.perf_counter() - start
    if error:
        d["last_note"] = f"The follow-up couldn't be answered: {error}"
        log_run("Answer from research", question, d["id"], seconds, counter, 0, "Failed", error=error)
        return
    if not force and (resp.different_topic or not resp.evidence_sufficient):
        st.session_state.pending_decision = {
            "idx": idx, "question": question, "missing": resp.missing.strip(), "different_topic": resp.different_topic,
        }
        log_run("Answer from research", question, d["id"], seconds, counter, 0, "Needs more research")
        return
    new_card = _no_new_sources(_strip_followup_fields(resp), card(d))
    _apply_update(idx, question, new_card, [], "Answer from research", seconds, counter)


def followup_research_further(idx: int, question: str, slot):
    """Fetch only the missing evidence (small tool budget), then update the same card."""
    d = st.session_state.dossiers[idx]
    counter = TokenCounter()
    pending: list = []
    start = time.perf_counter()
    result, error = None, None
    with slot.container():
        live = st.status(":shimmer[Looking for what's missing...]", type="compact")
        executor = build_agent_executor(cached_llm(), _tools(pending, lookup_budget=FURTHER_MAX_LOOKUPS),
                                        max_iterations=FURTHER_MAX_LOOKUPS + 2, max_execution_time=90)
        try:
            result = run_research(
                executor, research_further_instructions(evidence_pack(d), question, FURTHER_MAX_LOOKUPS),
                None, callbacks=[LiveProgress(live, "Updating the answer"), counter],
            )
            new_card = result.response
            if new_card is None:
                # Budget ran out before a final answer: keep what was found, answer from the evidence.
                _add_evidence(d, result.steps)
                live.update(label=":shimmer[Updating the answer from what was found...]")
                resp = answer_from_evidence(cached_llm(), evidence_pack(d), question, True, callbacks=[counter])
                new_card = _strip_followup_fields(resp) if resp else None
            if new_card is None:
                error = result.error or "No answer was produced."
            live.update(label="Research failed" if error else "Updated", state="error" if error else "complete")
        except Exception as e:
            error = str(e)
            live.update(label="Research failed", state="error")
    add_usage(counter)
    seconds = time.perf_counter() - start
    steps = result.steps if result else []
    _attach_saves(pending, d, question)
    if error:
        _add_evidence(d, steps)
        d["last_note"] = f"Researching further didn't finish: {error}"
        log_run("Research further", question, d["id"], seconds, counter,
                len([s for s in steps if s.tool in LOOKUP_TOOLS]), "Failed", error=error)
        return
    _apply_update(idx, question, new_card, steps, "Research further", seconds, counter)


# ---------- Components ----------

COMPOSER_PLACEHOLDER = "Ask a research question..."


def composer(disabled: bool, placeholder: str = COMPOSER_PLACEHOLDER):
    return st.chat_input(placeholder, key="composer", submit_mode="disable",
                         disabled=disabled, max_chars=2000)


def render_empty_state(disabled: bool):
    """Welcome screen with the chat bar centered on the page. Returns a submitted prompt."""
    st.space(96)
    st.title("What should I research?", text_alignment="center")
    st.caption("Ask anything, or start with one of the topics below.", text_alignment="center")
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
    st.caption("Here's how I'd research this. Adjust anything, then start.")

    current = []
    for n, s in enumerate(plan["steps"], 1):
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**Step {n}**")
                st.space("stretch")
                st.button("Remove", type="tertiary", key=f"del_{s['sid']}",
                          help=f"Remove step {n}", on_click=remove_plan_step, args=(s["sid"],))
            q = st.text_input(f"Step {n} question", value=s["question"], key=f"q_{s['sid']}",
                              label_visibility="collapsed", placeholder="What should this step find out?",
                              persist_state="session")
            srcs = st.segmented_control(f"Step {n} sources", plan["allowed"], selection_mode="multi",
                                        default=s["sources"], key=f"src_{s['sid']}",
                                        persist_state="session")
            if not srcs:
                st.caption(":orange[Pick at least one source for this step.]")
            elif s.get("expected"):
                st.caption(f"Looking for: {s['expected']}")
            current.append({"question": (q or "").strip(), "sources": list(srcs or [])})

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


def _citations(ids: list[int], sources: list[str]) -> str:
    marks = []
    for i in ids:
        if 1 <= i <= len(sources):
            src = sources[i - 1].strip()
            marks.append(f"[\\[{i}\\]]({src})" if src.startswith("http") else f"\\[{i}\\]")
    return " ".join(marks)


def _reveal(text: str):
    """Word-by-word reveal. (st.write_stream imports pandas, which this machine blocks.)"""
    slot = st.empty()
    shown = ""
    for word in text.split(" "):
        shown += word + " "
        slot.markdown(shown + "▌")
        time.sleep(0.02)
    slot.markdown(text)


def change_record(v: dict) -> str:
    if v["route"] == "New research":
        return f"Started new research session · {_plural(v['searches'], 'lookup')}"
    parts = [f"Updated: {v['change'].rstrip('.')}."]
    if v["searches"] == 0:
        parts.append("No new searches were required.")
    else:
        parts.append(_plural(v["searches"], "new search") + "."
                     + (f" Added {_plural(v['new_sources'], 'new source')}." if v["new_sources"] else ""))
    return " ".join(parts)


def render_report(idx: int, locked: bool = False):
    """The session's answer card. `locked` disables every control while a follow-up runs."""
    d = st.session_state.dossiers[idx]
    r = card(d)

    if r is None:
        st.badge("Failed", color=STATUS_COLORS["Failed"])
        st.title(d["query"])
        if d.get("error"):
            st.error(f"The research couldn't finish: {d['error']}")
        elif d.get("result"):
            st.warning("The agent replied, but not in the expected format. Here's its raw answer.")
            st.markdown(d["result"].raw_text)
        st.button("Try again", type="primary", on_click=queue_prompt, args=(d["query"], "New research"),
                  key=f"retry_{idx}", disabled=locked)
        return

    status = dossier_status(d)
    v = d["versions"][-1]

    # Header: answer-type chip, status if it needs attention, and actions.
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(r.answer_type, color="gray")
        if status != "Complete":
            st.badge(status, color=STATUS_COLORS[status])
        st.space("stretch")
        with st.popover("Copy", type="tertiary", disabled=locked):
            st.caption("Use the copy button in the corner.")
            st.code(report_markdown(d["result"]), language="markdown", wrap_lines=True)
        st.button("Save", type="tertiary", on_click=save_report, args=(idx,), key=f"save_{idx}", disabled=locked)
        with st.popover("More", type="tertiary", disabled=locked):
            st.download_button("Download as Markdown", data=report_markdown(d["result"]), type="tertiary",
                               file_name=f"{r.topic[:40].strip().replace(' ', '_') or 'report'}.md",
                               mime="text/markdown", key=f"dl_{idx}", on_click="ignore")
            st.button("Start new research", type="tertiary", on_click=new_research, key=f"newfrom_{idx}")

    st.title(r.topic)
    st.caption(f"Version {v['n']} · Updated {d['updated']} · {_plural(len(r.sources), 'source')} · "
               f"{d.get('depth', 'Standard')} depth")
    st.caption(f":material/history: {change_record(v)}")
    if d.get("last_note"):
        st.warning(d["last_note"])

    st.subheader("Answer", divider="gray")
    if d.pop("fresh", False) and "```" not in r.answer and "|" not in r.answer and not locked:
        _reveal(r.answer)
    else:
        st.markdown(r.answer)

    if r.details or r.summary:
        st.subheader("Details", divider="gray")
        if r.details:
            st.markdown("\n".join(f"- {f.text} {_citations(f.source_ids, r.sources)}" for f in r.details))
        if r.summary:
            st.markdown(r.summary)

    if r.sources:
        st.subheader("Evidence", divider="gray")
        by_id = {e.source_id: e for e in r.evidence}
        lines = []
        for i, s in enumerate(r.sources, 1):
            e = by_id.get(i)
            title = e.title.strip() if e and e.title and e.title.strip().lower() != s.strip().lower() else ""
            supports = e.supports.strip().rstrip(".") if e and e.supports else ""
            extra = " · ".join(x for x in (title, supports) if x)
            lines.append(f"{i}. {format_source(s)}" + (f" — {extra}" if extra else ""))
        st.markdown("\n".join(lines))
        grounded = d.get("checks", {}).get("grounded")
        if grounded:
            hits, total = grounded
            st.caption(f"{hits} of {total} sources appear in the research results the agent read."
                       + ("" if hits == total else " Treat the others with care."))
    with st.expander(f"Research steps ({len(d['evidence'])})"):
        if not d["evidence"]:
            st.caption("No lookups so far.")
        for s in d["evidence"]:
            with st.expander(f"{step_name(s.tool)}: {s.tool_input[:80]}", type="step"):
                st.text(s.output[:2000] + ("…" if len(s.output) > 2000 else ""))

    st.subheader("Next steps", divider="gray")
    with st.container(horizontal=True, gap="small"):
        for j, step in enumerate(r.next_steps[:3]):
            st.button(step, key=f"next_{idx}_{v['n']}_{j}", on_click=queue_prompt, args=(step,), disabled=locked)
        st.button("Check sources", type="tertiary", key=f"check_{idx}", disabled=locked,
                  on_click=queue_prompt, args=("Verify the main claims in this answer against fresh sources.",
                                               "Research further"))
        st.button("Ask follow-up", type="tertiary", key=f"ask_{idx}", disabled=locked,
                  on_click=prefill_composer, args=(f"About {r.topic}: ",))

    if len(d["versions"]) > 1:
        with st.expander(f"Change history ({len(d['versions'])} versions)"):
            for ver in reversed(d["versions"]):
                st.markdown(f"**Version {ver['n']}** · {ver['time']} · {ver['route']}  \n"
                            f"{ver['question']}  \n:gray[{change_record(ver)}]")


def render_decision_card():
    """Shown when a follow-up can't be answered from the current evidence."""
    dec = st.session_state.pending_decision
    with st.container(border=True):
        if dec["different_topic"]:
            st.markdown("**This looks like a different topic.**")
            st.caption(f"\"{dec['question']}\". Start a separate research session for it, "
                       "or answer from the current research anyway.")
        else:
            st.markdown("**The current research doesn't cover this yet.**")
            st.caption(f"\"{dec['question']}\"" + (f". Missing: {dec['missing']}" if dec["missing"] else ""))
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if dec["different_topic"]:
                st.button("Start new research", type="primary", on_click=decide, args=("new",), key="dec_new")
                st.button("Research further here", type="tertiary", on_click=decide, args=("further",),
                          key="dec_further")
                st.button("Answer anyway", type="tertiary", on_click=decide, args=("anyway",), key="dec_anyway")
            else:
                st.button("Research further", type="primary", on_click=decide, args=("further",), key="dec_further")
                st.button("Answer anyway", type="tertiary", on_click=decide, args=("anyway",), key="dec_anyway")
                st.button("Start new research", type="tertiary", on_click=decide, args=("new",), key="dec_new")
            st.space("stretch")
            if not dec["different_topic"]:
                st.caption(f"Research further: up to {_plural(FURTHER_MAX_LOOKUPS, 'search')}")


def _further_estimate() -> str:
    past = [r for r in st.session_state.runs if r["kind"] == "Research further" and r["status"] == "Complete"]
    tokens = (sum(r["tokens_in"] + r["tokens_out"] for r in past) // len(past)) if past else 8000
    cost = estimate_cost(int(tokens * 0.85), int(tokens * 0.15))
    return f"Up to {_plural(FURTHER_MAX_LOOKUPS, 'search')} · ~{tokens:,} tokens" + (f" · ~${cost:.4f}" if cost else "")


def render_followup_bar(disabled: bool = False):
    """Follow-up mode selector, shown beside (just above) the composer when a card is open."""
    hints = {
        "Answer from this research": "Uses the evidence already gathered · usually 0 searches",
        "Research further": _further_estimate(),
        "New research": "Starts a separate research session",
    }
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.segmented_control("Follow-up mode", FOLLOWUP_MODES, default=FOLLOWUP_MODES[0], required=True,
                             key="followup_mode", label_visibility="collapsed", persist_state="session",
                             disabled=disabled)
        st.space("stretch")
        st.caption(hints[followup_mode()])


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
