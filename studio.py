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

FURTHER_MAX_LOOKUPS = 2  # search budget when a follow-up needs evidence the session doesn't have
EVIDENCE_PACK_CHARS = 6000  # cap on stored snippets sent with each follow-up

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
        "queued_mode": None,  # "New research" when a button must start a separate session
        # Saves live in this visitor's session only. On a shared server (e.g. Streamlit
        # Cloud) a disk folder would be visible to every visitor and wiped on restart.
        "saved_files": [],
        "docs": {},  # uploaded documents: key -> {"name", "text"}
        "usage": {"input": 0, "output": 0},  # tokens across every model call
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    _migrate_old_sessions()


def _migrate_old_sessions():
    """Convert sessions saved by the earlier one-card-per-session format into threads,
    so an open browser tab survives an app update."""
    for d in st.session_state.dossiers:
        if "turns" in d:
            continue
        r = d.get("result")
        c = r.response if r else None
        d["turns"] = [{"question": d.get("query", ""), "route": "New research", "card": c,
                       "error": d.get("error"), "raw": r.raw_text if r else "", "time": d.get("created", ""),
                       "seconds": d.get("seconds", 0), "searches": len(d.get("evidence", [])),
                       "new_sources": len(c.sources) if c else 0, "checks": d.get("checks", {}), "fresh": False}]
        d.setdefault("evidence", [])
        d["sources"] = list(c.sources) if c else []
        d.setdefault("updated", d.get("created", ""))


def mode() -> str:
    return st.session_state.get("mode") or "Auto-research"


def depth() -> str:
    return st.session_state.get("depth") or "Standard"


def enabled_sources() -> list[str]:
    """Sources picked in the sidebar. Documents only count once something is uploaded."""
    picked = st.session_state.get("sources_pick", ["Web", "Wikipedia"]) or []
    return [x for x in SOURCE_OPTIONS if x in picked and (x != "Documents" or st.session_state.docs)]


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
    """The most recent successful answer card in a session."""
    for t in reversed(d["turns"]):
        if t["card"] is not None:
            return t["card"]
    return None


def dossier_status(d: dict) -> str:
    if card(d) is None:
        return "Failed"
    if any(r.get("dossier") == d["id"] for r in st.session_state.pending_saves):
        return "Needs approval"
    return "Complete"


def source_count(d: dict) -> int:
    return len(d["sources"])


def _card_markdown(c: ResearchResponse) -> str:
    parts = [f"*{c.answer_type}*", c.answer]
    if c.details:
        parts.append("\n".join(f"- {f.text}" + (f" [{', '.join(map(str, f.source_ids))}]" if f.source_ids else "")
                               for f in c.details))
    if c.summary:
        parts.append(c.summary)
    if c.sources:
        parts.append("Sources:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(c.sources, 1)))
    return "\n\n".join(parts)


def report_markdown(d: dict) -> str:
    """The whole research thread as Markdown: every question with its answer."""
    parts = [f"# {d['title']}"]
    for t in d["turns"]:
        parts.append(f"## {t['question']}")
        parts.append(_card_markdown(t["card"]) if t["card"] else f"_{t['error'] or 'No answer.'}_")
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
    """Compact context for follow-ups.

    Sends the original question, the last few questions with short answers, the session's
    numbered sources, and the newest tool snippets up to EVIDENCE_PACK_CHARS. Never the
    whole raw conversation or full pages.
    """
    parts = [f"ORIGINAL QUESTION: {d['query']}"]
    answered = [t for t in d["turns"] if t["card"] is not None][-4:]
    if answered:
        parts.append("CONVERSATION SO FAR:\n" + "\n\n".join(
            f"Q: {t['question']}\nA: {t['card'].answer[:600]}" for t in answered
        ))
    if d["sources"]:
        parts.append("SOURCES:\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(d["sources"], 1)))
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
    plan["steps"].append({"sid": uuid.uuid4().hex[:8], "question": "", "sources": [plan["allowed"][0]], "expected": ""})


def remove_plan_step(sid: str):
    plan = st.session_state.pending_plan
    plan["steps"] = [s for s in plan["steps"] if s["sid"] != sid]


def new_research_job(query: str, idx: int | None = None) -> dict:
    """A full-research job, following the sidebar's research mode.

    `idx` is the open thread the answer is appended to; None starts a new session.
    """
    job = {
        "Ask first": {"kind": "clarify", "query": query},
        "Review plan": {"kind": "plan", "query": query, "context": ""},
        "Auto-research": {"kind": "research", "query": query, "steps": None, "context": ""},
    }[mode()]
    job["idx"] = idx
    return job


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
    else:  # research it in the same thread, below the earlier messages
        st.session_state.job = new_research_job(dec["question"], dec["idx"])


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
    """The question as a chat message, with what's happening to it."""
    _user_message(title)
    st.badge(badge, color=STATUS_COLORS[badge])


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


def _key(source: str) -> str:
    return source.strip().lower().rstrip("/")


def _new_turn(question: str, route: str, c: ResearchResponse | None, seconds: float, searches: int = 0,
              new_sources: int = 0, error: str | None = None, raw: str = "", checks: dict | None = None) -> dict:
    return {"question": question, "route": route, "card": c, "error": error, "raw": raw, "time": now(),
            "seconds": seconds, "searches": searches, "new_sources": new_sources, "checks": checks or {},
            "fresh": c is not None}


def _add_sources(d: dict, c: ResearchResponse) -> int:
    """Add a card's sources to the session's source list. Returns how many were new."""
    known = {_key(s) for s in d["sources"]}
    added = 0
    for s in c.sources:
        if _key(s) not in known:
            d["sources"].append(s)
            known.add(_key(s))
            added += 1
    return added


def execute(query: str, steps: list[dict] | None, slot, context: str = "", idx: int | None = None,
            show_question: bool = True):
    """Run full research with live progress and store the answer as a turn.

    With `idx`, the answer is appended to that thread (below the earlier messages) instead of
    starting a new session. `show_question=False` when the question bubble is already on screen.
    """
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
        if show_question:
            _user_message(query)
        # When continuing a follow-up, `slot` is already that answer's assistant bubble.
        with st.chat_message("assistant") if show_question else st.container():
            st.caption(" · ".join(filter(None, [
                f"Following your {len(steps)}-step plan" if steps else None, f"{depth()} depth", ", ".join(sources),
            ])))
            live = st.status(":shimmer[Planning searches...]", type="compact")
        scroll_to_bottom()
        executor = build_agent_executor(cached_llm(), _tools(pending), max_iterations=cfg["max_iterations"],
                                        max_execution_time=cfg["max_seconds"])
        try:
            # A new session starts fresh: no earlier conversation is sent.
            result = run_research(executor, query + instructions, None, callbacks=[LiveProgress(live), counter])
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

    seconds = time.perf_counter() - start
    add_usage(counter)
    response = result.response if result and not error else None
    lookups = [s for s in (result.steps if result else []) if s.tool in LOOKUP_TOOLS]
    checks = quality_checks(response, lookups, steps)
    status = "Failed" if response is None else "Complete"

    if idx is not None:  # continue the open thread: a new message below the earlier ones
        d = st.session_state.dossiers[idx]
        _add_evidence(d, lookups)
        n_sources = _add_sources(d, response) if response else 0
        d["turns"].append(_new_turn(query, "New research", response, seconds, len(lookups), n_sources, error,
                                    result.raw_text if result else "", checks))
        d["updated"] = now()
        _attach_saves(pending, d, query)
        log_run("New research", query, d["id"], seconds, counter, len(lookups), status, checks, error)
        return

    d = {
        "id": uuid.uuid4().hex[:8],
        "query": query,
        "title": response.topic if response else query[:60],
        "created": now(),
        "updated": now(),
        "mode": mode(),
        "depth": depth(),
        "enabled_sources": sources,
        "sources": [],  # every source cited anywhere in this session
        "evidence": [],  # every lookup made in this session
        "turns": [],
    }
    _add_evidence(d, lookups)
    n_sources = _add_sources(d, response) if response else 0
    d["turns"].append(_new_turn(query, "New research", response, seconds, len(lookups), n_sources, error,
                                result.raw_text if result else "", checks))
    st.session_state.dossiers.append(d)
    st.session_state.active = len(st.session_state.dossiers) - 1
    _attach_saves(pending, d, query)
    log_run("New research", query, d["id"], seconds, counter, len(lookups), status, checks, error)


# ---------- Run stages: follow-ups in the same session ----------

def _append_answer(idx: int, question: str, new_card: ResearchResponse, new_steps: list, route: str,
                   seconds: float, counter: TokenCounter):
    """Add a follow-up answer as a new turn below the earlier ones."""
    d = st.session_state.dossiers[idx]
    _add_evidence(d, new_steps)
    searches = len([s for s in new_steps if s.tool in LOOKUP_TOOLS and not s.output.startswith("Search budget")])
    new_sources = _add_sources(d, new_card)
    checks = quality_checks(new_card, d["evidence"], None)
    d["turns"].append(_new_turn(question, route, new_card, seconds, searches, new_sources, checks=checks))
    d["updated"] = now()
    log_run(route, question, d["id"], seconds, counter, searches, "Complete", checks)


def _append_failure(idx: int, question: str, route: str, error: str, seconds: float):
    d = st.session_state.dossiers[idx]
    d["turns"].append(_new_turn(question, route, None, seconds, error=error))
    d["updated"] = now()


def _strip_followup_fields(resp) -> ResearchResponse:
    return ResearchResponse.model_validate(
        resp.model_dump(exclude={"evidence_sufficient", "different_topic", "missing"})
    )


def _known_sources_only(c: ResearchResponse, known: list[str]) -> ResearchResponse:
    """Answering from existing research runs no tools, so it can't legitimately cite new sources.
    Drop any source the session hasn't seen, and renumber the citations that remain."""
    allowed = {_key(s) for s in known}
    keep = [i for i, s in enumerate(c.sources, 1) if _key(s) in allowed]
    if len(keep) == len(c.sources):
        return c
    remap = {old: new for new, old in enumerate(keep, 1)}
    return c.model_copy(update={
        "sources": [c.sources[i - 1] for i in keep],
        "evidence": [e.model_copy(update={"source_id": remap[e.source_id]}) for e in c.evidence
                     if e.source_id in remap],
        "details": [f.model_copy(update={"source_ids": [remap[i] for i in f.source_ids if i in remap]})
                    for f in c.details],
    })


def _user_message(question: str):
    with st.chat_message("user"):
        st.markdown(question)


def followup_from_evidence(idx: int, question: str, slot, force: bool = False):
    """Answer a follow-up from the session's evidence first (no tool calls).

    If that evidence doesn't cover it, search for what's missing (within FURTHER_MAX_LOOKUPS).
    A different topic gets full research, still in this thread, so nothing earlier disappears.
    """
    d = st.session_state.dossiers[idx]
    counter = TokenCounter()
    start = time.perf_counter()
    resp, error = None, None
    with slot.container():
        _user_message(question)
        bubble = st.chat_message("assistant")  # one bubble for this answer, even if it escalates
        with bubble:
            with st.status(":shimmer[Checking the research so far...]", type="compact") as status:
                try:
                    resp = answer_from_evidence(cached_llm(), evidence_pack(d), question, force, callbacks=[counter])
                    if resp is None:
                        error = "The model returned no answer."
                except Exception as e:
                    error = str(e)
                status.update(label="Checked the research so far", state="error" if error else "complete")
        scroll_to_bottom()
    add_usage(counter)
    seconds = time.perf_counter() - start
    if error:
        _append_failure(idx, question, "Answer from research", f"This follow-up couldn't be answered: {error}", seconds)
        log_run("Answer from research", question, d["id"], seconds, counter, 0, "Failed", error=error)
        return
    if not force and resp.different_topic:
        log_run("Answer from research", question, d["id"], seconds, counter, 0, "Different topic")
        execute(question, None, bubble, idx=idx, show_question=False)
        return
    if not force and not resp.evidence_sufficient:
        log_run("Answer from research", question, d["id"], seconds, counter, 0, "Needs more research")
        followup_research_further(idx, question, bubble, show_question=False)
        return
    new_card = _known_sources_only(_strip_followup_fields(resp), d["sources"])
    _append_answer(idx, question, new_card, [], "Answer from research", seconds, counter)


def followup_research_further(idx: int, question: str, slot, show_question: bool = True):
    """Fetch only the missing evidence (small search budget), then answer below the earlier turns."""
    d = st.session_state.dossiers[idx]
    counter = TokenCounter()
    pending: list = []
    start = time.perf_counter()
    result, error, new_card = None, None, None
    with slot.container():
        if show_question:
            _user_message(question)
        bubble = st.chat_message("assistant") if show_question else st.container()
        live = bubble.status(":shimmer[Searching for what's missing...]", type="compact")
        scroll_to_bottom()
        executor = build_agent_executor(cached_llm(), _tools(pending, lookup_budget=FURTHER_MAX_LOOKUPS),
                                        max_iterations=FURTHER_MAX_LOOKUPS + 2, max_execution_time=90)
        try:
            result = run_research(
                executor, research_further_instructions(evidence_pack(d), question, FURTHER_MAX_LOOKUPS),
                None, callbacks=[LiveProgress(live, "Writing the answer"), counter],
            )
            new_card = result.response
            if new_card is None:
                # Budget ran out before a final answer: keep what was found, answer from the evidence.
                _add_evidence(d, result.steps)
                live.update(label=":shimmer[Writing the answer from what was found...]")
                resp = answer_from_evidence(cached_llm(), evidence_pack(d), question, True, callbacks=[counter])
                new_card = _strip_followup_fields(resp) if resp else None
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
        _add_evidence(d, steps)
        _append_failure(idx, question, "Research further", f"Researching further didn't finish: {error}", seconds)
        log_run("Research further", question, d["id"], seconds, counter,
                len([s for s in steps if s.tool in LOOKUP_TOOLS]), "Failed", error=error)
        return
    _append_answer(idx, question, new_card, steps, "Research further", seconds, counter)


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


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sup(i: int, sources: list[str]) -> str:
    """A small superscript citation, linked to its source when it's a URL."""
    mark = str(i).translate(_SUPERSCRIPT)
    if 1 <= i <= len(sources) and sources[i - 1].strip().startswith("http"):
        return f"[{mark}]({sources[i - 1].strip()})"
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
    if t["route"] == "New research":
        base = f"Researched · {_plural(t['searches'], 'lookup')}"
    elif t["searches"] == 0:
        base = "Answered from existing research · 0 new searches"
    else:
        base = f"Researched further · {_plural(t['searches'], 'new search')}"
    if t["new_sources"] and t["route"] != "New research":
        base += f" · {_plural(t['new_sources'], 'new source')}"
    return f"{base} · {t['time']}"


def _source_line(c: ResearchResponse) -> str:
    """One compact line of numbered source links, e.g. '¹ nature.com · ² wikipedia.org'."""
    items = []
    for i, s in enumerate(c.sources, 1):
        s = s.strip()
        label = domain(s) if s.startswith("http") else s
        items.append(f"{str(i).translate(_SUPERSCRIPT)} [{label}]({s})" if s.startswith("http")
                     else f"{str(i).translate(_SUPERSCRIPT)} {label}")
    return " · ".join(items)


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
    c = t["card"]
    if c is None:
        st.error(t["error"] or "No answer was produced.")
        if t.get("raw"):
            with st.expander("Raw reply"):
                st.markdown(t["raw"])
        if last:
            st.button("Try again", type="tertiary", icon=":material/refresh:", on_click=queue_prompt,
                      args=(t["question"],), key=f"retry_{idx}_{n}", disabled=locked)
        return

    answer = _inline_citations(c.answer, c.sources)
    if t.pop("fresh", False) and last and not locked and "```" not in answer and "|" not in answer:
        _reveal(answer)
    else:
        st.markdown(answer)

    if c.details:
        st.markdown("\n".join(f"- {_inline_citations(f.text, c.sources)}{_citations(f.source_ids, c.sources)}"
                              for f in c.details))
    if c.summary:
        st.markdown(_inline_citations(c.summary, c.sources))

    meta = f"{c.answer_type} · {turn_status(t)}"
    st.caption(f"{_source_line(c)}  \n{meta}" if c.sources else meta)
    grounded = t["checks"].get("grounded")
    if last and grounded and grounded[0] < grounded[1]:
        st.caption(f":orange[{grounded[1] - grounded[0]} of {grounded[1]} sources weren't in the results "
                   "the agent read. Treat them with care.]")


def render_session(idx: int, locked: bool = False):
    """The research thread: every question with its answer, newest at the bottom.

    `locked` disables every control while a follow-up is running.
    """
    d = st.session_state.dossiers[idx]
    c = card(d)
    status = dossier_status(d)

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if status != "Complete":
            st.badge(status, color=STATUS_COLORS[status])
        st.caption(f"{_plural(len(d['turns']), 'question')} · {_plural(len(d['sources']), 'source')} · "
                   f"{_plural(len(d['evidence']), 'lookup')} · {d.get('depth', 'Standard')} depth")
        st.space("stretch")
        if c is not None:
            with st.popover("More", type="tertiary", disabled=locked):
                st.button("Save to library", type="tertiary", on_click=save_report, args=(idx,), key=f"save_{idx}")
                st.download_button("Download as Markdown", data=report_markdown(d), type="tertiary",
                                   file_name=f"{d['title'][:40].strip().replace(' ', '_') or 'research'}.md",
                                   mime="text/markdown", key=f"dl_{idx}", on_click="ignore")
                st.caption("Copy as Markdown")
                st.code(report_markdown(d), language="markdown", wrap_lines=True, height=200)

    if d["evidence"]:
        with st.expander(f"Research steps · {_plural(len(d['evidence']), 'lookup')}"):
            for s in d["evidence"]:
                with st.expander(f"{step_name(s.tool)}: {s.tool_input[:80]}", type="step"):
                    st.text(s.output[:2000] + ("…" if len(s.output) > 2000 else ""))

    # Every message, oldest first. Earlier ones are only ever re-drawn, never replaced.
    fresh = d["turns"][-1].get("fresh", False)
    for n, t in enumerate(d["turns"]):
        _render_turn(idx, n, t, last=n == len(d["turns"]) - 1, locked=locked)

    last = d["turns"][-1]["card"]
    if last is not None and last.next_steps:
        # Drawn disabled while locked rather than left out: leaving it out would keep the
        # previous run's clickable copies on screen until the run ends.
        # Compact chips that wrap tightly; picking one asks it as a follow-up.
        key = f"related_{idx}_{len(d['turns'])}"
        st.pills("Related", last.next_steps[:3], key=key, on_change=pick_related, args=(key,),
                 disabled=locked)

    if fresh or locked:  # a new message just arrived, or one is being written below
        scroll_to_bottom()


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
