"""Shared state, helpers and components for the Research Studio pages."""

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
    TokenCounter,
    build_agent_executor,
    get_llm,
    make_clarifying_questions,
    make_plan,
    plan_to_instructions,
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

DEPTHS = {
    "Quick": {"steps": "1 to 2", "max_iterations": 6, "max_seconds": 60,
              "instruction": "Research depth: quick. Use at most 2 lookups. Give 2 to 3 key findings "
                             "and a short report of 1 to 2 paragraphs."},
    "Standard": {"steps": "2 to 4", "max_iterations": 10, "max_seconds": 150, "instruction": ""},
    "Deep": {"steps": "4 to 6", "max_iterations": 20, "max_seconds": 300,
             "instruction": "Research depth: deep. Cross-check important facts across several lookups. "
                            "Give 5 to 7 key findings and a detailed report of 4 to 6 paragraphs."},
}
DEPTH_CAPTIONS = ["Fewer lookups, short report", "Balanced", "More lookups, longer report"]

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

def init_state():
    defaults = {
        "dossiers": [],  # every research run this session (also the run log)
        "active": None,  # index of the report on screen
        "pending_clarify": None,  # clarifying questions awaiting answers
        "pending_plan": None,  # plan awaiting approval
        "pending_saves": [],  # file writes awaiting approval
        "queued_prompt": None,  # prompt queued by a button
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


def dossier_status(d: dict) -> str:
    if d.get("error"):
        return "Failed"
    if any(r.get("dossier") == d["id"] for r in st.session_state.pending_saves):
        return "Needs approval"
    return "Complete"


def source_count(d: dict) -> int:
    r = d.get("result")
    return len(r.response.sources) if r and r.response else 0


def report_markdown(result: AgentResult) -> str:
    r = result.response
    if r is None:
        return result.raw_text
    findings = "\n".join(f"- {f.text}" for f in r.key_findings)
    sources = "\n".join(f"{i}. {s}" for i, s in enumerate(r.sources, 1)) or "None"
    parts = [f"# {r.topic}", r.answer, f"## Key findings\n{findings}" if findings else "",
             f"## Detailed report\n{r.summary}", f"## Sources\n{sources}"]
    return "\n\n".join(p for p in parts if p) + "\n"


def history_for_agent() -> list[tuple[str, str]]:
    turns = []
    for d in st.session_state.dossiers:
        r = d.get("result")
        if r is None or r.response is None:
            continue
        turns.append(("user", d["query"]))
        turns.append(("assistant", r.response.answer or r.response.summary))
    return turns


def scope_estimate(steps: list[dict]) -> str:
    lookups = sum(len(s["sources"]) for s in steps)
    secs = 10 + 8 * lookups
    when = "under a minute" if secs < 60 else f"about {secs // 60}–{secs // 60 + 1} minutes"
    return f"{lookups} lookup{'s' if lookups != 1 else ''} · {when}"


# ---------- Automatic quality checks (run on every report) ----------

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


def quality_checks(result: AgentResult | None, plan_steps: list[dict] | None) -> dict:
    if result is None:
        return {"format": False, "grounded": None, "plan": None}
    used = {s.tool for s in result.steps}
    outputs = "\n".join(s.output for s in result.steps).lower()
    grounded = None
    if result.response and result.response.sources:
        hits = sum(_is_grounded(s, outputs, used) for s in result.response.sources)
        grounded = (hits, len(result.response.sources))
    plan = None
    if plan_steps:
        needed = {SOURCE_TOOLS[x] for s in plan_steps for x in s["sources"]}
        plan = needed <= used
    return {"format": result.response is not None, "grounded": grounded, "plan": plan}


# ---------- Callbacks ----------

def queue_prompt(prompt: str):
    st.session_state.queued_prompt = prompt


def prefill_composer(text: str):
    st.session_state.composer = text


def pick_suggestion():
    choice = st.session_state.get("suggestion")
    if choice:
        queue_prompt(SUGGESTIONS[choice])
    st.session_state.suggestion = None


def new_research():
    st.session_state.active = None
    st.session_state.pending_plan = None
    st.session_state.pending_clarify = None


def open_report(idx: int):
    new_research()
    st.session_state.active = idx


def clear_history():
    new_research()
    st.session_state.dossiers = []
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

    def __init__(self, status):
        self.status = status
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
        self.status.update(label=self._label("Drafting report"))
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


# ---------- Run stages ----------

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
    """Run the agent, showing a live progress strip in `slot`, then store the report."""
    sources = enabled_sources()
    cfg = DEPTHS[depth()]
    history = history_for_agent()
    pending: list = []
    counter = TokenCounter()
    result, error = None, None

    tools = []
    if "Web" in sources:
        tools.append(search_tool)
    if "Wikipedia" in sources:
        tools.append(wiki_tool)
    if "Documents" in sources:
        tools.append(make_documents_tool(list(st.session_state.docs.values())))
    tools.append(make_request_save_tool(pending))

    instructions = context + plan_to_instructions(steps)
    if cfg["instruction"]:
        instructions += f"\n\n{cfg['instruction']}"

    with slot.container():
        _working_header("Researching", query)
        st.caption(" · ".join(filter(None, [
            f"Following your {len(steps)}-step plan" if steps else None, f"{depth()} depth", ", ".join(sources),
        ])))
        live = st.status(":shimmer[Planning searches...]", type="compact")
        executor = build_agent_executor(cached_llm(), tools, max_iterations=cfg["max_iterations"],
                                        max_execution_time=cfg["max_seconds"])
        start = time.perf_counter()
        try:
            result = run_research(executor, query + instructions, history,
                                  callbacks=[LiveProgress(live), counter])
            if result.hit_limit:
                error = (f"Stopped at the {depth()} depth limit before finishing. "
                         "Try again with a deeper setting or a narrower question.")
            live.update(label="Research stopped" if error else "Research complete",
                        state="error" if error else "complete")
        except Exception as e:
            error = str(e)
            live.update(label="Research failed", state="error")

    add_usage(counter)
    d = {
        "id": uuid.uuid4().hex[:8],
        "query": query,
        "title": result.response.topic if result and result.response else query[:60],
        "result": result,
        "error": error,
        "seconds": time.perf_counter() - start,
        "created": datetime.now().strftime("%H:%M"),
        "fresh": True,
        "mode": mode(),
        "depth": depth(),
        "sources": sources,
        "tokens_in": counter.input_tokens,
        "tokens_out": counter.output_tokens,
        "checks": quality_checks(result, steps),
    }
    st.session_state.dossiers.append(d)
    st.session_state.active = len(st.session_state.dossiers) - 1
    for req in pending:
        req["dossier"] = d["id"]
        req["question"] = query
    st.session_state.pending_saves.extend(pending)


# ---------- Components ----------

COMPOSER_PLACEHOLDER = "Ask a research question..."


def composer(disabled: bool):
    return st.chat_input(COMPOSER_PLACEHOLDER, key="composer", submit_mode="disable",
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
    st.space("large")
    first = {
        "Ask first": "The agent asks a quick question or two if your request is ambiguous, "
                     "then proposes a short plan for you to review",
        "Review plan": "The agent proposes a short research plan for you to review",
        "Auto-research": "The agent plans its research on its own",
    }[mode()]
    with st.container(border=True):
        st.markdown("**How it works**")
        st.markdown(f"1. {first}\n"
                    f"2. It searches your selected sources: {', '.join(enabled_sources()) or 'none selected'}\n"
                    "3. You get a sourced report you can explore, check and save")
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


def render_report(idx: int):
    d = st.session_state.dossiers[idx]
    status = dossier_status(d)

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(status, color=STATUS_COLORS[status])
        st.caption(f"{d['created']} · {d['seconds']:.0f}s · {source_count(d)} sources · {d.get('depth', 'Standard')} depth")

    if d.get("error"):
        st.title(d["query"])
        st.error(f"The research couldn't finish: {d['error']}")
        st.button("Try again", type="primary", on_click=queue_prompt, args=(d["query"],), key=f"retry_{idx}")
        return

    result: AgentResult = d["result"]
    r = result.response
    if r is None:
        st.title(d["query"])
        st.warning("The agent replied, but not in the expected format. Here's its raw answer.")
        st.markdown(result.raw_text)
        return

    st.title(r.topic)
    if d["query"].strip().lower() != r.topic.strip().lower():
        st.caption(d["query"])

    if r.answer:
        with st.container(border=True):
            if d.pop("fresh", False):
                _reveal(r.answer)
            else:
                st.markdown(r.answer)

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.button("Follow up", icon=":material/reply:", key=f"follow_{idx}",
                  on_click=prefill_composer, args=(f"Follow-up on {r.topic}: ",))
        st.button("Save", type="tertiary", on_click=save_report, args=(idx,), key=f"save_{idx}")
        st.download_button("Download", data=report_markdown(result), type="tertiary",
                           file_name=f"{r.topic[:40].strip().replace(' ', '_') or 'report'}.md",
                           mime="text/markdown", key=f"dl_{idx}", on_click="ignore")
        with st.popover("Copy", type="tertiary"):
            st.caption("Use the copy button in the corner.")
            st.code(report_markdown(result), language="markdown", wrap_lines=True)

    if r.key_findings:
        st.subheader("Key findings")
        for j, f in enumerate(r.key_findings):
            with st.container(border=True):
                st.markdown(f"{f.text} {_citations(f.source_ids, r.sources)}")
                with st.container(horizontal=True, gap="small"):
                    st.button("Explore", type="tertiary", key=f"explore_{idx}_{j}", on_click=queue_prompt,
                              args=(f"Explore this further, in the context of {r.topic}: {f.text}",))
                    st.button("Check sources", type="tertiary", key=f"check_{idx}_{j}", on_click=queue_prompt,
                              args=(f"Check this claim against fresh sources and say whether it holds up: {f.text}",))
                    st.button("Ask follow-up", type="tertiary", key=f"ask_{idx}_{j}", on_click=prefill_composer,
                              args=(f"About \"{f.text[:80]}\": ",))

    st.subheader("Detailed report")
    st.markdown(r.summary)

    st.subheader("Sources")
    if r.sources:
        st.markdown("\n".join(f"{i}. {format_source(s)}" for i, s in enumerate(r.sources, 1)))
        grounded = d.get("checks", {}).get("grounded")
        if grounded:
            hits, total = grounded
            st.caption(f"{hits} of {total} sources appear in the research results the agent read."
                       + ("" if hits == total else " Treat the others with care."))
    else:
        st.caption("No sources were cited.")

    steps = result.steps
    with st.expander(f"Research steps ({len(steps)})"):
        if not steps:
            st.caption("The agent answered without calling any tools.")
        for s in steps:
            with st.expander(f"{step_name(s.tool)}: {s.tool_input[:80]}", type="step"):
                st.text(s.output[:2000] + ("…" if len(s.output) > 2000 else ""))


def render_approval_notice():
    n = len(st.session_state.pending_saves)
    if not n:
        return
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.markdown(f":orange[:material/pending_actions:] **{n} file{' is' if n == 1 else 's are'} waiting for your approval**")
        st.space("stretch")
        if st.button("Review", key="review_approvals"):
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
