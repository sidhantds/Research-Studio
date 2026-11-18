import uuid

import streamlit as st

import studio as s
from agent import route_request

# The research page is one active session shown as a chat: every question and answer is a
# message, and new ones are appended below the old ones.
#
# Follow-ups never start a new session on their own. The follow-up mode above the composer
# decides what a follow-up may do:
#   Use current research (default): answer from the session's evidence; never searches.
#   Research further: up to 2 new searches in this session.
#   New research: start a separate research case.
#
# Slow work runs as a "job" in its own script run:
#   1. A click or a submitted question only records the job and reruns.
#   2. On the next run, `body` redraws the session without live controls, so nothing stale can be
#      clicked (a click would abort the run), and the new message streams into `progress`,
#      which sits below the thread, exactly where the finished answer will land.
body = st.empty()
progress = st.container()

job = st.session_state.pop("job", None)  # popped first: an interrupted job never restarts itself


def followup_mode_control(disabled: bool):
    st.segmented_control("Follow-up mode", s.FOLLOWUP_MODES, selection_mode="single", required=True,
                         key="followup_mode", disabled=disabled, label_visibility="collapsed",
                         help=s.FOLLOWUP_HELP)


if job:
    kind = job["kind"]
    thread = job.get("idx")  # the session this job adds to; None = a new session
    if kind in ("further", "start", "research") or (kind == "followup" and job["mode"] != s.FOLLOWUP_MODES[0]):
        # Searching was a deliberate choice for this one question; go back to the default.
        st.session_state.followup_mode = s.FOLLOWUP_MODES[0]
    if thread is not None:
        with body.container():
            s.render_session(thread, locked=True)  # earlier messages stay on screen, locked
    # Redraw the controls below the thread, locked. If they were left out of this run, the
    # previous run's copies would stay on screen and clickable.
    s.render_approval_notice(disabled=True)
    if thread is not None:
        followup_mode_control(disabled=True)
    s.composer(disabled=True, placeholder="Working...")

    if kind == "followup":
        s.run_followup(thread, job["question"], job["mode"], progress)
    elif kind == "context":  # answer from current research even though it may be thin
        s.followup_from_session(thread, job["question"], job.get("route"), progress,
                                force=job.get("force", False), show_question=True)
    elif kind == "further":
        s.followup_research_further(thread, job["question"], progress, route=job.get("route"))
    elif kind == "start":
        # A new session: route first. Code and plain explanations are answered directly with no
        # research; everything else follows the research mode in Settings.
        with progress:
            with st.status(":shimmer[Reading the request...]", type="compact") as status:
                route = route_request(s.cached_llm(), job["query"], None)
                status.update(label=f"{s.TYPE_LABELS.get(route.response_type, 'Research')} · {route.reason}",
                              state="complete")
        if route.response_type in ("code", "explanation") and not route.needs_tools:
            s.answer_directly(job["query"], route, progress)
        else:
            nxt = s.new_research_job(job["query"], None, route.model_dump())
            if nxt["kind"] == "research":
                s.execute(job["query"], None, progress, idx=None, route=route)
            else:
                st.session_state.job = nxt  # clarify or plan first, as their own job
    elif kind == "plan":
        s.draft_plan(job["query"], job["context"], progress)
        st.session_state.pending_plan["idx"] = thread
        st.session_state.pending_plan["route"] = job.get("route")
    elif kind == "research":
        s.execute(job["query"], job["steps"], progress, job["context"], idx=thread, route=job.get("route"))
    st.rerun()

sources = s.enabled_sources()
if not sources:
    st.warning("Turn on at least one source in the sidebar to start researching.")

active = st.session_state.active
clarify = st.session_state.pending_clarify
plan = st.session_state.pending_plan
waiting = bool(clarify or plan)  # a card is waiting for the user's input below the thread

new_job = None
typed = None
on_welcome = False
with body.container():
    if active is not None:
        s.render_session(active, locked=waiting)  # the whole session, oldest message first
    if clarify:
        clarified = s.render_clarify_card()
        if clarified:
            new_job = {"kind": "plan", "query": clarified[0], "context": clarified[1], "idx": clarify.get("idx"),
                       "route": clarify.get("route")}
    elif plan:
        approved = s.render_plan_card()
        if approved:
            new_job = {"kind": "research", "query": approved[0], "steps": approved[1], "context": approved[2],
                       "idx": plan.get("idx"), "route": plan.get("route")}
    elif active is None:
        on_welcome = True
        typed = s.render_empty_state(disabled=not sources)  # chat bar centered on the page

s.render_approval_notice()

if not on_welcome:
    has_answer = active is not None and s.card(st.session_state.dossiers[active]) is not None
    if has_answer:
        followup_mode_control(disabled=waiting)
    fmode = st.session_state.followup_mode
    placeholder = {"Use current research": "Ask a follow-up about this research...",
                   "Research further": "Ask what to look up (up to 2 searches)...",
                   "New research": "Ask a new research question..."}[fmode] if has_answer else s.COMPOSER_PLACEHOLDER
    typed = s.composer(disabled=not sources or waiting, placeholder=placeholder)

prompt = typed or st.session_state.queued_prompt
st.session_state.queued_prompt = None
st.session_state.queued_mode = None

if prompt and sources and not new_job:
    st.session_state.pending_decision = None
    fmode = st.session_state.followup_mode
    if active is not None and s.card(st.session_state.dossiers[active]) is not None and fmode != "New research":
        # A follow-up in the active session: routed, then answered per the follow-up mode.
        new_job = {"kind": "followup", "idx": active, "question": prompt, "mode": fmode}
    else:
        # First question, "New research" mode, or a retry after a failed first answer: a new session.
        st.session_state.active = None
        new_job = {"kind": "start", "query": prompt}

if new_job:
    st.session_state.pending_plan = None
    st.session_state.pending_clarify = None
    st.session_state.job = new_job
    st.rerun()
