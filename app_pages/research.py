import uuid

import streamlit as st

import studio as s

# The research page is a chat thread: every question and answer is a message, and new ones are
# appended below the old ones. Nothing here ever swaps the thread out for something else; only
# "New research" in the sidebar starts an empty thread.
#
# Slow work (drafting a plan, researching, follow-ups) runs as a "job" in its own script run:
#   1. A click or a submitted question only records the job and reruns.
#   2. On the next run, `body` redraws the thread without live controls, so nothing stale can be
#      clicked (a click would abort the run), and the new message streams into `progress`,
#      which sits below the thread, exactly where the finished answer will land.
body = st.empty()
progress = st.container()

job = st.session_state.pop("job", None)  # popped first: an interrupted job never restarts itself

if job:
    kind = job["kind"]
    thread = job.get("idx")  # the open thread this job adds to; None = a new session
    if thread is not None:
        with body.container():
            s.render_session(thread, locked=True)  # earlier messages stay on screen, locked
    # Redraw the controls below the thread, locked. If they were left out of this run, the
    # previous run's copies would stay on screen and clickable.
    s.render_approval_notice(disabled=True)
    s.composer(disabled=True, placeholder="Researching...")

    if kind == "context":
        s.followup_from_evidence(thread, job["question"], progress, force=job.get("force", False))
    elif kind == "further":
        s.followup_research_further(thread, job["question"], progress)
    elif kind == "clarify":
        questions = s.draft_clarify(job["query"], progress)
        if questions:
            st.session_state.pending_clarify = {"id": uuid.uuid4().hex[:8], "query": job["query"],
                                                "questions": questions, "idx": thread}
        else:  # nothing to ask: go straight to the plan, as its own job
            st.session_state.job = {"kind": "plan", "query": job["query"], "context": "", "idx": thread}
    elif kind == "plan":
        s.draft_plan(job["query"], job["context"], progress)
        st.session_state.pending_plan["idx"] = thread
    elif kind == "research":
        s.execute(job["query"], job["steps"], progress, job["context"], idx=thread)
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
        s.render_session(active, locked=waiting)  # the whole thread, oldest message first
    if clarify:
        clarified = s.render_clarify_card()
        if clarified:
            new_job = {"kind": "plan", "query": clarified[0], "context": clarified[1], "idx": clarify.get("idx")}
    elif plan:
        approved = s.render_plan_card()
        if approved:
            new_job = {"kind": "research", "query": approved[0], "steps": approved[1], "context": approved[2],
                       "idx": plan.get("idx")}
    elif active is None:
        on_welcome = True
        typed = s.render_empty_state(disabled=not sources)  # chat bar centered on the page

s.render_approval_notice()

if not on_welcome:
    typed = s.composer(disabled=not sources or waiting,
                       placeholder="Ask a follow-up..." if active is not None else s.COMPOSER_PLACEHOLDER)

prompt = typed or st.session_state.queued_prompt
st.session_state.queued_prompt = None
st.session_state.queued_mode = None

if prompt and sources and not new_job:
    st.session_state.pending_decision = None
    if active is not None and s.card(st.session_state.dossiers[active]) is not None:
        # A follow-up: answered from this thread's evidence, searching more only if needed.
        new_job = {"kind": "context", "idx": active, "question": prompt}
    else:
        # First question, or a retry after a failed answer: full research. With a thread open,
        # the answer is appended to it rather than replacing it.
        new_job = s.new_research_job(prompt, active)

if new_job:
    st.session_state.pending_plan = None
    st.session_state.pending_clarify = None
    st.session_state.job = new_job
    st.rerun()
