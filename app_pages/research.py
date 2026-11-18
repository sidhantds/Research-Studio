import uuid

import streamlit as st

import studio as s

# Slow work (drafting a plan, researching, follow-ups) runs as a "job" in its own script run:
#   1. A click or a submitted question only records the job and reruns.
#   2. On the next run, `body` is redrawn without any live controls, so nothing stale
#      can be clicked (a click would abort the run), and progress renders in `progress`.
progress = st.container()
body = st.empty()

job = st.session_state.pop("job", None)  # popped first: an interrupted job never restarts itself

if job:
    kind = job["kind"]
    if kind in ("context", "further"):
        # Follow-ups keep the card on screen, locked, while it's being updated.
        with body.container():
            s.render_report(job["idx"], locked=True)
        s.render_approval_notice(disabled=True)
        s.render_followup_bar(disabled=True)
        s.composer(disabled=True, placeholder="Updating this research...")
        if kind == "context":
            s.followup_from_evidence(job["idx"], job["question"], progress, force=job.get("force", False))
        else:
            s.followup_research_further(job["idx"], job["question"], progress)
    else:
        s.composer(disabled=True)
        if kind == "clarify":
            questions = s.draft_clarify(job["query"], progress)
            if questions:
                st.session_state.pending_clarify = {"id": uuid.uuid4().hex[:8], "query": job["query"],
                                                    "questions": questions}
            else:  # nothing to ask: go straight to the plan, as its own job
                st.session_state.job = {"kind": "plan", "query": job["query"], "context": ""}
        elif kind == "plan":
            s.draft_plan(job["query"], job["context"], progress)
        elif kind == "research":
            s.execute(job["query"], job["steps"], progress, job["context"])
    st.rerun()

sources = s.enabled_sources()
if not sources:
    st.warning("Turn on at least one source in the sidebar to start researching.")

active = st.session_state.active
has_card = active is not None and s.card(st.session_state.dossiers[active]) is not None

# One primary view at a time: clarifying questions, plan, answer card, or the welcome screen.
new_job = None
typed = None
on_welcome = False
with body.container():
    if st.session_state.pending_clarify:
        clarified = s.render_clarify_card()
        if clarified:
            new_job = {"kind": "plan", "query": clarified[0], "context": clarified[1]}
    elif st.session_state.pending_plan:
        approved = s.render_plan_card()
        if approved:
            new_job = {"kind": "research", "query": approved[0], "steps": approved[1], "context": approved[2]}
    elif active is not None:
        s.render_report(active)
        if st.session_state.pending_decision and st.session_state.pending_decision["idx"] == active:
            s.render_decision_card()
    else:
        on_welcome = True
        typed = s.render_empty_state(disabled=not sources)  # chat bar centered on the page

s.render_approval_notice()

# With a card on screen, follow-ups stay in this research session by default.
showing_card = has_card and not st.session_state.pending_plan and not st.session_state.pending_clarify
if showing_card:
    s.render_followup_bar()
if not on_welcome:
    placeholder = "Ask a follow-up..." if showing_card and s.followup_mode() != "New research" else s.COMPOSER_PLACEHOLDER
    typed = s.composer(disabled=not sources, placeholder=placeholder)

prompt = typed or st.session_state.queued_prompt
forced_mode = st.session_state.queued_mode
st.session_state.queued_prompt = None
st.session_state.queued_mode = None

if prompt and sources and not new_job:
    st.session_state.pending_decision = None
    route = (forced_mode or s.followup_mode()) if showing_card else "New research"
    if route == "Answer from this research":
        new_job = {"kind": "context", "idx": active, "question": prompt}
    elif route == "Research further":
        new_job = {"kind": "further", "idx": active, "question": prompt}
    else:  # New research is always explicit: from the welcome screen, the selector, or a button
        st.session_state.active = None
        new_job = s.new_research_job(prompt)

if new_job:
    st.session_state.pending_plan = None
    st.session_state.pending_clarify = None
    st.session_state.job = new_job
    st.rerun()
