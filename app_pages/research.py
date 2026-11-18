import uuid

import streamlit as st

import studio as s

sources = s.enabled_sources()
if not sources:
    st.warning("Turn on at least one source in the sidebar to start researching.")

# One primary view at a time: clarifying questions, plan, report, or the welcome screen.
body = st.empty()
run_request = None  # (query, steps, context)
clarified = None  # (query, context)
typed = None
on_welcome = False
with body.container():
    if st.session_state.pending_clarify:
        clarified = s.render_clarify_card()
    elif st.session_state.pending_plan:
        run_request = s.render_plan_card()
    elif st.session_state.active is not None:
        s.render_report(st.session_state.active)
    else:
        on_welcome = True
        typed = s.render_empty_state(disabled=not sources)  # chat bar centered on the page

s.render_approval_notice()

# With a plan or report on screen, the chat bar is pinned to the bottom instead.
if not on_welcome:
    typed = s.composer(disabled=not sources)

# Slow work goes last, so the page above renders instantly.
prompt = typed or st.session_state.queued_prompt
st.session_state.queued_prompt = None

if prompt and sources:
    st.session_state.pending_plan = None
    st.session_state.pending_clarify = None
    mode = s.mode()
    if mode == "Ask first":
        body.empty()
        questions = s.draft_clarify(prompt, body)
        if questions:
            st.session_state.pending_clarify = {"id": uuid.uuid4().hex[:8], "query": prompt, "questions": questions}
            st.rerun()
        clarified = (prompt, "")  # nothing to ask: go straight to the plan
    elif mode == "Review plan":
        clarified = (prompt, "")
    else:
        run_request = (prompt, None, "")

if clarified:
    st.session_state.pending_clarify = None
    body.empty()
    s.draft_plan(*clarified, body)
    st.rerun()

if run_request:
    st.session_state.pending_plan = None
    body.empty()
    query, steps, context = run_request
    s.execute(query, steps, body, context)
    st.rerun()
