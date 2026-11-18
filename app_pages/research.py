import uuid

import streamlit as st

import studio as s
from agent import make_plan

# Top bar
with st.container(horizontal=True, vertical_alignment="center", gap="small"):
    st.markdown("**Research Studio**")
    st.space("stretch")
    st.segmented_control("Mode", s.MODES, default=s.MODES[0], required=True, key="mode",
                         label_visibility="collapsed", persist_state="session",
                         help="**Review plan**: approve the plan before research starts.  \n"
                              "**Auto-research**: the agent plans and runs on its own.")
    st.button("New research", icon=":material/add:", on_click=s.new_research,
              disabled=st.session_state.active is None and not st.session_state.pending_plan)

# One primary view at a time: plan, report, or empty state.
body = st.empty()
run_request = None
with body.container():
    if st.session_state.pending_plan:
        run_request = s.render_plan_card()
    elif st.session_state.active is not None:
        s.render_report(st.session_state.active)
    else:
        s.render_empty_state()

s.render_approval_notice()

typed = st.chat_input("Ask a research question...", key="composer", submit_mode="disable")

# Slow work goes last, so the page above renders instantly.
prompt = typed or st.session_state.queued_prompt
st.session_state.queued_prompt = None

if prompt:
    st.session_state.pending_plan = None
    if s.mode() == "Review plan":
        body.empty()
        with body.container():
            st.badge("Drafting plan", color="blue")
            st.title(prompt)
            with st.status(":shimmer[Drafting a research plan...]", type="compact"):
                steps = make_plan(s.cached_llm(), prompt)
        for step in steps:
            step["sid"] = uuid.uuid4().hex[:8]
        st.session_state.pending_plan = {"id": uuid.uuid4().hex[:8], "query": prompt, "steps": steps}
        st.rerun()
    run_request = (prompt, None)

if run_request:
    st.session_state.pending_plan = None
    body.empty()
    s.execute(*run_request, body)
    st.rerun()
