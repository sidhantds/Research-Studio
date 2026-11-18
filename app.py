"""Research Studio. Run with:  streamlit run app.py

Ask -> review plan -> watch research -> read & act on report.
"""

import streamlit as st

import studio as s
from agent import MODEL_NAME

st.set_page_config(
    page_title="Research Studio",
    page_icon=":material/travel_explore:",
    layout="centered",
)

s.init_state()

if not s.has_api_key():
    st.error("No `GOOGLE_API_KEY` found. Add it to your `.env` file and restart the app.")
    st.stop()

n_approvals = len(st.session_state.pending_saves)
research_page = st.Page(s.RESEARCH_PAGE, title="Research", icon=":material/travel_explore:",
                        url_path="research", default=True)
page = st.navigation(
    [
        research_page,
        st.Page("app_pages/library.py", title="Library", icon=":material/folder_open:", url_path="library"),
        st.Page(s.APPROVALS_PAGE, title=f"Approvals ({n_approvals})" if n_approvals else "Approvals",
                icon=":material/verified_user:", url_path="approvals"),
    ],
    position="top",
)

# Narrow rail: recent research, plus advanced settings tucked away at the bottom.
with st.sidebar:
    st.markdown("**Recent research**")
    dossiers = st.session_state.dossiers
    if not dossiers:
        st.caption("Your reports will be listed here.")
    for i in reversed(range(len(dossiers))[-10:]):
        d = dossiers[i]
        is_active = i == st.session_state.active and page.url_path == "research"
        title = d["title"] if len(d["title"]) <= 42 else d["title"][:42] + "…"
        if st.button(title, key=f"rail_{d['id']}", width="stretch",
                     type="secondary" if is_active else "tertiary"):
            s.open_report(i)
            st.switch_page(research_page)
        st.caption(f"{s.dossier_status(d)} · {d['created']} · {s.source_count(d)} sources")

    st.space("stretch")
    with st.expander("Advanced"):
        st.caption(f"Model: `{MODEL_NAME}`")
        st.caption("Tools: web search (DuckDuckGo, Bing, Mojeek, Yahoo), Wikipedia, "
                   "and file saving, which always needs your approval.")
        st.button("Clear history", type="tertiary", on_click=s.clear_history,
                  disabled=not dossiers, key="clear_history")

page.run()
