"""Research Studio. Run with:  streamlit run app.py

Ask -> review plan -> watch research -> read & act on report.
"""

import importlib
import os

import streamlit as st

import agent
import studio
import tools


def _reload_changed_modules():
    """Reload helper modules whose files changed since they were imported.

    Streamlit reruns this script on every interaction but keeps imported modules in memory.
    After a code update (a git push to Streamlit Cloud, or an edit while running locally),
    the new app.py could otherwise run against the old studio.py and crash. Reloads go in
    dependency order (tools -> agent -> studio) so each picks up the fresh version below it.
    """
    changed = False
    for mod in (tools, agent, studio):
        path = str(mod.__file__)
        mtime = os.path.getmtime(path)
        # No timestamp yet means the module was imported by code from before this guard
        # existed (or the process just started): reload once to be sure it's current.
        if changed or getattr(mod, "_loaded_mtime", None) != mtime:
            importlib.reload(mod)
            changed = True  # modules that import this one must reload too
        setattr(mod, "_loaded_mtime", os.path.getmtime(path))


_reload_changed_modules()
s = studio
MODEL_NAME = agent.MODEL_NAME

st.set_page_config(
    page_title="Research Studio",
    page_icon=":material/travel_explore:",
    layout="centered",
)

s.init_state()

if not s.has_api_key():
    st.error("No `GOOGLE_API_KEY` found. Add it to your `.env` file (or Streamlit secrets) and restart the app.")
    st.stop()

research = st.Page(s.RESEARCH_PAGE, title="Research", icon=":material/travel_explore:",
                   url_path="research", default=True)
library = st.Page("app_pages/library.py", title="Library", icon=":material/folder_open:", url_path="library")
approvals = st.Page(s.APPROVALS_PAGE, title="Approvals", icon=":material/verified_user:", url_path="approvals")
runs = st.Page("app_pages/runs.py", title="Runs & quality", icon=":material/monitoring:", url_path="runs")

# Built-in menu hidden so the sidebar can follow our own order (title and New research first).
page = st.navigation([research, library, approvals, runs], position="hidden")

# Sidebar: only persistent navigation and controls that affect every research run.
with st.sidebar:
    st.markdown("### Research Studio")
    # While a plan is drafted or research runs, lock the controls: any widget click
    # would make Streamlit abort the run in progress.
    busy = bool(st.session_state.get("job"))
    if busy:
        st.caption(":material/hourglass_top: Working... controls unlock when it's done.")
    if st.button("New research", icon=":material/add:", width="stretch", disabled=busy):
        s.new_research()
        st.switch_page(research)

    n_approvals = len(st.session_state.pending_saves)
    st.page_link(research, width="stretch", disabled=busy)
    st.page_link(library, width="stretch", disabled=busy)
    st.page_link(approvals, label=f"Approvals · **{n_approvals}**" if n_approvals else "Approvals",
                 width="stretch", disabled=busy)
    st.page_link(runs, width="stretch", disabled=busy)

    st.divider()

    st.radio("Research mode", s.MODES, index=s.MODES.index("Review plan"), key="mode",
             captions=s.MODE_CAPTIONS, disabled=busy)

    st.markdown("**Sources**")
    st.checkbox("Web search", value=True, key="src_web", disabled=busy)
    st.checkbox("Wikipedia", value=True, key="src_wiki", disabled=busy)
    use_docs = st.checkbox("Uploaded documents", value=False, key="src_docs", disabled=busy)
    if use_docs:
        uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True,
                                   key="uploads", label_visibility="collapsed", disabled=busy,
                                   help="PDF, TXT or Markdown. Stays in your session only.")
        s.ingest_uploads(uploads)
        if not st.session_state.docs:
            st.caption("Add a document to use this source.")

    st.radio("Depth", list(s.DEPTHS), index=1, key="depth", captions=s.DEPTH_CAPTIONS, disabled=busy)

    st.divider()

    usage = st.session_state.usage
    n_runs = len(st.session_state.runs)
    cost = s.estimate_cost(usage["input"], usage["output"])
    st.markdown("**Usage**")
    st.caption(f"This session: {n_runs} run{'s' if n_runs != 1 else ''}  \n"
               f"Tokens: {usage['input'] + usage['output']:,}  \n"
               + (f"Est. cost: ${cost:.4f}" if cost is not None else "Est. cost: add prices in Settings"))

    with st.popover("Settings", icon=":material/settings:", width="stretch", disabled=busy):
        st.markdown("**Model**")
        st.caption(f"`{MODEL_NAME}`")
        st.markdown("**API connection**")
        st.caption(":green[●] Connected" if s.has_api_key() else ":red[●] No API key")
        st.markdown("**Cost estimate**")
        st.caption("Prices per million tokens, from your provider's pricing page.")
        st.number_input("Input price (USD per 1M tokens)", min_value=0.0, step=0.01, format="%.3f",
                        value=float(os.getenv("PRICE_INPUT_PER_M", 0) or 0), key="price_in")
        st.number_input("Output price (USD per 1M tokens)", min_value=0.0, step=0.01, format="%.3f",
                        value=float(os.getenv("PRICE_OUTPUT_PER_M", 0) or 0), key="price_out")
        st.markdown("**Privacy**")
        st.caption("Reports, uploads and saved files stay in your browser session and aren't stored "
                   "on the server. Questions are sent to Gemini; search queries go to the search engines.")
        st.markdown("**Theme**")
        st.caption("Light")
        st.button("Clear history", type="tertiary", on_click=s.clear_history,
                  disabled=not st.session_state.dossiers, key="clear_history")

page.run()
