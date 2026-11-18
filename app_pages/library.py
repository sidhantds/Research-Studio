import streamlit as st

import studio as s

st.title("Library")

reports_tab, files_tab = st.tabs(["Reports", "Saved files"])

with reports_tab:
    dossiers = st.session_state.dossiers
    if not dossiers:
        st.caption("Reports you create this session will appear here.")
    for i in reversed(range(len(dossiers))):
        d = dossiers[i]
        status = s.dossier_status(d)
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"**{d['title']}**")
                st.space("stretch")
                st.badge(status, color=s.STATUS_COLORS[status])
            r = d.get("result")
            preview = (r.response.answer or r.response.summary[:160]) if r and r.response else d.get("error") or ""
            if preview:
                st.caption(preview)
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(f"{d['created']} · {s.source_count(d)} sources")
                st.space("stretch")
                if st.button("Open", key=f"lib_open_{d['id']}", type="tertiary"):
                    s.open_report(i)
                    st.switch_page(s.RESEARCH_PAGE)

with files_tab:
    s.render_saved_files()
