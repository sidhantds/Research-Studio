import streamlit as st

import studio as s

st.title("Library")

reports_tab, files_tab = st.tabs(["Reports", "Saved files"])

with reports_tab:
    dossiers = st.session_state.dossiers
    if not dossiers:
        st.caption("Research sessions you start will appear here.")
    for i in reversed(range(len(dossiers))):
        d = dossiers[i]
        status = s.dossier_status(d)
        c = s.card(d)
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"**{d['title']}**")
                st.space("stretch")
                if c:
                    st.badge(c.answer_type, color="gray")
                st.badge(status, color=s.STATUS_COLORS[status])
            preview = c.answer if c else d.get("error") or ""
            if preview:
                st.caption(preview[:200] + ("…" if len(preview) > 200 else ""))
            with st.container(horizontal=True, vertical_alignment="center"):
                n_versions = len(d["versions"])
                st.caption(f"Version {n_versions} · Updated {d['updated']} · {s.source_count(d)} sources")
                st.space("stretch")
                if st.button("Open", key=f"lib_open_{d['id']}", type="tertiary"):
                    s.open_report(i)
                    st.switch_page(s.RESEARCH_PAGE)

with files_tab:
    s.render_saved_files()
