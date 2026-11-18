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
                    st.badge(s.TYPE_LABELS.get(c.response_type, "Answer"), color="gray")
                st.badge(status, color=s.STATUS_COLORS[status])
            preview = c.direct_answer if c else (d["turns"][0]["error"] if d["turns"] else "") or ""
            if preview:
                st.caption(preview[:200] + ("…" if len(preview) > 200 else ""))
            with st.container(horizontal=True, vertical_alignment="center"):
                n_q, n_s = len(d["turns"]), s.source_count(d)
                n_c, n_r = len(s.all_claims(d)), len(s.review_queue(d))
                st.caption(f"{n_q} question{'s' if n_q != 1 else ''} · Updated {d['updated']} · "
                           f"{n_s} source{'s' if n_s != 1 else ''} · {n_c} claim{'s' if n_c != 1 else ''}"
                           + (f" · {n_r} to review" if n_r else ""))
                st.space("stretch")
                if st.button("Open", key=f"lib_open_{d['id']}", type="tertiary"):
                    s.open_report(i)
                    st.switch_page(s.RESEARCH_PAGE)

with files_tab:
    s.render_saved_files()
