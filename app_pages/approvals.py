import streamlit as st

import studio as s

st.title("Approvals")
st.caption("The agent asks before it saves anything.")

pending = st.session_state.pending_saves
if not pending:
    with st.container(border=True):
        st.markdown(":green[:material/check_circle:] **You're all caught up.**")
        st.caption("Nothing is waiting for approval.")
    if st.button("Back to research"):
        st.switch_page(s.RESEARCH_PAGE)

for req in pending:
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Save this report**")
            st.space("stretch")
            st.badge("Needs approval", color=s.STATUS_COLORS["Needs approval"])
        if req.get("question"):
            st.caption(f"Requested while researching: {req['question']}")
        with st.expander("Preview"):
            st.text(req["data"][:3000])
        st.text_input("File name", value=req["filename"], key=f"save_name_{req['id']}",
                      help="Saved to your library for this session. Download it from there to keep it.")
        with st.container(horizontal=True, gap="small"):
            st.button("Approve", type="primary", on_click=s.approve_save, args=(req["id"],),
                      key=f"ok_{req['id']}")
            st.button("Skip", type="tertiary", on_click=s.skip_save, args=(req["id"],),
                      key=f"skip_{req['id']}")
