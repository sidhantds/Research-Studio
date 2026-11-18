import streamlit as st

import studio as s

st.title("Runs & quality")
st.caption("Every model run this session, new research and follow-ups, with automatic quality checks.")

runs = st.session_state.runs
if not runs:
    with st.container(border=True):
        st.markdown("**No runs yet.**")
        st.caption("Run some research and its timing, token use and quality checks will show up here.")
    st.stop()


def pct(values: list[bool]) -> str:
    return f"{sum(values) / len(values):.0%}" if values else "–"


tokens_in = sum(r["tokens_in"] for r in runs)
tokens_out = sum(r["tokens_out"] for r in runs)
cost = s.estimate_cost(tokens_in, tokens_out)
followups = [r for r in runs if r["kind"] != "New research"]

with st.container(horizontal=True):
    st.metric("Runs", len(runs), border=True)
    st.metric("Succeeded", pct([r["status"] != "Failed" for r in runs]), border=True)
    st.metric("Avg time", f"{sum(r['seconds'] for r in runs) / len(runs):.0f}s", border=True)
    st.metric("Est. cost", f"${cost:.4f}" if cost is not None else "–", border=True,
              help="Set prices in Settings to estimate cost.")

st.subheader("Quality checks")
grounded = [r["checks"]["grounded"] for r in runs if r["checks"].get("grounded")]
plan_runs = [r["checks"]["plan"] for r in runs if r["checks"].get("plan") is not None]
with st.container(horizontal=True):
    st.metric("Valid answer format", pct([r["checks"]["format"] for r in runs if "format" in r["checks"]]),
              border=True, help="Share of runs whose answer parsed into a full answer card.")
    st.metric("Sources grounded",
              f"{sum(h for h, _ in grounded) / max(sum(t for _, t in grounded), 1):.0%}" if grounded else "–",
              border=True, help="Cited sources that appear in the search results the agent actually read.")
    st.metric("Plan followed", pct(plan_runs), border=True,
              help="Runs with an approved plan where the agent used every source the plan called for.")
    st.metric("Follow-ups with no new search", pct([r["searches"] == 0 for r in followups]), border=True,
              help="Follow-ups answered from evidence the session already had.")
claims = [r["checks"]["claims"] for r in runs if r["checks"].get("claims")]
with st.container(horizontal=True):
    st.metric("Claims supported",
              f"{sum(a for a, _ in claims) / max(sum(b for _, b in claims), 1):.0%}" if claims else "–",
              border=True, help="Claims whose quoted passage was found in the retrieved text and fully backs them.")
    st.metric("Claims checked", sum(b for _, b in claims) if claims else "–", border=True)
    st.metric("Prompt version", runs[-1].get("prompt_version", "–"), border=True)

st.subheader("Recent runs")


def cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


rows = ["| Time | Type | Question | Status | Time taken | Searches | Tokens | Grounded | Claims |",
        "|---|---|---|---|---|---|---|---|---|"]
for r in reversed(runs):
    g = r["checks"].get("grounded")
    cl = r["checks"].get("claims")
    q = r["query"] if len(r["query"]) <= 40 else r["query"][:40] + "…"
    rows.append("| " + " | ".join(cell(x) for x in [
        r["time"], r["kind"], q, r["status"], f"{r['seconds']:.0f}s", r["searches"],
        f"{r['tokens_in'] + r['tokens_out']:,}", f"{g[0]}/{g[1]}" if g else "–", f"{cl[0]}/{cl[1]}" if cl else "–",
    ]) + " |")
st.markdown("\n".join(rows))

failures = [r for r in runs if r["status"] == "Failed"]
if failures:
    st.subheader("Failures")
    for r in reversed(failures):
        with st.expander(f"{r['time']} · {r['kind']} · {r['query'][:60]}"):
            st.text(r["error"] or "Unknown error")

with st.expander("How the checks work"):
    st.markdown(
        "- **Valid answer format**: the answer parsed into the answer card structure.\n"
        "- **Sources grounded**: each cited source is matched against the text the tools returned in "
        "this research session. A source that never appeared there may be made up.\n"
        "- **Plan followed**: with an approved plan, the agent called every source the plan listed.\n"
        "- **Follow-ups with no new search**: follow-ups answered from evidence the session already had. "
        "Higher means less repeated searching and lower cost.\n"
        "- **Claims supported**: a separate check splits each answer into claims and quotes the passage "
        "behind each one. The quote must appear word for word in the text the tools returned, or the claim "
        "is downgraded.\n\n"
        "These run automatically on every run. They catch common problems but don't judge whether "
        "the answers are correct."
    )
