import streamlit as st

import studio as s

st.title("Runs & quality")
st.caption("Every research run this session, with automatic quality checks.")

runs = st.session_state.dossiers
if not runs:
    with st.container(border=True):
        st.markdown("**No runs yet.**")
        st.caption("Run some research and its timing, token use and quality checks will show up here.")
    st.stop()


def pct(values: list[bool]) -> str:
    return f"{sum(values) / len(values):.0%}" if values else "–"


ok = [d for d in runs if not d.get("error")]
grounded = [d["checks"]["grounded"] for d in runs if d.get("checks", {}).get("grounded")]
tokens_in = sum(d.get("tokens_in", 0) for d in runs)
tokens_out = sum(d.get("tokens_out", 0) for d in runs)
cost = s.estimate_cost(tokens_in, tokens_out)

with st.container(horizontal=True):
    st.metric("Runs", len(runs), border=True)
    st.metric("Succeeded", pct([not d.get("error") for d in runs]), border=True)
    st.metric("Avg time", f"{sum(d['seconds'] for d in runs) / len(runs):.0f}s", border=True)
    st.metric("Est. cost", f"${cost:.4f}" if cost is not None else "–", border=True,
              help="Set prices in Settings to estimate cost.")

st.subheader("Quality checks")
with st.container(horizontal=True):
    st.metric("Valid report format", pct([d["checks"]["format"] for d in runs if "checks" in d]), border=True,
              help="Share of runs whose answer parsed into a full report.")
    st.metric("Sources grounded",
              f"{sum(h for h, _ in grounded) / max(sum(t for _, t in grounded), 1):.0%}" if grounded else "–",
              border=True, help="Cited sources that appear in the search results the agent actually read.")
    plan_runs = [d["checks"]["plan"] for d in runs if d.get("checks", {}).get("plan") is not None]
    st.metric("Plan followed", pct(plan_runs), border=True,
              help="Runs with an approved plan where the agent used every source the plan called for.")

st.subheader("Recent runs")


def cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


rows = ["| Time | Question | Mode | Depth | Status | Time taken | Steps | Tokens | Grounded |",
        "|---|---|---|---|---|---|---|---|---|"]
for d in reversed(runs):
    g = d.get("checks", {}).get("grounded")
    n_steps = len(d["result"].steps) if d.get("result") else 0
    q = d["query"] if len(d["query"]) <= 40 else d["query"][:40] + "…"
    rows.append("| " + " | ".join(cell(x) for x in [
        d["created"], q, d.get("mode", "–"), d.get("depth", "–"), s.dossier_status(d), f"{d['seconds']:.0f}s",
        n_steps, f"{d.get('tokens_in', 0) + d.get('tokens_out', 0):,}", f"{g[0]}/{g[1]}" if g else "–",
    ]) + " |")
st.markdown("\n".join(rows))

failures = [d for d in runs if d.get("error")]
if failures:
    st.subheader("Failures")
    for d in reversed(failures):
        with st.expander(f"{d['created']} · {d['query'][:60]}"):
            st.text(d["error"])

with st.expander("How the checks work"):
    st.markdown(
        "- **Valid report format**: the answer parsed into the report structure "
        "(title, answer, findings, report, sources).\n"
        "- **Sources grounded**: each cited source is matched against the text the tools returned. "
        "A source that never appeared there may be made up.\n"
        "- **Plan followed**: with an approved plan, the agent called every source the plan listed.\n\n"
        "These run automatically on every report. They catch common problems but don't judge whether "
        "the findings are correct."
    )
