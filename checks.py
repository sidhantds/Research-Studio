"""Verification and quality checks. Pure functions: no Streamlit, no network, no model calls.

This is the code behind "the model never grades its own work unchecked":
- every quote a model gives is matched word for word against the text the tools returned,
- answers are cleaned before they're shown (unknown sources dropped, citations renumbered),
- each run gets automatic quality checks.
"""

import re
from urllib.parse import urlparse

from agent import Answer, ClaimCheck

# Sidebar source -> the tool that serves it
SOURCE_TOOLS = {"Web": "duckduckgo_search", "Wikipedia": "wikipedia", "Documents": "search_documents"}
LOOKUP_TOOLS = set(SOURCE_TOOLS.values())

STEP_NAMES = {
    "duckduckgo_search": "Web search",
    "wikipedia": "Wikipedia",
    "search_documents": "Your documents",
    "request_save_file": "Save request",
}


def step_name(tool: str) -> str:
    return STEP_NAMES.get(tool, tool.replace("_", " ").capitalize())


def domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.") or url


def source_key(source: str) -> str:
    """How two citations of the same source are recognised as one."""
    return source.strip().lower().rstrip("/")


# ---------- Word-for-word quote matching ----------

def norm_text(s: str) -> str:
    """Lowercase, drop quote marks and Markdown emphasis, collapse whitespace."""
    s = re.sub(r"[\"'“”‘’`*_]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def quote_in(quote: str, haystack: str) -> bool:
    """True if every part of a quote (split on ellipses) appears word for word in `haystack`.

    `haystack` must already be normalised with norm_text. Parts shorter than 12 characters are
    ignored, and a quote with no part that long never matches: "the" can't count as evidence.
    """
    parts = [p.strip(" .,;:") for p in re.split(r"\.\.\.|…", norm_text(quote))]
    parts = [p for p in parts if len(p) >= 12]
    return bool(parts) and all(p in haystack for p in parts)


def evidence_text(steps: list, cap: int = 24000) -> str:
    """The retrieved text claims are checked against, newest lookups first, up to `cap` chars."""
    out, used = [], 0
    for s in reversed(steps):
        block = f"(Retrieved with {step_name(s.tool)}: {s.tool_input[:80]})\n{s.output}"
        if used + len(block) > cap:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


def apply_quote_checks(checked: list[ClaimCheck], evidence: str, sources: list[str]) -> list[dict]:
    """Turn the checker model's verdicts into stored claims, trusting only quotes found in `evidence`.

    A "Supported" or "Partially supported" claim whose quote isn't in the retrieved text is
    downgraded to "Insufficient evidence". A conflicting quote that isn't there is dropped.
    """
    hay = norm_text(evidence)

    def src(i: int) -> str:
        return sources[i - 1] if 1 <= i <= len(sources) else ""

    claims = []
    for k in checked:
        found = bool(k.quote.strip()) and quote_in(k.quote, hay)
        verdict, reason, downgraded = k.verdict, k.reason.strip(), False
        if verdict in ("Supported", "Partially supported") and not found:
            verdict, downgraded = "Insufficient evidence", True
            reason = "The quoted passage isn't in the text the agent retrieved, so it can't count as support."
        conflict = k.conflict_quote.strip() if k.conflict_quote.strip() and quote_in(k.conflict_quote, hay) else ""
        claims.append({
            "claim": k.claim.strip(), "verdict": verdict, "reason": reason,
            "quote": k.quote.strip() if found else "", "source": src(k.source_id) if found else "",
            "conflict_quote": conflict, "conflict_source": src(k.conflict_source_id) if conflict else "",
            "downgraded": downgraded, "review": None,
        })
    return claims


def needs_review(c: dict) -> bool:
    """A claim waits for the user if it isn't fully supported or has conflicting evidence."""
    return c["review"] is None and (c["verdict"] != "Supported" or bool(c["conflict_quote"]))


# ---------- Cleaning an answer before it's shown ----------

_MARKER = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\](?!\()")


def remap_markers(text: str, mapping: dict[str, str]) -> str:
    """Rewrite [n] / [n, m] markers through `mapping`; markers for dropped sources disappear.
    Text inside ``` code fences is never touched."""
    def repl(m: re.Match) -> str:
        ids = [mapping[x.strip()] for x in m.group(1).split(",") if x.strip() in mapping]
        return "".join(f"[{i}]" for i in ids)
    parts = text.split("```")
    for k in range(0, len(parts), 2):
        parts[k] = _MARKER.sub(repl, parts[k])
    return "```".join(parts)


def finalize(c, allowed: set[str] | None, steps: list, context_mode: str, new_searches: int) -> Answer:
    """Make an answer safe to show and store.

    - `allowed` (for answers written without searching): drop any source the session never saw.
    - Number the sources 1..n and rewrite the [n] markers to match.
    - Keep a source's quote only if it appears word for word in the retrieved text.
    - Record how the answer was produced (context_mode, new_searches).
    """
    if type(c) is not Answer:  # a FollowupAnswer, or an Answer from before a module reload
        c = Answer.model_validate(c.model_dump(exclude={"evidence_sufficient", "missing"}))
    keep = [s for s in c.sources if allowed is None or source_key(s.url or s.title) in allowed]
    mapping, sources = {}, []
    hay = norm_text(evidence_text(steps)) if steps else ""
    for n, s in enumerate(keep, 1):
        mapping[s.id] = str(n)
        quote = s.quote if s.quote and hay and quote_in(s.quote, hay) else None
        sources.append(s.model_copy(update={"id": str(n), "quote": quote}))
    return c.model_copy(update={
        "sources": sources,
        "direct_answer": remap_markers(c.direct_answer, mapping),
        "details": [remap_markers(x, mapping) for x in c.details],
        "comparison": [r.model_copy(update={"value": remap_markers(r.value, mapping)}) for r in c.comparison]
        if c.comparison else None,
        "context_mode": context_mode,
        "new_searches": new_searches,
    })


# ---------- Automatic quality checks ----------

def is_grounded(source: str, outputs: str, used_tools: set[str]) -> bool:
    """Did this cited source actually appear in what the tools returned? `outputs` is lowercased."""
    src = source_key(source)
    if not src:
        return False
    if src.startswith("http"):
        dom = domain(src)
        if src in outputs or (dom and dom in outputs):
            return True
        return "wikipedia.org" in dom and "wikipedia" in used_tools
    if f"[document: {src}]" in outputs or src in outputs:
        return True
    return src.startswith("wikipedia") and "wikipedia" in used_tools


def quality_checks(response: Answer | None, steps: list, plan_steps: list[dict] | None) -> dict:
    """format: the reply parsed. grounded: (cited sources seen in the results, cited sources).
    plan: every source an approved plan asked for was used (None without a plan)."""
    used = {s.tool for s in steps}
    outputs = "\n".join(s.output for s in steps).lower()
    grounded = None
    if response and response.sources:
        hits = sum(is_grounded(s.url or s.title, outputs, used) for s in response.sources)
        grounded = (hits, len(response.sources))
    plan = None
    if plan_steps:
        needed = {SOURCE_TOOLS[x] for s in plan_steps for x in s["sources"]}
        plan = needed <= used
    return {"format": response is not None, "grounded": grounded, "plan": plan}
