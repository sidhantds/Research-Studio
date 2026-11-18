"""The verification layer: quote matching, claim downgrades, answer cleaning, quality checks."""

from agent import Answer, ClaimCheck, ComparisonRow, Source, ToolStep
from checks import (
    apply_quote_checks,
    evidence_text,
    finalize,
    is_grounded,
    needs_review,
    norm_text,
    quality_checks,
    quote_in,
    remap_markers,
)

# ---------- quote_in ----------

HAY = norm_text("Sodium-ion batteries use sodium ions as charge carriers. They are cheaper than lithium-ion cells.")


def test_exact_quote_matches():
    assert quote_in("use sodium ions as charge carriers", HAY)


def test_match_ignores_case_quotes_and_whitespace():
    assert quote_in('  “Sodium-ion   BATTERIES use sodium ions” ', HAY)


def test_invented_quote_does_not_match():
    assert not quote_in("sodium-ion batteries are more energy dense than lithium", HAY)


def test_ellipsis_quote_needs_every_part_present():
    assert quote_in("sodium-ion batteries use sodium ... cheaper than lithium-ion cells", HAY)
    assert not quote_in("sodium-ion batteries use sodium ... far more expensive than lithium", HAY)


def test_short_fragments_cannot_count_as_evidence():
    assert not quote_in("they are", HAY)
    assert not quote_in("", HAY)


# ---------- apply_quote_checks ----------

EVIDENCE = "Sodium-ion batteries use sodium ions as charge carriers. Their energy density is lower than lithium-ion."
SOURCES = ["https://example.org/a", "https://example.org/b"]


def _claim(**kw) -> ClaimCheck:
    base = {"claim": "Sodium-ion cells use sodium ions.", "source_id": 1,
            "quote": "use sodium ions as charge carriers", "verdict": "Supported", "reason": "Stated directly."}
    return ClaimCheck(**(base | kw))


def test_supported_claim_with_real_quote_is_kept():
    [c] = apply_quote_checks([_claim()], EVIDENCE, SOURCES)
    assert c["verdict"] == "Supported"
    assert c["source"] == SOURCES[0]
    assert c["downgraded"] is False
    assert not needs_review(c)


def test_supported_claim_with_invented_quote_is_downgraded():
    [c] = apply_quote_checks([_claim(quote="sodium-ion batteries last fifty years")], EVIDENCE, SOURCES)
    assert c["verdict"] == "Insufficient evidence"
    assert c["downgraded"] is True
    assert c["quote"] == "" and c["source"] == ""
    assert needs_review(c)


def test_conflict_quote_is_kept_only_if_real():
    real = _claim(conflict_quote="energy density is lower than lithium-ion", conflict_source_id=2)
    fake = _claim(conflict_quote="sodium-ion is banned in the EU", conflict_source_id=2)
    [c_real, c_fake] = apply_quote_checks([real, fake], EVIDENCE, SOURCES)
    assert c_real["conflict_source"] == SOURCES[1]
    assert needs_review(c_real)  # supported, but conflicting evidence still goes to review
    assert c_fake["conflict_quote"] == ""


def test_out_of_range_source_id_gives_no_source():
    [c] = apply_quote_checks([_claim(source_id=9)], EVIDENCE, SOURCES)
    assert c["source"] == ""


def test_reviewed_claim_leaves_the_queue():
    [c] = apply_quote_checks([_claim(verdict="Unsupported")], EVIDENCE, SOURCES)
    assert needs_review(c)
    c["review"] = "accepted"
    assert not needs_review(c)


# ---------- remap_markers ----------

def test_markers_are_renumbered_and_dropped_sources_removed():
    assert remap_markers("A [2]. B [3]. C [2, 3].", {"2": "1", "3": "2"}) == "A [1]. B [2]. C [1][2]."
    assert remap_markers("Gone [5].", {"1": "1"}) == "Gone ."


def test_markers_inside_code_and_links_are_untouched():
    text = "See [1].\n```python\nx = a[1]\n```\n[docs](https://x.y)"
    out = remap_markers(text, {"1": "2"})
    assert "See [2]." in out
    assert "x = a[1]" in out
    assert "[docs](https://x.y)" in out


# ---------- finalize ----------

def _answer(**kw) -> Answer:
    base = {
        "direct_answer": "Cheap [2]. Abundant [1].",
        "sources": [
            Source(id="1", title="A", url="https://www.example.org/sodium-ion",
                   quote="cheaper than lithium-ion cells because sodium is abundant"),
            Source(id="2", title="B", url="https://unknown.example.net/x", quote="made-up passage that is long"),
        ],
    }
    return Answer(**(base | kw))


def test_finalize_keeps_only_real_quotes(web_step):
    out = finalize(_answer(), None, [web_step], "new_research", 1)
    by_url = {s.url: s for s in out.sources}
    assert by_url["https://www.example.org/sodium-ion"].quote is not None
    assert by_url["https://unknown.example.net/x"].quote is None
    assert out.context_mode == "new_research" and out.new_searches == 1


def test_finalize_drops_sources_the_session_never_saw(web_step):
    allowed = {"https://www.example.org/sodium-ion"}
    out = finalize(_answer(), allowed, [web_step], "answer_from_session", 0)
    assert [s.url for s in out.sources] == ["https://www.example.org/sodium-ion"]
    assert [s.id for s in out.sources] == ["1"]
    assert out.direct_answer == "Cheap . Abundant [1]."  # marker for the dropped source is gone


def test_finalize_renumbers_comparison_cells(web_step):
    rows = [ComparisonRow(option="Na-ion", aspect="Cost", value="Low [2]")]
    out = finalize(_answer(comparison=rows), None, [web_step], "new_research", 1)
    assert out.comparison[0].value == "Low [2]"
    out = finalize(_answer(comparison=rows), {"https://unknown.example.net/x"}, [web_step], "new_research", 1)
    assert out.comparison[0].value == "Low [1]"


# ---------- evidence_text ----------

def test_evidence_text_is_newest_first_and_capped():
    steps = [ToolStep("wikipedia", "old", "old text"), ToolStep("wikipedia", "new", "new text")]
    text = evidence_text(steps)
    assert text.index("new text") < text.index("old text")
    assert "old text" not in evidence_text(steps, cap=60)


# ---------- quality checks ----------

def test_grounded_counts_only_sources_in_the_results(web_step):
    outputs = web_step.output.lower()
    assert is_grounded("https://www.example.org/sodium-ion", outputs, {"duckduckgo_search"})
    assert not is_grounded("https://made-up.example.net/page", outputs, {"duckduckgo_search"})


def test_wikipedia_source_is_grounded_when_the_wiki_tool_ran():
    assert is_grounded("https://en.wikipedia.org/wiki/Battery", "", {"wikipedia"})
    assert not is_grounded("https://en.wikipedia.org/wiki/Battery", "", {"duckduckgo_search"})


def test_quality_checks(web_step):
    checks = quality_checks(_answer(), [web_step], [{"question": "q", "sources": ["Web", "Wikipedia"]}])
    assert checks["format"] is True
    assert checks["grounded"] == (1, 2)
    assert checks["plan"] is False  # the plan asked for Wikipedia, which never ran
    assert quality_checks(None, [], None) == {"format": False, "grounded": None, "plan": None}
