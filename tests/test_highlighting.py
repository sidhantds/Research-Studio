"""Claim highlighting in the answer text (display logic, no model calls)."""

from studio import _confidence_bar, _highlight_claims


def _cl(claim, verdict="Supported", review=None, conflict=""):
    return {"claim": claim, "verdict": verdict, "review": review, "conflict_quote": conflict}


ANSWER = ("Sodium-ion batteries cost less than lithium-ion cells because sodium is abundant [1]. "
          "Their energy density is lower [2]. The weather was nice.")


def test_sentences_are_coloured_by_their_claim_verdict():
    claims = [_cl("Sodium-ion batteries cost less than lithium-ion because sodium is abundant."),
              _cl("Sodium-ion energy density is lower than lithium-ion.", "Unsupported")]
    out, n = _highlight_claims(ANSWER, claims)
    assert n == 2
    assert ":green-background[Sodium-ion batteries cost less" in out
    assert ":red-background[Their energy density is lower.]" in out
    assert "The weather was nice." in out and "background[The weather" not in out


def test_citation_markers_move_outside_the_highlight():
    out, _ = _highlight_claims(ANSWER, [_cl("Sodium-ion batteries cost less than lithium-ion because sodium is abundant.")])
    assert "abundant.] [1]" in out


def test_review_decision_overrides_the_verdict_colour():
    out, _ = _highlight_claims(ANSWER, [_cl("Their energy density is lower.", "Unsupported", review="accepted")])
    assert ":green-background[Their energy density is lower.]" in out


def test_code_and_unmatched_text_are_left_alone():
    code = "Run this:\n```python\nprint('hi')\n```"
    assert _highlight_claims(code, [_cl("print hi")]) == (code, 0)
    assert _highlight_claims("Nothing related here.", [_cl("Quantum computers factor large numbers.")])[1] == 0
    assert _highlight_claims(ANSWER, []) == (ANSWER, 0)


def test_confidence_bar_has_one_segment_per_colour():
    bar = _confidence_bar([_cl("a"), _cl("b"), _cl("c", "Unsupported")])
    assert bar.count("flex:") == 2  # one segment for green (2 claims), one for red (1)
    assert "flex:2;" in bar and "flex:1;" in bar
