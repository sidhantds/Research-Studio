"""Cases survive a refresh: save to JSON, load back, nothing lost; nothing outside the folder."""

import pytest

import persistence
from agent import Answer, Source, ToolStep

SID = "0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def tmp_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "SESSIONS_DIR", tmp_path)
    return tmp_path


def _state():
    card = Answer(response_type="research", title="T", direct_answer="A [1].",
                  sources=[Source(id="1", title="S", url="https://example.org/a", quote="a real quote here")])
    turn = {"question": "Q?", "card": card, "fresh": True, "claims": [{"claim": "c", "verdict": "Supported"}],
            "checks": {"grounded": (1, 1)}, "lookups": [("Web search", "q", False)]}
    failed = {"question": "Q2?", "card": None, "error": "timeout", "fresh": False, "claims": [], "checks": {}}
    d = {"id": "abc", "title": "T", "turns": [turn, failed],
         "evidence": [ToolStep("duckduckgo_search", "q", "out")], "sources": []}
    return {"dossiers": [d], "runs": [{"kind": "New research"}], "pending_saves": [], "saved_files": [],
            "usage": {"input": 5, "output": 7}, "active": 0, "not_persisted": "x"}


def test_round_trip_keeps_cases_answers_and_evidence():
    persistence.save(SID, _state())
    back = persistence.load(SID)
    d = back["dossiers"][0]
    assert isinstance(d["turns"][0]["card"], Answer)
    assert d["turns"][0]["card"].sources[0].quote == "a real quote here"
    assert d["turns"][1]["card"] is None and d["turns"][1]["error"] == "timeout"
    assert isinstance(d["evidence"][0], ToolStep) and d["evidence"][0].output == "out"
    assert back["usage"] == {"input": 5, "output": 7} and back["active"] == 0
    assert "not_persisted" not in back


def test_restored_answers_are_not_animated_again():
    persistence.save(SID, _state())
    assert persistence.load(SID)["dossiers"][0]["turns"][0]["fresh"] is False


def test_tuples_come_back_usable():
    persistence.save(SID, _state())
    t = persistence.load(SID)["dossiers"][0]["turns"][0]
    hits, total = t["checks"]["grounded"]
    name, query, failed = t["lookups"][0]
    assert (hits, total, name, failed) == (1, 1, "Web search", False)


@pytest.mark.parametrize("sid", ["../../etc/passwd", "..\\secrets", "ABC", "", None, "0" * 31, "g" * 32])
def test_only_our_own_ids_are_accepted(sid, tmp_sessions):
    assert not persistence.valid_sid(sid)
    persistence.save(sid, _state())
    assert list(tmp_sessions.iterdir()) == []
    assert persistence.load(sid) is None


def test_unknown_or_corrupt_session_starts_fresh(tmp_sessions):
    assert persistence.load(SID) is None
    (tmp_sessions / f"{SID}.json").write_text("{not json", encoding="utf-8")
    assert persistence.load(SID) is None


def test_out_of_range_active_case_is_cleared():
    state = _state()
    state["active"] = 5
    persistence.save(SID, state)
    assert persistence.load(SID)["active"] is None
