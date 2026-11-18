"""Source rules and document search: enforced in code, so they're tested in code."""

import pytest

import tools
from tools import _filter_blocks, domain_allowed, make_documents_tool, make_search_tool, parse_domains


def test_parse_domains_normalises_input():
    assert parse_domains("https://www.NASA.gov/, .edu  arxiv.org; nasa.gov") == ["nasa.gov", "edu", "arxiv.org"]
    assert parse_domains("") == []


@pytest.mark.parametrize("url, allow, deny, expected", [
    ("https://www.nasa.gov/page", [], [], True),
    ("https://science.nasa.gov/x", ["nasa.gov"], [], True),       # subdomain of an allowed domain
    ("https://example.com", ["nasa.gov"], [], False),             # not on the allowlist
    ("https://mit.edu/paper", ["edu"], [], True),                 # a bare TLD rule
    ("https://www.reddit.com/r/x", [], ["reddit.com"], False),
    ("https://notreddit.com", [], ["reddit.com"], True),          # suffix must be a whole label
    ("https://old.reddit.com", ["reddit.com"], ["reddit.com"], False),  # deny beats allow
])
def test_domain_rules(url, allow, deny, expected):
    assert domain_allowed(url, allow, deny) is expected


def test_filter_blocks_drops_denied_results():
    text = "A\nbody\nURL: https://good.org/1\n\nB\nbody\nURL: https://bad.com/2"
    kept, dropped = _filter_blocks(text, [], ["bad.com"])
    assert "good.org" in kept and "bad.com" not in kept
    assert dropped == 1


def test_search_tool_enforces_rules_without_trusting_the_model(monkeypatch):
    fake = "A\nbody\nURL: https://allowed.org/1\n\nB\nbody\nURL: https://other.com/2"
    monkeypatch.setattr(tools, "_cached", lambda kind, q, fetch: fake)
    out = make_search_tool(["allowed.org"], []).func("anything")
    assert "allowed.org" in out
    assert "other.com" not in out
    assert "1 result(s) removed" in out


def test_search_tool_reports_when_every_result_is_filtered(monkeypatch):
    monkeypatch.setattr(tools, "_cached", lambda kind, q, fetch: "A\nbody\nURL: https://other.com/2")
    out = make_search_tool(["allowed.org"], []).func("anything")
    assert out.startswith("No results from allowed sources")


def test_no_rules_means_the_plain_search_tool():
    assert make_search_tool([], []) is tools.search_tool


def test_failed_search_returns_a_message_instead_of_raising(monkeypatch):
    def boom(kind, q, fetch):
        raise TimeoutError("slow")
    monkeypatch.setattr(tools, "_cached", boom)
    assert tools._safe_search("q").startswith("Web search failed")


def test_document_search_returns_best_matching_chunk():
    docs = [{"name": "notes.txt", "text": "Solar panels convert light. " * 5 + "Wind turbines spin. " * 5}]
    tool = make_documents_tool(docs)
    out = tool.func("solar panels")
    assert out.startswith("[Document: notes.txt]")
    assert "Solar panels" in out
    assert make_documents_tool([]).func("anything") == "No documents have been uploaded."


def test_no_tool_can_write_to_disk():
    from agent import TOOLS
    assert {t.name for t in TOOLS} == {"duckduckgo_search", "wikipedia"}
