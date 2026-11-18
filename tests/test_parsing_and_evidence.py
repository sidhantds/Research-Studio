"""Answer parsing (the JSON repair path) and evidence compression for cheap follow-ups."""

import json

import pytest

import agent
from agent import Answer, Source, parse_response, plan_to_instructions
from evidence import build_pack, compress_step, keywords, split_results, top_items

# ---------- parse_response ----------

VALID = {"response_type": "code", "title": "Hello · Java", "direct_answer": "Prints hello."}


def test_strict_json_parses():
    assert parse_response(json.dumps(VALID)).title == "Hello · Java"


def test_json_wrapped_in_prose_is_extracted():
    text = "Sure! Here you go:\n```json\n" + json.dumps(VALID) + "\n```\nHope that helps."
    assert parse_response(text).response_type == "code"


def test_answer_wrapped_in_the_schema_itself_is_unwrapped():
    assert parse_response(json.dumps({"properties": VALID})).direct_answer == "Prints hello."


def test_empty_reply_raises_without_calling_the_model(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: pytest.fail("repair must not run on an empty reply"))
    with pytest.raises(Exception):
        parse_response("   ")


def test_unparseable_reply_gets_one_repair_call(monkeypatch):
    calls = []

    class FakeLLM:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt, config=None):
            calls.append(prompt)
            return Answer(direct_answer="repaired")

    monkeypatch.setattr(agent, "get_llm", FakeLLM)
    assert parse_response("plain prose, no JSON at all").direct_answer == "repaired"
    assert len(calls) == 1


# ---------- schema validators ----------

def test_source_drops_non_http_urls_and_fills_domain():
    assert Source(id=1, url="javascript:alert(1)").url is None
    s = Source(id="[2]", url="https://www.nature.com/articles/x")
    assert s.id == "2" and s.domain == "nature.com"


def test_plan_instructions_name_the_tools():
    text = plan_to_instructions([{"question": "Costs?", "sources": ["Web", "Wikipedia"]}])
    assert "1. Costs? (use web search and wikipedia)" in text
    assert plan_to_instructions(None) == ""


# ---------- evidence ----------

def test_split_web_results(web_step):
    blocks = split_results(web_step.tool, web_step.output)
    assert [b["url"] for b in blocks] == ["https://www.example.org/sodium-ion", "https://energy.example.com/report"]


def test_failed_lookups_give_no_evidence():
    assert split_results("duckduckgo_search", "Web search failed (TimeoutError)") == []


def test_wikipedia_results_split_by_page():
    out = "Page: Battery\nURL: https://en.wikipedia.org/wiki/Battery\nSummary: A battery stores energy."
    [b] = split_results("wikipedia", out)
    assert b["title"] == "Battery" and b["text"] == "A battery stores energy."


def test_compress_step_deduplicates(web_step):
    registry, items = [], []
    assert compress_step(web_step.tool, web_step.tool_input, web_step.output, "sodium cost", registry, items) == 2
    assert compress_step(web_step.tool, web_step.tool_input, web_step.output, "sodium cost", registry, items) == 0
    assert [s["id"] for s in registry] == ["R1", "R2"]
    assert all(i["quote"] in web_step.output.replace("\n", " ") for i in items)  # quotes are verbatim


def test_top_items_ranks_by_the_follow_up(web_step):
    registry, items = [], []
    compress_step(web_step.tool, web_step.tool_input, web_step.output, "sodium", registry, items)
    assert top_items(items, "utilities grid storage", k=1)[0]["url"] == "https://energy.example.com/report"


def test_keywords_drop_stopwords_and_stem():
    assert keywords("The batteries are charging") == {"batteri", "charg"}


def test_pack_never_contains_raw_pages(web_step):
    registry, items = [], []
    compress_step(web_step.tool, web_step.tool_input, web_step.output, "sodium", registry, items)
    pack = build_pack("Is sodium-ion viable?", "Yes [1].", None, ["Is sodium-ion viable?"], items, "cost?")
    assert pack.startswith("ORIGINAL QUESTION: Is sodium-ion viable?")
    assert "EVIDENCE (strongest items" in pack
    assert web_step.output not in pack
