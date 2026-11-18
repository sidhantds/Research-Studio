"""Keyword routing: the rules that decide most requests without a model call."""

import pytest

from agent import keyword_route


@pytest.mark.parametrize("question, expected", [
    ("Generate hello world in Java", "code"),
    ("Write a Python function to reverse a string", "code"),
    ("Research the history of solid-state batteries", "research"),
    ("Compare lithium-ion vs sodium-ion batteries", "comparison"),
    ("What is the difference between TCP and UDP?", "comparison"),
    ("Is it true that the Great Wall is visible from space?", "fact_check"),
    ("Fact-check: humans use only 10% of their brains", "fact_check"),
    ("What are the latest trends in renewable energy?", "research"),
])
def test_clear_requests_are_routed_by_rules(question, expected):
    route = keyword_route(question, has_session=False)
    assert route is not None
    assert route.response_type == expected
    assert route.reason.startswith("Rule:")


def test_code_requests_need_no_tools():
    route = keyword_route("Write a bash script that backs up a folder", has_session=False)
    assert route.response_type == "code"
    assert route.needs_tools is False


def test_code_request_about_recent_facts_is_not_treated_as_code():
    # "latest" means fresh facts are needed, so it must not skip research
    route = keyword_route("Write code using the latest 2026 stock market statistics", has_session=False)
    assert route.response_type == "research"
    assert route.needs_tools is True


def test_explain_this_only_without_research_inside_a_session():
    question = "Explain how this code works"
    assert keyword_route(question, has_session=True).response_type == "explanation"
    assert keyword_route(question, has_session=False) is None  # nothing to refer to: let the model decide


def test_ambiguous_request_falls_through_to_the_model():
    assert keyword_route("Tell me about Mars", has_session=False) is None


def test_first_matching_rule_wins():
    # "research" is checked before "compare"
    route = keyword_route("Research and compare React and Vue", has_session=False)
    assert route.response_type == "research"
