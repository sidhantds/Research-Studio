"""Small eval suite for the parts of the pipeline that decide quality.

    python evals/run_evals.py          # offline: keyword router + quote checker, no API key needed
    python evals/run_evals.py --live   # also runs the full router (rules + Gemini fallback)

Routing: each case has the response type a person would expect. Offline, only the keyword
rules run, so a case they don't decide is reported as "deferred to the model", not as wrong.
Quote checking: does the word-for-word matcher accept real quotes and reject paraphrased,
altered or invented ones? A false accept is the dangerous error: it would let an unsupported
claim count as supported.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agent import PROMPT_VERSION, MODEL_NAME, keyword_route, route_request  # noqa: E402
from checks import norm_text, quote_in  # noqa: E402


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_routing(live: bool) -> None:
    cases = load("routing.jsonl")
    llm = None
    if live:
        from agent import get_llm
        llm = get_llm()

    correct, deferred, misses = 0, 0, []
    confusion: Counter = Counter()
    for c in cases:
        if live:
            route = route_request(llm, c["question"], "a research session on this topic" if c["session"] else None)
        else:
            route = keyword_route(c["question"], c["session"])
        if route is None:
            deferred += 1
            continue
        got = route.response_type
        confusion[(c["expected"], got)] += 1
        if got == c["expected"]:
            correct += 1
        else:
            misses.append((c["question"], c["expected"], got))

    decided = len(cases) - deferred
    print(f"## Routing ({'rules + model' if live else 'keyword rules only'})\n")
    print("| Metric | Value |\n|---|---|")
    print(f"| Cases | {len(cases)} |")
    if not live:
        print(f"| Decided by rules (no model call) | {decided} ({decided / len(cases):.0%}) |")
        print(f"| Deferred to the model | {deferred} |")
    print(f"| Accuracy on decided cases | {correct}/{decided} ({correct / max(decided, 1):.0%}) |")
    if misses:
        print("\n| Misrouted question | Expected | Got |\n|---|---|---|")
        for q, exp, got in misses:
            print(f"| {q} | {exp} | {got} |")
    print()


def eval_quotes() -> None:
    cases = load("quotes.jsonl")
    tp = fp = tn = fn = 0
    wrong = []
    for c in cases:
        got = quote_in(c["quote"], norm_text(c["evidence"]))
        if got and c["expected"]:
            tp += 1
        elif got and not c["expected"]:
            fp += 1
        elif not got and not c["expected"]:
            tn += 1
        else:
            fn += 1
        if got != c["expected"]:
            wrong.append(c)

    print("## Quote checking\n")
    print("| Metric | Value |\n|---|---|")
    print(f"| Cases | {len(cases)} |")
    print(f"| Real quotes accepted | {tp}/{tp + fn} |")
    print(f"| Altered or invented quotes rejected | {tn}/{tn + fp} |")
    print(f"| **False accepts** (unsupported claim would pass) | **{fp}** |")
    for c in wrong:
        print(f"\n- Wrong on: {c['note']} (expected {'accept' if c['expected'] else 'reject'})")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="route with the model too (needs GOOGLE_API_KEY)")
    args = ap.parse_args()
    print(f"# Research Studio evals\n\nModel `{MODEL_NAME}` · prompt version `{PROMPT_VERSION}`\n")
    eval_routing(args.live)
    eval_quotes()
