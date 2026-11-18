"""Structured evidence for research sessions.

Raw tool output (search results, Wikipedia extracts, document chunks) is split into one item
per source and compressed to its most relevant sentence and passage. Follow-ups are answered
from the strongest few items instead of a blunt character cap, so the model gets the facts it
needs at a fraction of the tokens. No model calls happen here: it's fast and repeatable.

The raw text is still kept on the session (outside this module) for the audit trail and for
checking quotes word for word; it just isn't sent to the model again.
"""

import re
from datetime import datetime
from urllib.parse import urlparse

_STOP = set("""a an and are as at be been but by can did do does for from has have how i in into is it its
of on or so than that the their them then there these they this to too was were what when where which who
why will with would you your about after also any could more most not only other our out over such use used
using very via vs""".split())

PACK_ITEMS = 8  # evidence items sent with a follow-up (the spec's 5-10 strongest)
_QUOTE_CHARS = 350
_CLAIM_CHARS = 220


def _stem(w: str) -> str:
    """Crude suffix stripping so 'launched', 'launches' and 'launch' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def keywords(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower()) if len(w) > 2 and w not in _STOP}


def score_relevance(text: str, query_words: set[str]) -> float:
    """Share of the query's keywords that appear in the text (0 to 1)."""
    if not query_words:
        return 0.0
    return round(len(keywords(text) & query_words) / len(query_words), 3)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", text) if len(s.strip()) > 20]


def extract_claim(text: str, query_words: set[str]) -> str:
    """The single most relevant sentence: what this source says about the question."""
    sents = _sentences(text)
    if not sents:
        return text.strip()[:_CLAIM_CHARS]
    best = max(sents, key=lambda s: (score_relevance(s, query_words), -abs(len(s) - 140)))
    return best[:_CLAIM_CHARS]


def extract_best_quote(text: str, query_words: set[str]) -> str:
    """The most relevant passage, word for word: the best sentence plus its neighbour if it fits."""
    sents = _sentences(text)
    if not sents:
        return text.strip()[:_QUOTE_CHARS]
    i = max(range(len(sents)), key=lambda k: score_relevance(sents[k], query_words))
    quote = sents[i]
    if i + 1 < len(sents) and len(quote) + len(sents[i + 1]) < _QUOTE_CHARS:
        quote += " " + sents[i + 1]
    return quote[:_QUOTE_CHARS]


def split_results(tool: str, output: str) -> list[dict]:
    """Raw tool output -> one {title, url, text} per source. Failures and empty results give []."""
    if not output or re.match(r"^(Web search failed|Wikipedia lookup failed|No results|No matching|"
                              r"No documents|Search budget)", output):
        return []
    blocks = []
    if tool == "search_documents":
        for m in re.finditer(r"\[Document: ([^\]]+)\]\n(.*?)(?=\n\n\[Document: |\Z)", output, re.S):
            blocks.append({"title": m.group(1), "url": None, "text": m.group(2)})
        return blocks
    if tool == "wikipedia":
        for m in re.finditer(r"Page: (.+?)\nURL: (\S+)\nSummary: (.*?)(?=\n\nPage: |\Z)", output, re.S):
            blocks.append({"title": m.group(1).strip(), "url": m.group(2), "text": m.group(3)})
        return blocks
    for block in output.split("\n\n"):  # web: title / body / URL
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        url = next((ln[5:].strip() for ln in lines if ln.startswith("URL: ")), None)
        body = [ln for ln in lines[1:] if not ln.startswith("URL: ")]
        if url and body:
            blocks.append({"title": lines[0].strip(), "url": url, "text": " ".join(body)})
    return blocks


def compress_evidence(block: dict, source: dict, query_words: set[str]) -> dict:
    """One evidence item: what the source says, the passage that says it, and how relevant it is."""
    return {
        "source_id": source["id"],
        "title": source["title"],
        "url": source["url"],
        "domain": source["domain"],
        "claim": extract_claim(block["text"], query_words),
        "quote": extract_best_quote(block["text"], query_words),
        "retrieved_at": source["retrieved_at"],
        "confidence": score_relevance(block["text"], query_words),
    }


def register_source(registry: list[dict], title: str, url: str | None) -> dict:
    """Find or add a retrieved source in the session's registry (ids are stable: "R1", "R2", ...)."""
    key = (url or title).strip().lower().rstrip("/")
    for s in registry:
        if (s["url"] or s["title"]).strip().lower().rstrip("/") == key:
            return s
    s = {"id": f"R{len(registry) + 1}", "title": title, "url": url,
         "domain": urlparse(url).netloc.removeprefix("www.") if url else title,
         "retrieved_at": datetime.now().isoformat(timespec="seconds")}
    registry.append(s)
    return s


def compress_step(tool: str, tool_input: str, output: str, question: str,
                  registry: list[dict], items: list[dict]) -> int:
    """Add a lookup's results to the session as evidence items. Returns how many were new."""
    qw = keywords(question) | keywords(tool_input)
    added = 0
    seen = {(i["url"] or i["title"], i["quote"]) for i in items}
    for block in split_results(tool, output):
        item = compress_evidence(block, register_source(registry, block["title"], block["url"]), qw)
        if (item["url"] or item["title"], item["quote"]) not in seen:
            items.append(item)
            seen.add((item["url"] or item["title"], item["quote"]))
            added += 1
    return added


def top_items(items: list[dict], question: str, k: int = PACK_ITEMS) -> list[dict]:
    """The k items most relevant to this question (re-scored now, not at retrieval time)."""
    qw = keywords(question)
    ranked = sorted(items, key=lambda i: (score_relevance(i["claim"] + " " + i["quote"] + " " + i["title"], qw),
                                          i["confidence"]), reverse=True)
    return ranked[:k]


def build_pack(original_question: str, current_answer: str, current_code: str | None,
               earlier_questions: list[str], items: list[dict], followup: str) -> str:
    """The context sent with a follow-up: no raw pages, no full chat history."""
    parts = [f"ORIGINAL QUESTION: {original_question}"]
    if earlier_questions:
        parts.append("EARLIER QUESTIONS (for reference only): " + " | ".join(q[:120] for q in earlier_questions[-3:]))
    parts.append(f"CURRENT ANSWER:\n{current_answer[:1200]}")
    if current_code:
        parts.append(f"CURRENT CODE:\n```\n{current_code[:6000]}\n```")
    chosen = top_items(items, followup + " " + original_question)
    if chosen:
        parts.append("EVIDENCE (strongest items for this follow-up):\n" + "\n\n".join(
            f"- {i['title']} · {i['domain']}" + (f" · {i['url']}" if i["url"] else "")
            + f" · retrieved {i['retrieved_at'][:16].replace('T', ' ')}\n  says: {i['claim']}\n  quote: \"{i['quote']}\""
            for i in chosen))
    else:
        parts.append("EVIDENCE: (none gathered yet)")
    return "\n\n".join(parts)
