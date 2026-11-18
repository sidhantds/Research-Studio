import wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime
import time
import re

def save_to_txt(data: str, filename: str = "Research_output.txt"):
      timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      formatted_text = f"-- Research Output---\nTimestamp:{timestamp}\n\n{data}\n\n"\

      with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
      return f"Data sucessfully saved to {filename}"

save_tool= Tool(
      name="save_text_to_file",
      func= save_to_txt,
      description="seach the web for information",
)

# Cache of successful lookups by normalized query, shared across sessions (public data only).
# Repeat questions and follow-ups reuse results instead of searching again. Failures aren't cached.
_CACHE: dict[tuple[str, str], str] = {}
_CACHE_MAX = 500


def _norm(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _cached(kind: str, query: str, fetch) -> str:
    """Return a cached result, or fetch and cache it. Exceptions propagate uncached."""
    key = (kind, _norm(query))
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))  # drop the oldest entry
        _CACHE[key] = fetch(query)
    return _CACHE[key]


# Engines tried one at a time, fastest and most reliable first. Each gets a short timeout and
# the whole search a total budget, so one engine timing out doesn't sink the lookup.
# (Bing is left out: it ignores the timeout and can hang for ~25s.)
_ENGINES = ["yahoo", "duckduckgo", "mojeek", "brave", "google"]
_ENGINE_TIMEOUT = 8
_SEARCH_BUDGET_SECONDS = 25


def _fetch_web(query: str, timelimit: str | None = None) -> str:
    from ddgs import DDGS

    started, errors = time.monotonic(), []
    for engine in _ENGINES:
        if time.monotonic() - started > _SEARCH_BUDGET_SECONDS:
            break
        try:
            results = DDGS(timeout=_ENGINE_TIMEOUT).text(query, backend=engine, max_results=6 if timelimit else 4,
                                                         timelimit=timelimit)
        except Exception as e:
            errors.append(f"{engine}: {type(e).__name__}")
            continue
        if results:
            return "\n\n".join(
                f"{r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}" for r in results
            )
    raise RuntimeError("no engine returned results (" + ", ".join(errors) + ")")


def _safe_search(query: str) -> str:
    """Never let a network hiccup crash the whole agent run.

    Returns title, snippet and URL for each result so the agent can cite real links
    (and so citations can be checked against what the agent actually read).
    """
    try:
        return _cached("web", query, _fetch_web)
    except Exception as e:
        return (
            f"Web search failed ({type(e).__name__}: {e}). "
            "Try a shorter or different query, or the wikipedia tool. Don't fill gaps from memory."
        )


search_tool = Tool(
    name="duckduckgo_search",
    func=_safe_search,
    description="Search the web for current information. Input should be a search query.",
)

api_wrapper = WikipediaAPIWrapper(
    wiki_client=wikipedia,
    top_k_results=1,
    doc_content_chars_max=1200,  # enough text to reuse as evidence in follow-ups
)
_wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


def _wiki_with_urls(query: str) -> str:
    """Wikipedia results with each article's URL, so they can be cited and checked."""
    text = _wiki.run(query)
    return re.sub(
        r"^Page: (.+)$",
        lambda m: f"Page: {m.group(1)}\nURL: https://en.wikipedia.org/wiki/{m.group(1).strip().replace(' ', '_')}",
        text, flags=re.M,
    )


def _safe_wiki(query: str) -> str:
    """Wikipedia occasionally returns an empty/HTML body (JSONDecodeError); don't crash."""
    try:
        return _cached("wiki", query, _wiki_with_urls)
    except Exception as e:
        return (
            f"Wikipedia lookup failed ({type(e).__name__}). "
            "Try the web search tool or a different query. Don't fill gaps from memory."
        )


wiki_tool = Tool(
    name="wikipedia",
    func=_safe_wiki,
    description=_wiki.description,
)

# ---------- Source rules (allowlist, denylist, recency) ----------
# Enforced here, in code, on every result, so a rule can't be ignored by the model.

RECENCY = {"Any time": None, "Past year": "y", "Past month": "m", "Past week": "w"}


def parse_domains(text: str) -> list[str]:
    """'nasa.gov, .edu  arxiv.org' -> ['nasa.gov', 'edu', 'arxiv.org']"""
    out = []
    for part in re.split(r"[\s,;]+", text or ""):
        part = re.sub(r"^https?://", "", part.strip().lower()).strip("./").removeprefix("www.")
        if part and part not in out:
            out.append(part)
    return out


def _host(url: str) -> str:
    return re.sub(r"^https?://", "", url.strip().lower()).split("/")[0].removeprefix("www.")


def domain_allowed(url: str, allow: list[str], deny: list[str]) -> bool:
    """A domain rule matches the domain itself and its subdomains; 'gov' matches every .gov site."""
    host = _host(url)
    match = lambda rule: host == rule or host.endswith("." + rule)  # noqa: E731
    if any(match(r) for r in deny):
        return False
    return not allow or any(match(r) for r in allow)


def _filter_blocks(text: str, allow: list[str], deny: list[str]) -> tuple[str, int]:
    """Drop result blocks whose URL breaks the rules. Returns (kept text, number dropped)."""
    kept, dropped = [], 0
    for block in text.split("\n\n"):
        m = re.search(r"^URL: (\S+)", block, re.M)
        if m and not domain_allowed(m.group(1), allow, deny):
            dropped += 1
        else:
            kept.append(block)
    return "\n\n".join(kept), dropped


def make_search_tool(allow: list[str], deny: list[str], recency: str = "Any time") -> Tool:
    """Web search that only returns results from allowed domains (and within the recency window)."""
    timelimit = RECENCY.get(recency)
    if not allow and not deny and not timelimit:
        return search_tool

    def search(query: str) -> str:
        q = query
        if allow and len(allow) <= 3:  # steer the engine too; the filter below is what enforces it
            q += " " + " OR ".join(f"site:{d}" for d in allow)
        try:
            text = _cached(f"web:{timelimit}", q, lambda x: _fetch_web(x, timelimit))
        except Exception as e:
            return f"Web search failed ({type(e).__name__}). Try a different query. Don't fill gaps from memory."
        kept, dropped = _filter_blocks(text, allow, deny)
        if not kept.strip():
            return (f"No results from allowed sources ({', '.join(allow) or 'any'}; blocked: "
                    f"{', '.join(deny) or 'none'}). Try a different query, or report the gap. "
                    "Don't fill gaps from memory.")
        return kept + (f"\n\n({dropped} result(s) removed by the source rules.)" if dropped else "")

    return Tool(name=search_tool.name, func=search,
                description=search_tool.description
                + (f" Only results from: {', '.join(allow)}." if allow else "")
                + (f" Limited to the {recency.lower()}." if timelimit else ""))


def make_wiki_tool(allow: list[str], deny: list[str]) -> Tool | None:
    """Wikipedia, unless the source rules exclude wikipedia.org (then None)."""
    if not domain_allowed("https://en.wikipedia.org", allow, deny):
        return None
    return wiki_tool

# ---------- Uploaded documents ----------

def _chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i + size] for i in range(0, max(len(text), 1), size - overlap)] if text else []


def make_documents_tool(docs: list[dict]) -> Tool:
    """Keyword search over the user's uploaded documents ({"name", "text"} dicts)."""
    chunks = [(d["name"], c) for d in docs for c in _chunk(d["text"])]

    def search_documents(query: str) -> str:
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3}
        if not chunks:
            return "No documents have been uploaded."
        scored = sorted(
            ((sum(c.lower().count(w) for w in words), name, c) for name, c in chunks),
            key=lambda t: t[0], reverse=True,
        )
        hits = [t for t in scored[:3] if t[0] > 0]
        if not hits:
            return "No matching passages in the uploaded documents."
        return "\n\n".join(f"[Document: {name}]\n{c}" for _, name, c in hits)

    names = ", ".join(d["name"] for d in docs) or "none"
    return Tool(
        name="search_documents",
        func=search_documents,
        description=f"Search the user's uploaded documents ({names}). Input: keywords or a question.",
    )
