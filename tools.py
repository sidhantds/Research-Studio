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


def _fetch_web(query: str) -> str:
    from ddgs import DDGS

    started, errors = time.monotonic(), []
    for engine in _ENGINES:
        if time.monotonic() - started > _SEARCH_BUDGET_SECONDS:
            break
        try:
            results = DDGS(timeout=_ENGINE_TIMEOUT).text(query, backend=engine, max_results=4)
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
