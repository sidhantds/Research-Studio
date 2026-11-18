import wikipedia
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_core.tools import Tool
from datetime import datetime
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
# ddgs has no "html" backend (it silently fell back to random engines, incl.
# Startpage which times out). List reliable engines explicitly; ddgs tries them in turn.
_ddg_search = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(
        backend="duckduckgo,bing,mojeek,yahoo", max_results=3
    )
)


def _safe_search(query: str) -> str:
    """Never let a network hiccup crash the whole agent run.

    Returns title, snippet and URL for each result so the agent can cite real links
    (and so citations can be checked against what the agent actually read).
    """
    try:
        results = _ddg_search.api_wrapper.results(query, max_results=3)
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"{r.get('title', '')}\n{r.get('snippet', '')}\nURL: {r.get('link', '')}" for r in results
        )
    except Exception as e:
        return (
            f"Web search failed ({type(e).__name__}: {e}). "
            "Use the wikipedia tool or your own knowledge instead."
        )


search_tool = Tool(
    name="duckduckgo_search",
    func=_safe_search,
    description="Search the web for current information. Input should be a search query.",
)

api_wrapper = WikipediaAPIWrapper(
    wiki_client=wikipedia,
    top_k_results=1,
    doc_content_chars_max=100,
)
_wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


def _safe_wiki(query: str) -> str:
    """Wikipedia occasionally returns an empty/HTML body (JSONDecodeError); don't crash."""
    try:
        return _wiki.run(query)
    except Exception as e:
        return (
            f"Wikipedia lookup failed ({type(e).__name__}). "
            "Use the duckduckgo_search tool or your own knowledge instead."
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
