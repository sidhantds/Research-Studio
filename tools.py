import wikipedia
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_core.tools import Tool
from datetime import datetime

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
    """Never let a network hiccup crash the whole agent run."""
    try:
        return _ddg_search.run(query)
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