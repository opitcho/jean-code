"""The web tools: every function here is one a model calls, and nothing else is.

`llm.tool_schema` builds each tool's schema from its type hints and Google-style docstring,
so every tool here is fully typed and documents its arguments. Plain helpers belong elsewhere (e.g. `pages`).
Tools bound to state live with it, marked `@tool` too: `Shell.bash`, the research `Notebook`'s, the agent's own.
"""

from ddgs import DDGS

import pages
from llm import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo.

    Args:
        query: What to search for.
        max_results: Maximum number of results to return.

    Returns:
        One block per result with its title, URL and snippet.
    """
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return f"No results for '{query}'."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


@tool
def fetch_page(url: str, max_chars: int = 8000, start: int = 0) -> str:
    """Fetch a web page or PDF and return part of its readable text.

    Args:
        url: Full URL of the page, including http:// or https://.
        max_chars: Maximum number of characters of text to return.
        start: Character to start at. A long page ends with a line saying where the part shown ends; fetch again
            with that position as `start` to read on.

    Returns:
        The page's title and visible text (or a PDF's text), from `start`, cut to max_chars.
    """
    text = pages.read_page(url)
    return pages.window_text(text, *pages.window(text, start, max_chars))
