"""The web tools: `web_search` and `fetch_page`.

Every function here is a tool a model calls, and nothing else is: plain helpers go in `pages`. Each is fully typed
and documents its arguments, since its schema is built from them.
"""

from ddgs import DDGS

from jean_code.core.tool_schema import tool
from jean_code.tools import pages


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
