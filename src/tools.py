"""Tools an agent can register in its `tool_map`.

Ollama builds each tool's schema from its type hints and Google-style docstring,
so every tool here is fully typed and documents its arguments.
"""

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First number to add.
        b: Second number to add.

    Returns:
        The sum of a and b.
    """
    return a + b


def sub(a: int, b: int) -> int:
    """Subtract one integer from another.

    Args:
        a: Number to subtract from.
        b: Number to subtract.

    Returns:
        The difference a - b.
    """
    return a - b


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


def fetch_page(url: str, max_chars: int = 8000) -> str:
    """Fetch a web page and return its readable text.

    Args:
        url: Full URL of the page, including http:// or https://.
        max_chars: Maximum number of characters of text to return.

    Returns:
        The page's title and visible text, truncated to max_chars.
    """
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; jean-code/0.1)"},
    )
    response.raise_for_status()

    if "html" not in response.headers.get("content-type", "html"):
        text = response.text
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        lines = (line.strip() for line in soup.get_text("\n").splitlines())
        text = "\n".join(line for line in lines if line)
        if title:
            text = f"{title}\n\n{text}"

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated: {len(text) - max_chars} more characters]"
    return text
