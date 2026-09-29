"""Tools to pass to an agent in `tools`, and the page-reading helpers they share.

`llm.tool_schema` builds each tool's schema from its type hints and Google-style docstring,
so every tool here is fully typed and documents its arguments.
"""

import io
import logging

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from pypdf import PdfReader

# pypdf logs a warning per font it can't fully decode (dozens per paper), which would flood the REPL; the text is
# still extracted.
logging.getLogger("pypdf").setLevel(logging.ERROR)


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


def read_page(url: str) -> str:
    """Download a page and return its whole readable text: an HTML page's title and visible text, a PDF's text
    (by content type, or a body starting with `%PDF`), or any other body as it is. Raises on HTTP errors."""
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; jean-code/0.1)"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "html")
    if "pdf" in content_type or response.content.startswith(b"%PDF"):
        return pdf_text(response.content)
    if "html" not in content_type:
        return response.text
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line)
    return f"{title}\n\n{text}" if title else text


def pdf_text(data: bytes) -> str:
    """The text of every page of a PDF, pages separated by blank lines."""
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def window(text: str, start: int, max_chars: int) -> tuple[int, int]:
    """The (start, end) of the part of `text` that fetch_page returns: at most `max_chars` from `start`."""
    start = min(max(start, 0), len(text))
    return start, min(start + max_chars, len(text))


def window_text(text: str, start: int, end: int) -> str:
    """`text[start:end]`, followed by a line saying where it is unless it's the whole text."""
    if start == 0 and end == len(text):
        return text
    if start == end:
        return f"[nothing at start={start}: the page has {len(text)} characters]"
    more = f"fetch again with start={end} for more" if end < len(text) else "the end of the page"
    return f"{text[start:end]}\n\n[characters {start}–{end} of {len(text)}; {more}]"


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
    text = read_page(url)
    return window_text(text, *window(text, start, max_chars))
