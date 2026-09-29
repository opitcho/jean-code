"""Reading web pages: download a page as text (HTML, PDF or plain), and cut that text into windows.
"""

import io
import logging
from functools import lru_cache

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

# pypdf logs a warning per font it can't fully decode (dozens per paper), which would flood the REPL; the text is
# still extracted.
logging.getLogger("pypdf").setLevel(logging.ERROR)


@lru_cache(maxsize=64)  # fetch_page reads a long page one window per call; download it once
def read_page(url: str) -> str:
    """Download a page and return its whole readable text: an HTML page's title and visible text, a PDF's text
    (by content type, or a body starting with `%PDF`), or any other body as it is. Raises on HTTP errors.

    Agents never get this whole text: they read it through `tools.fetch_page`, a window at a time."""
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
    return html_text(response.text)


def html_text(html: str) -> str:
    """An HTML page's title and the text a reader sees, one line per text node. Also dropped: MathML's TeX source
    (arXiv math would read "333 × \\times"), and screen-reader-only text like "(opens in a new window)"."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "form", "annotation",
                     "annotation-xml"]):
        tag.decompose()
    for tag in soup.select(".sr-only, .visually-hidden"):
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
