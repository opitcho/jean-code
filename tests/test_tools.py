"""Tests for the page tools: `read_page` (HTML, PDF, other text) and `fetch_page`'s windows. No network: httpx.get
is replaced.

Each test names the bug it would catch.
"""

import httpx
import pytest

import tools
from conftest import make_pdf
from tools import fetch_page, read_page


@pytest.fixture
def serve(monkeypatch):
    """Serve `pages[url] = (body, content type)` in place of the web."""
    pages: dict[str, tuple[bytes, str]] = {}

    def get(url, **kwargs):
        body, content_type = pages[url]
        return httpx.Response(200, content=body, headers={"content-type": content_type},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(tools.httpx, "get", get)
    return pages


def test_fetch_page_returns_the_window_from_start_and_says_where_it_is(serve):
    # Catches a `start` that's ignored, an off-by-one window, or a position line the model can't continue from.
    serve["https://x.test/a"] = (b"0123456789" * 3, "text/plain")
    assert fetch_page("https://x.test/a", max_chars=10) == (
        "0123456789\n\n[characters 0–10 of 30; fetch again with start=10 for more]")
    assert fetch_page("https://x.test/a", max_chars=12, start=10) == (
        "012345678901\n\n[characters 10–22 of 30; fetch again with start=22 for more]")
    assert fetch_page("https://x.test/a", max_chars=10, start=25) == (
        "56789\n\n[characters 25–30 of 30; the end of the page]")


def test_a_page_that_fits_is_returned_whole_and_a_start_past_the_end_says_so(serve):
    serve["https://x.test/a"] = (b"short page", "text/plain")
    assert fetch_page("https://x.test/a") == "short page"
    assert fetch_page("https://x.test/a", start=50) == "[nothing at start=10: the page has 10 characters]"


def test_html_gives_the_title_and_visible_text(serve):
    # Catches the split into read_page losing the HTML cleanup.
    html = b"<html><head><title>T</title><script>var x;</script></head><body><nav>menu</nav><p>Body text</p></body></html>"
    serve["https://x.test/h"] = (html, "text/html; charset=utf-8")
    text = read_page("https://x.test/h")
    assert text.startswith("T\n\n") and "Body text" in text
    assert "var x" not in text and "menu" not in text


def test_a_pdf_gives_its_text_by_content_type_or_by_its_first_bytes(serve):
    # Catches a PDF returned as binary noise, as fetch_page did before pypdf.
    pdf = make_pdf(["Hello from a PDF.", "Second line here."])
    serve["https://x.test/p.pdf"] = (pdf, "application/pdf")
    serve["https://x.test/download"] = (pdf, "application/octet-stream")
    for url in ("https://x.test/p.pdf", "https://x.test/download"):
        text = read_page(url)
        assert "Hello from a PDF." in text and "Second line here." in text
