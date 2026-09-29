"""Tests for reading pages: `read_page` (HTML, PDF, other text) and `html_text`. No network: httpx.get is replaced.

Each test names the bug it would catch.
"""

from conftest import make_pdf
from pages import html_text, read_page


def test_html_gives_the_title_and_visible_text(serve):
    # Catches the split into read_page losing the HTML cleanup.
    html = b"<html><head><title>T</title><script>var x;</script></head><body><nav>menu</nav><p>Body text</p></body></html>"
    serve["https://x.test/h"] = (html, "text/html; charset=utf-8")
    text = read_page("https://x.test/h")
    assert text.startswith("T\n\n") and "Body text" in text
    assert "var x" not in text and "menu" not in text


def test_html_text_drops_math_tex_source_and_screen_reader_text():
    # Catches arXiv math read twice ("333 × \times") and links read with "(opens in a new window)", which no reader
    # of the page sees and so no researcher quotes.
    html = ('<p>up to 333<math alttext="\\times"><semantics><mo>×</mo><annotation encoding="application/x-tex">'
            '\\times</annotation></semantics></math> relative to a <a>hard<span class="sr-only">(opens in a new '
            'window)</span></a> scramble</p>')
    assert html_text(html) == "up to 333\n×\nrelative to a\nhard\nscramble"


def test_a_pdf_gives_its_text_by_content_type_or_by_its_first_bytes(serve):
    # Catches a PDF returned as binary noise, as fetch_page did before pypdf.
    pdf = make_pdf(["Hello from a PDF.", "Second line here."])
    serve["https://x.test/p.pdf"] = (pdf, "application/pdf")
    serve["https://x.test/download"] = (pdf, "application/octet-stream")
    for url in ("https://x.test/p.pdf", "https://x.test/download"):
        text = read_page(url)
        assert "Hello from a PDF." in text and "Second line here." in text
