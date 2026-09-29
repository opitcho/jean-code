"""Tests for `fetch_page`: its windows, and that paging through a page downloads it once. No network: httpx.get
is replaced.

Each test names the bug it would catch.
"""

import pages
from tools import fetch_page


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


def test_reading_a_page_window_by_window_downloads_it_once(serve, monkeypatch):
    # Catches every fetch_page call downloading the whole page again.
    serve["https://x.test/a"] = (b"0123456789" * 3, "text/plain")
    get, calls = pages.httpx.get, []
    monkeypatch.setattr(pages.httpx, "get", lambda url, **kwargs: calls.append(url) or get(url, **kwargs))
    for start in (0, 10, 20):
        fetch_page("https://x.test/a", max_chars=10, start=start)
    assert calls == ["https://x.test/a"]
