"""Tests for research runs: the notebook's tools, the rendered report, and how runs end. No network and no model:
`pages.read_page` serves fixed pages, and a client scripted per task plays the lead and each researcher.

Each test names the bug it would catch.
"""

import copy
import json
import threading
from pathlib import Path

import pytest

from conftest import call, response, usage
from pages import html_text
from research import (Notebook, check_quote, read_facts, render_report, research_lead, research_profile,
                      research_result, researcher_profile, summary_of)
from subagents import Run
from usage import Budget

TIMEOUT = 5  # seconds any wait in these tests may take before it counts as a hang
FIXTURES = Path(__file__).parent / "fixtures"

PAGE_A = "Alpha Tool\n\nAlpha is a package manager written in Rust. It installs packages ten times faster than pip."
PAGE_B = "Beta docs\n\nBeta stores its lockfile as TOML in the project root. The format is stable since version two."
QUOTE_A = "Alpha is a package manager written in Rust."
QUOTE_B = "Beta stores its lockfile as TOML in the project root."


@pytest.fixture
def pages(monkeypatch):
    """Serve `pages[url] = text` (or a callable returning it) in place of the web."""
    served: dict = {}

    def read_page(url: str) -> str:
        page = served[url]
        return page() if callable(page) else page

    monkeypatch.setattr("pages.read_page", read_page)
    return served


class TaskClient:
    """A scripted client per agent: each request goes to the script of its first user message (the agent's task).

    An item is a response, an exception to raise, or a callable returning a response (to wait for something first).
    Thread-safe, since the lead and its researchers call it at once.
    """

    def __init__(self, scripts: dict[str, list]):
        self.scripts = {task: list(items) for task, items in scripts.items()}
        self._lock = threading.Lock()

    def complete(self, request: dict) -> dict:
        task = next(m["content"] for m in request["messages"] if m["role"] == "user")
        with self._lock:
            script = self.scripts.get(task)
            if not script:
                raise AssertionError(f"no scripted response left for task {task!r}")
            item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item() if callable(item) else item)


def new_notebook(tmp_path) -> Notebook:
    return Notebook.create(tmp_path, brief="the brief", path="main", model="m")


def record(notebook: Notebook, url: str, page: str, pages: dict, quote: str, claim: str = "a claim",
           by: str = "main/1", date: str | None = None) -> str:
    """Fetch the page as researcher `by`, then record one fact from it."""
    pages.setdefault(url, page)
    researcher = notebook.tools_for(by)
    researcher["fetch_page"](url)
    return researcher["record_fact"](claim, url, quote, date)


def researcher_script(url: str, quote: str, claim: str, cost: float = 0.001) -> list:
    """A researcher that fetches one page, records one fact from it, and ends with a note."""
    return [response(tool_calls=[call("fetch_page", url=url), call("record_fact", claim=claim, url=url, quote=quote)],
                     usage=usage(cost=cost)),
            response(f"covered {claim}")]


def outcome_dir(text: str) -> Path:
    """The run folder named on a result's first line: "Report: <dir>/report.md"."""
    return Path(text.splitlines()[0].removeprefix("Report: ")).parent


# ---------------------------------------------------------------- pages and facts


def test_fetch_page_saves_each_window_it_returned_once(tmp_path, pages):
    # Catches a quote checked against text no model read, or a page file that grows on every refetch.
    notebook = new_notebook(tmp_path)
    pages["https://a.test/"] = "x" * 30
    fetch = notebook.tools_for("main/1")["fetch_page"]
    fetch("https://a.test/", max_chars=10)
    fetch("https://a.test/", max_chars=10)
    notebook.tools_for("main/2")["fetch_page"]("https://a.test/", max_chars=10, start=10)
    index = json.loads((notebook.dir / "pages" / "index.json").read_text())
    assert index == {"https://a.test": {"file": "P1.txt", "length": 30, "windows": [[0, 10], [10, 20]],
                                        "by": ["main/1", "main/2"]}}
    assert (notebook.dir / "pages" / "P1.txt").read_text() == "x" * 10 + "\n\n" + "x" * 10 + "\n\n"


def test_record_fact_verifies_a_verbatim_quote_and_flags_a_paraphrase(tmp_path, pages):
    # Catches a checker that passes anything, or fails on spacing and case the page doesn't matter for.
    notebook = new_notebook(tmp_path)
    assert record(notebook, "https://a.test", PAGE_A, pages, "  alpha IS a package\nmanager written in Rust. ") == (
        "F1 recorded (quote verified)")
    assert record(notebook, "https://a.test", PAGE_A, pages, "Alpha is a Rust-based package manager tool.") == (
        "F2 recorded, NOT verified: the quote isn't on the page (as read so far). "
        "Fix the quote with record_fact again, or leave it.")
    facts = read_facts(notebook.dir)
    assert [(f.id, f.verified, f.problem) for f in facts] == [
        ("F1", True, None), ("F2", False, "the quote isn't on the page (as read so far)")]


def test_record_fact_refuses_a_page_nobody_fetched_and_writes_nothing(tmp_path, pages):
    # Catches citations to pages that were never read: the URL a model typed from memory.
    notebook = new_notebook(tmp_path)
    reply = notebook.tools_for("main/1")["record_fact"]("a claim", "https://never.test", QUOTE_A)
    assert reply == "Not recorded: https://never.test wasn't fetched in this run; fetch_page it first."
    assert not (notebook.dir / "facts.jsonl").exists()


def test_record_fact_finds_the_page_under_either_spelling_of_its_url(tmp_path, pages):
    # Catches a correct fact refused because the model added a #fragment or a trailing slash.
    notebook = new_notebook(tmp_path)
    pages["https://a.test/doc"] = PAGE_A
    researcher = notebook.tools_for("main/1")
    researcher["fetch_page"]("https://a.test/doc")
    for url in ("https://a.test/doc/", "https://a.test/doc#install"):
        assert "(quote verified)" in researcher["record_fact"]("a claim", url, QUOTE_A)


@pytest.mark.parametrize("fixture, quote", [
    ("pep8_excerpt.txt", "One of Guido's key insights is that code is read much more often than it is written."),
    ("pep8_excerpt.txt", 'As PEP 20 says, "Readability counts". A style guide is about consistency.'),
    ("pep8_excerpt.txt", "However, know when to be inconsistent - sometimes style guide recommendations"),
    ("wikipedia_python_excerpt.txt", "There should be one-and preferably only one-obvious way to do it."),
    ("wikipedia_python_excerpt.txt", "IEEE 754 number with 53 bits of precision"),
    # Quotes from a live run that failed before spaces before punctuation were dropped: HTML extraction puts each
    # link on its own line, so the page says "splitting\n, merging" and "sort=True)\n.".
    ("pypi_pypdf_page.txt", "pypdf is a free and open-source pure-python PDF library capable of splitting, merging, "
                            "cropping, and transforming the pages of PDF files."),
    ("pymupdf_app1_page.txt", 'you can request a reordering following the scheme "top-left to bottom-right" by '
                              'executing page.get_text("text", sort=True).'),
])
def test_quotes_verify_against_real_pages_despite_their_typography(fixture, quote):
    # Catches the checker failing correct quotes on curly quotes, dashes and non-breaking spaces, which would send
    # researchers round after round "fixing" them. The fixtures are text read_page returned from the real pages.
    assert check_quote(quote, (FIXTURES / fixture).read_text()) is None


def test_a_quote_verifies_across_the_spaces_html_extraction_adds():
    # Catches correct quotes failing where HTML extraction split a word or code tokens apart, as on a live run.
    page = 'print(\n page. extract_text (extraction_mode = "layout"))\nPyMuPD F and pypdfium generally outperformed others'
    assert check_quote('print(page.extract_text(extraction_mode="layout")) PyMuPDF and pypdfium generally', page) is None


# Quotes that failed on a live run (facts F75, F14, F15, F16, F85, F106, F101, F32, F86, F77, F141) though the
# sentence was on the page. The .txt fixtures are the text read_page saved; the .html ones are the pages' raw HTML.
@pytest.mark.parametrize("fixture, quote", [
    ("arxiv_2508_05635_excerpt.txt",
     "At its core lies GE-Base, an instruction-conditioned, multi-view video diffusion model trained on approximately "
     "3,000 hours of video-language paired data spanning over one million real-world robotic manipulation episodes "
     "from the AgiBot-World-Beta dataset."),
    ("arxiv_2511_02097_excerpt.txt",
     "contemporary architectures for capturing world dynamics can be broadly stratified along a methodological "
     "spectrum: implicit world modeling (e.g., LLMs, VLMs, and VLAs), latent dynamics modeling, and video generation "
     "paradigms, each targeting distinct representational granularities and predictive mechanisms."),
    ("arxiv_2511_02097_excerpt.txt",
     "A central point of contention concerns its generative capability, as illustrated by NVIDIA, who define world "
     "models as systems that learn environmental dynamics from multimodal data and generate videos capturing spatial "
     "and physical properties."),
    ("arxiv_2511_02097_excerpt.txt",
     "Indeed, many recent world models adopt video generation as their core mechanism, enabling the prediction of "
     "future states encompassing observations, actions, and environmental changes."),
    ("arxiv_2411_14499_excerpt.txt",
     "A key limitation is its causal reasoning ability (, ), restricting it to passively generating sequences without "
     "actively predicting how actions might alter events."),
    ("arxiv_2502_20694_excerpt.txt",
     "Previous video generation evaluation mainly uses single-number metric such as Frechet Video Distance (FVD) and "
     "CLIPSIM."),
])
def test_a_quote_verifies_without_the_citations_the_page_has(fixture, quote):
    # Catches correct quotes failing because the researcher left out "[81]" or "(Bu et al., 2025a)", or kept only
    # the brackets of a citation whose numbers the page split onto their own lines.
    assert check_quote(quote, (FIXTURES / fixture).read_text()) is None


def test_a_quote_that_keeps_the_citation_still_verifies():
    # Catches citations stripped from the page only, which would fail researchers who quote them.
    page = (FIXTURES / "arxiv_2511_02097_excerpt.txt").read_text()
    assert check_quote("as illustrated by NVIDIA [81], who define world models as systems that learn", page) is None


def test_a_quote_verifies_across_a_pdf_line_break_hyphen():
    # Catches "no language encoder" failing on pypdf's "no lan-\nguage encoder".
    quote = ("On LIBERO, a 0.09B probe with no language encoder scores at or near reported SOTA, and most reported "
             "gains are not provably statistically significant.")
    assert check_quote(quote, (FIXTURES / "arxiv_2606_04233_pdf_excerpt.txt").read_text()) is None


@pytest.mark.parametrize("fixture, quote", [
    ("arxiv_2505_12705_excerpt.html",
     "In simulation, we apply DreamGen to the RoboCasa benchmark [20], scaling synthetic data up to 333× relative to "
     "the original human demonstrations. This yields log-linear improvements in policy performance as the number of "
     "neural trajectories increases (Figure 4)."),
    ("arxiv_2503_06800_excerpt.html",
     "Our findings reveal major shortcomings, with even the best model achieving only 22% joint performance (i.e., "
     "high semantic and physical commonsense adherence) on the hard subset of VideoPhy-2."),
    ("openai_rubiks_cube_excerpt.html",
     "Our robot still hasn’t perfected its technique though, as it solves the Rubik’s Cube 60% of the time (and only "
     "20% of the time for a maximally difficult scramble)."),
])
def test_a_quote_verifies_against_html_math_and_hidden_link_text(fixture, quote):
    # Catches arXiv math read as "333 × \times" (the TeX source after the symbol), and OpenAI links read as
    # "difficult⁠(opens in a new window)": text no reader of the page sees.
    assert check_quote(quote, html_text((FIXTURES / fixture).read_text())) is None


@pytest.mark.parametrize("quote", [
    # A word changed between citations.
    "contemporary architectures for capturing world dynamics can be broadly stratified along a methodological "
    "spectrum: implicit world modeling (e.g., LLMs, VLMs, and VLAs), explicit dynamics modeling",
    # Two passages joined with "...": what the researcher prompt forbids.
    "A central point of contention concerns its generative capability, ... who define world models as systems",
])
def test_ignoring_citations_still_fails_a_changed_or_cut_quote(quote):
    # Catches citation stripping loose enough to verify a quote the page doesn't say.
    assert check_quote(quote, (FIXTURES / "arxiv_2511_02097_excerpt.txt").read_text()) is not None


def test_a_short_quote_is_recorded_unverified(tmp_path, pages):
    # Catches "MIT" verifying against any license page.
    notebook = new_notebook(tmp_path)
    assert record(notebook, "https://a.test", PAGE_A, pages, "written in Rust") == (
        "F1 recorded, NOT verified: the quote has 3 words; quote at least 6. "
        "Fix the quote with record_fact again, or leave it.")


def test_a_long_quote_with_few_spaces_verifies():
    # Catches the word count refusing a verbatim requirements line or JSON field, which is as specific as a sentence.
    page = "requirements.txt\npdfminer.six==20260107\nPillow>=12.2.0\npypdfium2>=5.9.0\n"
    assert check_quote("pdfminer.six==20260107\nPillow>=12.2.0\npypdfium2>=5.9.0", page) is None


def test_researchers_recording_at_once_get_unique_consecutive_ids(tmp_path, pages):
    # Catches two researchers given the same id, or interleaved half-lines in facts.jsonl.
    notebook = new_notebook(tmp_path)
    pages["https://a.test"] = PAGE_A
    for i in range(5):
        notebook.tools_for(f"main/{i}")["fetch_page"]("https://a.test")
    start = threading.Barrier(5)

    def work(i):
        start.wait(TIMEOUT)
        notebook.tools_for(f"main/{i}")["record_fact"](f"claim {i}", "https://a.test", QUOTE_A)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    lines = (notebook.dir / "facts.jsonl").read_text().splitlines()
    assert len(lines) == 5 and sorted(json.loads(line)["id"] for line in lines) == ["F1", "F2", "F3", "F4", "F5"]


def test_a_new_notebook_on_the_same_folder_continues_the_ids(tmp_path, pages):
    # Catches ids kept in memory, which a second Notebook (a resumed run, a grader) would restart at F1.
    notebook = new_notebook(tmp_path)
    record(notebook, "https://a.test", PAGE_A, pages, QUOTE_A)
    again = Notebook(notebook.dir)
    assert again.tools_for("main/2")["record_fact"]("claim", "https://a.test", QUOTE_A).startswith("F2 ")


# ---------------------------------------------------------------- draft and report


def test_write_section_replaces_by_title_and_keeps_the_others(tmp_path):
    # Catches a rewrite that appends a duplicate section, or loses the ones around it.
    notebook = new_notebook(tmp_path)
    notebook.write_section("Summary", "first")
    notebook.write_section("Details", "details")
    assert notebook.write_section("summary", "second") == "Saved 'summary' (1 of 2 sections). Cites no facts."
    assert notebook.read_draft() == "## summary\n\nsecond\n\n## Details\n\ndetails\n"


def test_write_section_refuses_a_second_level_heading_inside(tmp_path):
    # Catches content that silently splits into two sections, one with a title the lead never chose.
    notebook = new_notebook(tmp_path)
    with pytest.raises(ValueError, match="use ### for headings"):
        notebook.write_section("Summary", "intro\n## Oops\ntext")
    assert notebook.read_draft() == ""


def test_write_section_reports_citation_problems_when_it_saves(tmp_path, pages):
    # Catches mistakes that only surface in the finished report, too late for the lead to fix.
    notebook = new_notebook(tmp_path)
    record(notebook, "https://a.test", PAGE_A, pages, QUOTE_A)
    record(notebook, "https://a.test", PAGE_A, pages, "written in Rust")
    reply = notebook.write_section("Summary", "Alpha is Rust [F1][F2], see https://a.test and [F99].")
    assert reply == ("Saved 'Summary' (1 of 1 sections). Cites F1, F2, F99. Unknown: F99. Unverified: F2. "
                     "Raw URLs: 1 — cite facts instead.")


def test_render_report_numbers_sources_by_page_and_lists_their_quotes(tmp_path, pages):
    # Catches numbering by fact instead of page, the wrong order, lost quotes or dates, and a dropped ⚠.
    notebook = new_notebook(tmp_path)
    record(notebook, "https://a.test", PAGE_A, pages, QUOTE_A, date="2025-03-14")                        # F1
    record(notebook, "https://b.test", PAGE_B, pages, QUOTE_B)                                            # F2
    record(notebook, "https://a.test/", PAGE_A, pages, "It installs packages ten times faster than pip.")  # F3
    record(notebook, "https://b.test", PAGE_B, pages, "stable since two", by="main/2")                    # F4
    notebook.write_section("Summary", "- Beta uses TOML [F2].\n- Alpha is Rust and fast [F1][F3].\n- Gamma [F99].")
    notebook.write_section("Details", "A paragraph with no citation.\n\nAnother, cited [F2, F4].")
    notebook.write_section("Caveats", "Nothing found on Gamma [F4].")
    report = render_report(notebook.dir)
    assert report == """\
## Summary

- Beta uses TOML [1].
- Alpha is Rust and fast [2].
- Gamma [F99?].

## Details

A paragraph with no citation.

Another, cited [1].

## Caveats

Nothing found on Gamma [1⚠].

## Sources

1. https://b.test
   - "Beta stores its lockfile as TOML in the project root."
   - ⚠ "stable since two" (the quote has 3 words; quote at least 6)
2. https://a.test (2025-03-14)
   - "Alpha is a package manager written in Rust."
   - "It installs packages ten times faster than pip."

---
Facts: 4 recorded, 3 verified, 4 cited, 0 unused. Unverified (⚠): F4. Unknown ids: F99. \
Uncited paragraphs outside Caveats: 1.
"""


def test_render_report_lists_facts_no_section_cites(tmp_path, pages):
    # Catches a run that stopped before drafting leaving an empty report, though its facts are on disk.
    notebook = new_notebook(tmp_path)
    record(notebook, "https://a.test", PAGE_A, pages, QUOTE_A, claim="Alpha is Rust", by="main/1")
    record(notebook, "https://b.test", PAGE_B, pages, "stable since two", claim="Beta is stable", by="main/2")
    report = render_report(notebook.dir)
    assert report.startswith("""\
## Uncited facts

Facts the researchers recorded that no section cites.

### main/1
- Alpha is Rust [1] ✓

### main/2
- Beta is stable [2] ⚠

## Sources
""")
    assert report.endswith("Facts: 2 recorded, 1 verified, 0 cited, 2 unused. Unverified (⚠): F2.\n")
    assert summary_of(report).startswith("## Uncited facts\n")


def test_the_summary_is_the_summary_section_else_the_first(tmp_path):
    notebook = new_notebook(tmp_path)
    notebook.write_section("Findings", "found things")
    assert summary_of(render_report(notebook.dir)).startswith("## Findings\nfound things")
    notebook.write_section("Summary", "the answer")
    assert summary_of(render_report(notebook.dir)) == "## Summary\nthe answer"


# ---------------------------------------------------------------- results and stopping


def test_a_researcher_stopped_by_its_budget_keeps_its_facts(tmp_path, pages):
    # Catches facts lost with a researcher that ran out of money mid-task.
    notebook = new_notebook(tmp_path)
    pages["https://a.test"] = PAGE_A
    client = TaskClient({"find alpha": researcher_script("https://a.test", QUOTE_A, "Alpha is Rust", cost=1.0)})
    run = researcher_profile(notebook, client=client).start("find alpha", Budget(0.10, name="main/1"))
    outcome = run.outcome(TIMEOUT)
    assert (outcome.status, outcome.reason) == ("stopped", "budget spent")
    assert [f.claim for f in read_facts(notebook.dir)] == ["Alpha is Rust"]
    assert outcome.text.startswith('Facts recorded (1):\nF1 ✓ Alpha is Rust\n    "Alpha is a package manager')


def test_end_to_end_a_lead_and_two_researchers_leave_a_complete_run_folder(tmp_path, pages):
    # Catches any break in the chain: spawn → facts on disk → results with each researcher's own facts → sections
    # → report.md, agents/ and budget.json, with the tree's spend adding up.
    pages.update({"https://a.test": PAGE_A, "https://b.test": PAGE_B})
    client = TaskClient({
        "the brief": [
            response(tool_calls=[call("spawn", profile="researcher", task="find alpha", background=True),
                                 call("spawn", profile="researcher", task="find beta", background=True)]),
            response(tool_calls=[call("subagent", run_id=1, wait=600), call("subagent", run_id=2, wait=600)]),
            response(tool_calls=[call("write_section", title="Summary", content="- Alpha and Beta [F1][F2]."),
                                 call("write_section", title="Caveats", content="None.")]),
            response(tool_calls=[call("facts", unused_only=True)]),
            response("The report is done."),
        ],
        "find alpha": researcher_script("https://a.test", QUOTE_A, "Alpha is Rust"),
        "find beta": researcher_script("https://b.test", QUOTE_B, "Beta uses TOML"),
    })
    run = research_profile(client=client, runs_dir=tmp_path).start("the brief", Budget(1.0))
    outcome = run.outcome(TIMEOUT)
    assert outcome.status == "done", outcome.text
    dir = outcome_dir(outcome.text)

    results = [m["content"] for m in run.agent.messages if m["role"] == "tool" and "<subagent-result" in m["content"]]
    assert "Alpha is Rust" in results[0] and QUOTE_A in results[0] and "https://a.test" in results[0]
    assert "Beta" not in results[0] and "covered Alpha is Rust" in results[0]
    assert "Beta uses TOML" in results[1] and "Alpha" not in results[1]

    report = (dir / "report.md").read_text()
    assert "- Alpha and Beta [1][2]." in report and "https://a.test" in report and "https://b.test" in report
    assert outcome.text == (f"Report: {dir / 'report.md'}\n\n## Summary\n- Alpha and Beta [1][2].\n\n"
                            "Facts: 2 recorded, 2 verified, 2 cited, 0 unused.")
    assert sorted(p.name for p in (dir / "agents").iterdir()) == ["main-1.json", "main-2.json", "main.json"]
    budget = json.loads((dir / "budget.json").read_text())
    assert sum(row["cost"] for row in budget["breakdown"]) == pytest.approx(budget["spent"]) and budget["spent"] > 0
    assert (dir / "brief.md").read_text() == "the brief"


def test_researchers_that_spend_the_whole_budget_still_leave_a_report(tmp_path, pages):
    # Catches starvation losing everything: child limits aren't reserved, so researchers can empty the lead's pool
    # before it writes a word.
    pages.update({"https://a.test": PAGE_A, "https://b.test": PAGE_B})
    client = TaskClient({
        "the brief": [
            response(tool_calls=[call("spawn", profile="researcher", task="find alpha", background=True),
                                 call("spawn", profile="researcher", task="find beta", background=True)]),
            response(tool_calls=[call("subagent", run_id=1, wait=600), call("subagent", run_id=2, wait=600)]),
        ],
        "find alpha": researcher_script("https://a.test", QUOTE_A, "Alpha is Rust", cost=0.15),
        "find beta": researcher_script("https://b.test", QUOTE_B, "Beta uses TOML", cost=0.15),
    })
    outcome = research_profile(client=client, runs_dir=tmp_path).start("the brief", Budget(0.25)).outcome(TIMEOUT)
    assert (outcome.status, outcome.reason) == ("stopped", "budget spent")
    report = (outcome_dir(outcome.text) / "report.md").read_text()
    for text in (report, outcome.text):
        assert "## Uncited facts" in text and "Alpha is Rust [" in text and "Beta uses TOML [" in text


def test_a_lead_stopped_after_drafting_hands_back_its_summary(tmp_path, pages):
    # Catches a stopped run's result without the report, or with [n] numbers that don't match report.md.
    notebook = new_notebook(tmp_path)
    record(notebook, "https://a.test", PAGE_A, pages, QUOTE_A)
    record(notebook, "https://b.test", PAGE_B, pages, QUOTE_B)
    client = TaskClient({"the brief": [response(
        tool_calls=[call("write_section", title="Summary", content="- Beta uses TOML [F2], Alpha is Rust [F1]."),
                    call("write_section", title="Details", content="More on Beta [F2].")],
        usage=usage(cost=1.0))]})
    lead = research_lead(notebook, Budget(0.5), client=client)
    outcome = Run.start(lead, "the brief", report=lambda: research_result(lead, notebook)).outcome(TIMEOUT)
    assert (outcome.status, outcome.reason) == ("stopped", "budget spent")
    assert "## Summary\n- Beta uses TOML [1], Alpha is Rust [2].\n\nFacts: 2 recorded" in outcome.text
    assert "- Beta uses TOML [1], Alpha is Rust [2]." in (notebook.dir / "report.md").read_text()


def test_a_researcher_the_lead_never_collected_is_stopped_before_the_report(tmp_path, pages):
    # Catches a report rendered while a researcher still writes: facts missing from it, or counts that don't match
    # facts.jsonl. A cancelled researcher finishes the tool calls of the reply it's on, so its last fact lands.
    slow_started = threading.Event()

    def slow_page():
        slow_started.set()
        threading.Event().wait(0.3)  # the lead finishes, and cancels this researcher, meanwhile
        return PAGE_B

    pages.update({"https://a.test": PAGE_A, "https://slow.test": slow_page})
    client = TaskClient({
        "the brief": [
            response(tool_calls=[call("spawn", profile="researcher", task="find both", background=True)]),
            lambda: slow_started.wait(TIMEOUT) and response("The report is done."),
        ],
        "find both": [
            response(tool_calls=[call("fetch_page", url="https://a.test"),
                                 call("record_fact", claim="Alpha is Rust", url="https://a.test", quote=QUOTE_A)]),
            response(tool_calls=[call("fetch_page", url="https://slow.test"),
                                 call("record_fact", claim="Beta uses TOML", url="https://slow.test", quote=QUOTE_B)]),
        ],
    })
    run = research_profile(client=client, runs_dir=tmp_path).start("the brief", Budget(1.0))
    outcome = run.outcome(TIMEOUT)
    assert outcome.status == "done"
    researcher = run.agent.subagents.runs[1][1].outcome(0)
    assert (researcher.status, researcher.reason) == ("stopped", "cancelled")
    report = (outcome_dir(outcome.text) / "report.md").read_text()
    assert "- Alpha is Rust [1] ✓" in report and "- Beta uses TOML [2] ✓" in report
    assert report.endswith("Facts: 2 recorded, 2 verified, 0 cited, 2 unused.\n")


def test_one_research_profile_gives_each_run_its_own_folder(tmp_path):
    # Catches a notebook built once per profile, which two app chats (or two spawns) would then share.
    client = TaskClient({"brief one": [response("done")], "brief two": [response("done")]})
    profile = research_profile(client=client, runs_dir=tmp_path)
    runs = [profile.start("brief one", Budget(1.0)), profile.start("brief two", Budget(1.0))]
    dirs = [outcome_dir(run.outcome(TIMEOUT).text) for run in runs]
    assert dirs[0] != dirs[1] and [(d / "brief.md").read_text() for d in dirs] == ["brief one", "brief two"]
