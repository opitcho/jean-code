"""Research runs: researchers record checked facts into a run folder, and a lead writes a cited report from them.

Researchers search the web and record *facts*: one claim with a quote, where the quote is checked against the page
the researcher fetched. The lead has no web tools. It writes the report's sections from those facts, citing them as
`[F12]`, and `render_report` turns the ids into numbered sources. The folder is the run's only state (see
`Notebook`), so the lead, the renderer and anything grading a run later read the same files.

Who uses what:
    research_profile     the coding agent's `research` subagent: a fresh folder and a lead per run
    researcher_profile   the lead's `researcher` subagent: researchers writing into the lead's notebook
    Notebook             the tools that write to a folder: fetch_page and record_fact (researchers), facts,
                         write_section and read_draft (the lead)
    render_report        the folder → report.md, at the end of every run; an eval can call it on any folder
    read_facts           the facts in a folder, for the renderer, the lead's `facts` and each researcher's result
"""

import json
import os
import re
import secrets
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pages
import tools
from agent import Agent, DEFAULT_MODEL
from config import RESEARCH_DIR
from llm import ChatClient, tool
from prompts import RESEARCH_LEAD_PROMPT, RESEARCHER_PROMPT
from subagents import Profile, Run, agent_tree, last_reply
from usage import Budget

MIN_QUOTE_WORDS = 6  # shorter quotes ("MIT", "10x faster") match too many pages to check anything…
MIN_QUOTE_CHARS = 40  # …unless they're long without spaces, like a JSON field or a requirements line
CITATION = re.compile(r"(?:\[F\d+(?:\s*,\s*F\d+)*\])+")  # [F12], [F12, F13], [F12][F13]: one match each
RAW_URL = re.compile(r"https?://\S+")
SUMMARY_CHARS = 3000  # the most of the report's Summary a result hands back

RESEARCH_DESCRIPTION = (
    "researches a scoped question on the web and writes a cited report file; give it the brief from the research "
    "skill and run it in the background")
RESEARCHER_DESCRIPTION = (
    "searches the web for one sub-question and records facts, with quotes checked against the pages, into this "
    "run's notebook; give it the sub-question, what's in and out of scope, what the others cover and what facts "
    "to return")

# Curly quotes → straight, hyphens and dashes dropped. NFKC already turns non-breaking spaces into spaces and "…"
# into "...".
PUNCTUATION = str.maketrans({c: "'" for c in "‘’‚‛′"} | {c: '"' for c in "“”„‟″"} | {c: None for c in "-‐‑‒–—―−"})
# In-text citations: "[81]", "[5, 6, 7]", "(338, 35)" and "(Bu et al., 2025a)". Researchers leave them out of quotes.
REFERENCE_MARK = re.compile(r"\[[\d\s,;]*\]|\([\d\s,;]*\)|\(\s*[A-Z][^()]*?\b(?:1[89]|20)\d\d[a-z]?\s*\)")


@dataclass(frozen=True)
class Fact:
    """One line of facts.jsonl: a claim, the quote that supports it, and whether the quote was found on the page."""

    id: str
    claim: str
    url: str
    quote: str
    verified: bool
    by: str  # the budget path of the researcher that recorded it
    problem: str | None = None  # why it isn't verified
    date: str | None = None  # the page's publication date, as the page gives it
    time: str = ""  # when it was recorded


class Notebook:
    """A research run's folder, and the tools that write to it. The files are the only state: nothing is cached, so a
    new Notebook on the same folder carries on where the last one stopped.

        run.json      {"path", "model", "started"}
        brief.md      the task, as given
        pages/        P1.txt, …: the parts of each fetched page that a model saw
          index.json  {page key: {"file", "length", "windows": [[start, end]], "by": [paths]}}
        facts.jsonl   one Fact per line, only ever appended to
        draft.md      the lead's sections, as "## Title" blocks, in the order first written
        report.md     rendered from the above at the end of the run
        agents/       one <path>.json per agent (messages and usage), budget.json: written at the end of the run

    Thread-safe: one lock guards ids, facts.jsonl, the page files, index.json and draft.md. It's never held during a
    download. index.json and draft.md are replaced whole (temp file, then os.replace), so a reader never sees half a
    write.
    """

    def __init__(self, dir: str | Path):
        self.dir = Path(dir)
        self._lock = threading.Lock()

    @classmethod
    def create(cls, runs_dir: str | Path, brief: str, path: str, model: str) -> "Notebook":
        """A new folder `<runs_dir>/<YYYYmmdd-HHMMSS>-<6 hex>`, with the brief and the run's metadata."""
        dir = Path(runs_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        (dir / "pages").mkdir(parents=True)  # exist_ok=False: two runs never share a folder
        (dir / "brief.md").write_text(brief)
        write_json(dir / "run.json", {"path": path, "model": model, "started": now()})
        return cls(dir)

    # ---------------------------------------------------------------- researcher tools

    def tools_for(self, by: str) -> dict[str, Callable]:
        """The researcher tools, bound to this folder and to `by`, the researcher's budget path."""

        @tool
        def fetch_page(url: str, max_chars: int = 8000, start: int = 0) -> str:
            """Fetch a web page or PDF and return part of its readable text. What you read here is what quotes are
            checked against.

            Args:
                url: Full URL of the page, including http:// or https://.
                max_chars: Maximum number of characters of text to return.
                start: Character to start at. A long page ends with a line saying where the part shown ends; fetch
                    again with that position as `start` to read on.

            Returns:
                The page's title and visible text (or a PDF's text), from `start`, cut to max_chars.
            """
            return self._fetch_page(by, url, max_chars, start)

        @tool
        def record_fact(claim: str, url: str, quote: str, date: str | None = None) -> str:
            """Record one fact for the report, with the passage of a fetched page that supports it.

            Args:
                claim: The fact, in one sentence of your own.
                url: The URL you fetched the page from with fetch_page.
                quote: At least one full sentence copied verbatim, without cuts or ellipses, from the text
                    fetch_page returned for that URL. It is checked against that text.
                date: The page's publication or update date, as the page gives it, if it shows one.

            Returns:
                The fact's id and whether its quote was found on the page.
            """
            return self._record_fact(by, claim, url, quote, date)

        return {"fetch_page": fetch_page, "record_fact": record_fact}

    def _fetch_page(self, by: str, url: str, max_chars: int, start: int) -> str:
        text = pages.read_page(url)  # the download, outside the lock
        start, end = pages.window(text, start, max_chars)
        with self._lock:
            index = self._index()
            entry = index.setdefault(page_key(url), {"file": f"P{len(index) + 1}.txt", "length": 0,
                                                     "windows": [], "by": []})
            entry["length"] = len(text)
            if end > start and [start, end] not in entry["windows"]:
                with open(self.dir / "pages" / entry["file"], "a") as f:
                    f.write(text[start:end] + "\n\n")
                entry["windows"].append([start, end])
            if by not in entry["by"]:
                entry["by"].append(by)
            write_json(self.dir / "pages" / "index.json", index)
        return pages.window_text(text, start, end)

    def _record_fact(self, by: str, claim: str, url: str, quote: str, date: str | None) -> str:
        entry = self._index().get(page_key(url))
        if entry is None:
            return f"Not recorded: {url} wasn't fetched in this run; fetch_page it first."
        problem = check_quote(quote, (self.dir / "pages" / entry["file"]).read_text())
        with self._lock:
            id = f"F{len(read_facts(self.dir)) + 1}"  # counted from the file, so a new Notebook carries on
            fact = Fact(id, claim, url, quote, problem is None, by, problem, date or None, now())
            with open(self.dir / "facts.jsonl", "a") as f:
                f.write(json.dumps(asdict(fact)) + "\n")
        if problem:
            return f"{id} recorded, NOT verified: {problem}. Fix the quote with record_fact again, or leave it."
        return f"{id} recorded (quote verified)"

    def _index(self) -> dict:
        path = self.dir / "pages" / "index.json"
        return json.loads(path.read_text()) if path.exists() else {}

    # ---------------------------------------------------------------- lead tools

    @tool
    def facts(self, ids: list[str] | None = None, unused_only: bool = False) -> str:
        """List the facts the researchers have recorded so far, with their quotes and sources.

        Args:
            ids: Only these facts, e.g. ["F3", "F12"]. All of them if omitted.
            unused_only: Only facts that no section of the draft cites yet.

        Returns:
            One entry per fact: its id, ✓ (quote verified) or ⚠ (not verified, and why), the claim, then the quote
            and URL.
        """
        facts = read_facts(self.dir)
        if ids:
            wanted = set(ids)
            facts = [f for f in facts if f.id in wanted]
        if unused_only:
            cited = set(cited_ids(self.read_draft()))
            facts = [f for f in facts if f.id not in cited]
        return format_facts(facts) or "No facts match."

    @tool
    def write_section(self, title: str, content: str) -> str:
        """Write one section of the report's draft, replacing the section with the same title if there is one.

        Args:
            title: The section's title, e.g. "Summary" or "Caveats", without "##".
            content: The section's text in Markdown, citing facts as [F12] or [F12, F13]. Use ### for headings
                inside it; a line starting with "## " would start another section.

        Returns:
            What was saved, which facts it cites, and any citation problems: unknown ids, unverified facts, raw URLs.
        """
        title = title.strip().lstrip("#").strip()
        if not title:
            raise ValueError("the title is empty")
        if re.search(r"^## ", content, re.MULTILINE):
            raise ValueError("the content has a line starting with '## ', which would split the section in two; "
                             "use ### for headings inside a section")
        with self._lock:
            sections = parse_sections(self.read_draft())
            titles = [t.casefold() for t, _ in sections]
            if title.casefold() in titles:
                sections[titles.index(title.casefold())] = (title, content.strip())
            else:
                sections.append((title, content.strip()))
            write_text(self.dir / "draft.md", "\n\n".join(f"## {t}\n\n{body}" for t, body in sections) + "\n")

        facts = {f.id: f for f in read_facts(self.dir)}
        cited = cited_ids(content)
        notes = [f"Cites {', '.join(cited)}." if cited else "Cites no facts."]
        if unknown := [id for id in cited if id not in facts]:
            notes.append(f"Unknown: {', '.join(unknown)}.")
        if unverified := [id for id in cited if id in facts and not facts[id].verified]:
            notes.append(f"Unverified: {', '.join(unverified)}.")
        if urls := len(RAW_URL.findall(content)):
            notes.append(f"Raw URLs: {urls} — cite facts instead.")
        position = [t.casefold() for t, _ in sections].index(title.casefold()) + 1
        return f"Saved '{title}' ({position} of {len(sections)} sections). {' '.join(notes)}"

    @tool
    def read_draft(self) -> str:
        """Read the report's draft as it is now: every section written so far.

        Returns:
            The draft in Markdown, one "## Title" block per section, or an empty string if nothing is written yet.
        """
        path = self.dir / "draft.md"
        return path.read_text() if path.exists() else ""


# ---------------------------------------------------------------- reading a folder


def read_facts(dir: str | Path) -> list[Fact]:
    """Every fact recorded in the folder, in id order."""
    path = Path(dir) / "facts.jsonl"
    if not path.exists():
        return []
    return [Fact(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def format_facts(facts: list[Fact]) -> str:
    """Facts as the lead reads them: `F12 ✓ claim`, then the quote, URL and date."""
    entries = []
    for f in facts:
        mark = "✓" if f.verified else "⚠"
        head = f"{f.id} {mark} {f.claim}" + ("" if f.verified else f"  (not verified: {f.problem})")
        entries.append(f'{head}\n    "{f.quote}" — {f.url}' + (f" ({f.date})" if f.date else ""))
    return "\n".join(entries)


def render_report(dir: str | Path) -> str:
    """The report for a folder: the draft with [F12] replaced by source numbers, the facts no section cites, the
    sources with each quote listed under its page, and a footer of counts. Works with an empty or partial draft.

    Sources are numbered by page, in order of first citation, so two facts from one page share a number. A citation
    of an unverified fact is marked [3⚠], so a reader of the Summary alone sees it. An unknown id stays as [F99?].
    """
    dir = Path(dir)
    facts = {f.id: f for f in read_facts(dir)}
    draft = (dir / "draft.md").read_text() if (dir / "draft.md").exists() else ""
    sources: dict[str, list[Fact]] = {}  # page key → the facts numbered under it, in order

    def number(fact: Fact) -> int:
        listed = sources.setdefault(page_key(fact.url), [])
        if fact not in listed:
            listed.append(fact)
        return list(sources).index(page_key(fact.url)) + 1

    def cite(match: re.Match) -> str:
        unverified: dict[str, bool] = {}  # source number (or "F99?") → every fact cited from it is unverified
        for id in re.findall(r"F\d+", match.group(0)):
            fact = facts.get(id)
            label = f"{id}?" if fact is None else str(number(fact))
            unverified[label] = unverified.get(label, True) and fact is not None and not fact.verified
        return "".join(f"[{label}{'⚠' if bad else ''}]" for label, bad in unverified.items())

    body = CITATION.sub(cite, draft).strip()
    parts = [body] if body else []

    cited = set(cited_ids(draft))
    uncited = [f for f in facts.values() if f.id not in cited]
    if uncited:
        lines = ["## Uncited facts", "", "Facts the researchers recorded that no section cites."]
        for by in dict.fromkeys(f.by for f in uncited):
            lines += ["", f"### {by}"]
            lines += [f"- {f.claim} [{number(f)}] {'✓' if f.verified else '⚠'}" for f in uncited if f.by == by]
        parts.append("\n".join(lines))

    if sources:
        lines = ["## Sources", ""]
        for n, listed in enumerate(sources.values(), 1):
            date = next((f.date for f in listed if f.date), None)
            lines.append(f"{n}. {listed[0].url}" + (f" ({date})" if date else ""))
            for f in listed:
                lines.append(f'   - "{f.quote}"' if f.verified else f'   - ⚠ "{f.quote}" ({f.problem})')
        parts.append("\n".join(lines))

    parts.append(f"---\n{footer(dir)}")
    return "\n\n".join(parts) + "\n"


def footer(dir: str | Path) -> str:
    """Counts that show how far to trust a report: facts recorded, verified, cited and unused, then any citation
    problems in the draft."""
    dir = Path(dir)
    facts = read_facts(dir)
    draft = (dir / "draft.md").read_text() if (dir / "draft.md").exists() else ""
    cited = set(cited_ids(draft))
    known = {f.id for f in facts}
    line = (f"Facts: {len(facts)} recorded, {sum(f.verified for f in facts)} verified, "
            f"{len(cited & known)} cited, {len(known - cited)} unused.")
    if unverified := [f.id for f in facts if not f.verified]:
        line += f" Unverified (⚠): {', '.join(unverified)}."
    if unknown := [id for id in cited_ids(draft) if id not in known]:
        line += f" Unknown ids: {', '.join(unknown)}."
    if urls := len(RAW_URL.findall(draft)):
        line += f" Raw URLs in the draft: {urls}."
    if paragraphs := uncited_paragraphs(draft):
        line += f" Uncited paragraphs outside Caveats: {paragraphs}."
    return line


def uncited_paragraphs(draft: str) -> int:
    """Paragraphs and list items with no [F…], outside the Caveats section, where listing gaps needs no source."""
    count = 0
    for title, body in parse_sections(draft):
        if title.casefold() == "caveats":
            continue
        for block in re.split(r"\n\s*\n", body):
            lines = [line for line in block.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            items = re.split(r"\n(?=\s*(?:[-*+]|\d+\.)\s)", "\n".join(lines))  # one unit per list item
            count += sum(1 for item in items if item.strip() and not CITATION.search(item))
    return count


def parse_sections(markdown: str) -> list[tuple[str, str]]:
    """`## Title` blocks as (title, body), in order. Text before the first one is dropped."""
    sections: list[tuple[str, str]] = []
    for block in re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]:
        title, _, body = block.partition("\n")
        sections.append((title.strip(), body.strip()))
    return sections


def cited_ids(markdown: str) -> list[str]:
    """The fact ids cited in `markdown`, in order of first use."""
    ids = (id for match in CITATION.finditer(markdown) for id in re.findall(r"F\d+", match.group(0)))
    return list(dict.fromkeys(ids))


def page_key(url: str) -> str:
    """The URL a page is filed under: no #fragment and no trailing /, so both spellings find the same page."""
    return url.strip().split("#", 1)[0].rstrip("/")


def normalize(text: str) -> str:
    """Text as quotes are compared: NFKC, straight quotes, case-folded, and without whitespace, hyphens, dashes,
    invisible format characters (U+2060 word joiners, soft hyphens) or in-text citations. Without this, correct
    quotes from real pages fail on typography, on how HTML extraction splits inline elements onto their own lines
    ("splitting\n, merging", "page. extract_text (", "PyMuPD F"), on PDF line breaks inside words ("lan-\nguage"), and
    on the citations researchers leave out ("the AgiBot-World-Beta dataset (Bu et al., 2025a)."). Stripping from both
    sides keeps quotes that do include a citation verifying too."""
    text = unicodedata.normalize("NFKC", text).translate(PUNCTUATION)
    text = REFERENCE_MARK.sub("", "".join(c for c in text if unicodedata.category(c) != "Cf"))
    return "".join(text.casefold().split())


def check_quote(quote: str, page: str) -> str | None:
    """None if the quote is on the page, else why it isn't verified."""
    quote = quote.strip().strip("\"'")
    words = len(quote.split())
    if words < MIN_QUOTE_WORDS and len(quote) < MIN_QUOTE_CHARS:
        return f"the quote has {words} words; quote at least {MIN_QUOTE_WORDS}"
    if normalize(quote) not in normalize(page):
        return "the quote isn't on the page (as read so far)"
    return None


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data) -> None:
    write_text(path, json.dumps(data, indent=2))


def write_text(path: Path, text: str) -> None:
    """Replace the file whole: a reader sees the old content or the new, never half of it."""
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text)
    os.replace(temp, path)


# ---------------------------------------------------------------- agents and profiles


def researcher(notebook: Notebook, budget: Budget, model: str = DEFAULT_MODEL,
               client: ChatClient | None = None) -> Agent:
    return Agent(model, system_prompt=RESEARCHER_PROMPT,
                 tools={"web_search": tools.web_search, **notebook.tools_for(budget.path)},
                 budget=budget, max_tool_rounds=15, compact_at=None, client=client)


def research_lead(notebook: Notebook, budget: Budget, model: str = DEFAULT_MODEL,
                  client: ChatClient | None = None) -> Agent:
    # max_tool_rounds=40: the default 10 runs out on 3 spawns + 3 collects + 4 sections.
    return Agent(model, system_prompt=RESEARCH_LEAD_PROMPT,
                 tools={"facts": notebook.facts, "write_section": notebook.write_section,
                        "read_draft": notebook.read_draft},
                 subagent_profiles={"researcher": researcher_profile(notebook, model, client)},
                 budget=budget, max_tool_rounds=40, compact_at=None, client=client)


def researcher_profile(notebook: Notebook, model: str = DEFAULT_MODEL, client: ChatClient | None = None) -> Profile:
    """The lead's profile: researchers that all write into this lead's notebook."""
    def start(task: str, budget: Budget) -> Run:
        agent = researcher(notebook, budget, model, client)
        return Run.start(agent, task, report=lambda: researcher_result(agent, notebook))
    return Profile(RESEARCHER_DESCRIPTION, start, max_cost=0.10, max_cost_cap=0.25)


def research_profile(model: str = DEFAULT_MODEL, client: ChatClient | None = None,
                     runs_dir: str | Path = RESEARCH_DIR) -> Profile:
    """The caller's profile: each run gets a fresh folder under `runs_dir` and a lead bound to it. Outside an agent,
    e.g. for an eval, a run is `research_profile(runs_dir=…).start(brief, Budget(…))`."""
    def start(task: str, budget: Budget) -> Run:
        notebook = Notebook.create(runs_dir, brief=task, path=budget.path, model=model)
        lead = research_lead(notebook, budget, model, client)
        return Run.start(lead, task, report=lambda: research_result(lead, notebook))
    return Profile(RESEARCH_DESCRIPTION, start, max_cost=0.50, max_cost_cap=2.0)


def researcher_result(agent: Agent, notebook: Notebook) -> str:
    """What the lead gets back from a researcher: every fact it recorded, in full, then its last reply as the note."""
    mine = [f for f in read_facts(notebook.dir) if f.by == agent.budget.path]
    return f"Facts recorded ({len(mine)}):\n{format_facts(mine)}\n\nNote:\n{last_reply(agent) or '(none)'}"


def research_result(lead: Agent, notebook: Notebook) -> str:
    """What the caller gets back: the report's path, its Summary and the counts. The only code that finishes a run
    folder, so every run, stopped or not, leaves report.md, agents/ and budget.json."""
    lead.subagents.close()  # cancel and wait: a cancelled researcher still runs its last reply's tool calls,
                            # so its last record_facts land before the report counts them
    report = render_report(notebook.dir)
    write_text(notebook.dir / "report.md", report)
    (notebook.dir / "agents").mkdir(exist_ok=True)
    for path, agent in agent_tree(lead).items():
        write_json(notebook.dir / "agents" / f"{path.replace('/', '-')}.json",
                   {"path": path, "task": agent.budget.task, "messages": agent.messages, "usage": agent.usage})
    write_json(notebook.dir / "budget.json", {"limit": lead.budget.limit, "spent": lead.budget.spent,
                                              "breakdown": lead.budget.breakdown()})
    return f"Report: {notebook.dir / 'report.md'}\n\n{summary_of(report)}\n\n{footer(notebook.dir)}"


def summary_of(report: str) -> str:
    """The rendered report's Summary section, else its first section (with no draft, the uncited facts)."""
    sections = parse_sections(report.rsplit("\n---\n", 1)[0])  # without the footer
    if not sections:
        return "(the report is empty: no facts and no draft)"
    title, body = next((s for s in sections if s[0].casefold() == "summary"), sections[0])
    if len(body) > SUMMARY_CHARS:
        body = body[:SUMMARY_CHARS] + "\n… (cut; the rest is in the report)"
    return f"## {title}\n{body}"
