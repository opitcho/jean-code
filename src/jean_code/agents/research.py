"""Research runs: researchers record checked facts; a lead writes a cited report.

The lead has no web tools. It writes the report from the facts its researchers record in the run's `Notebook`,
which is the run's only state. `research_profile` gives each run a fresh folder and a lead; `researcher_profile`
gives a lead researchers that write into its notebook. Every agent built here streams its replies.
"""

from dataclasses import asdict
from pathlib import Path

from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.core.llm import StreamingBackend, render_message
from jean_code.core.node import AgentNode
from jean_code.core.subagents import Profile, Run, agent_tree, last_reply
from jean_code.tools import web
from jean_code.tools.notebook import (
    Notebook,
    footer,
    format_facts,
    parse_sections,
    read_facts,
    render_report,
    write_json,
    write_text,
)

# The researcher prompt is separate from coding.SEARCHER_PROMPT on purpose: a searcher answers with facts and URLs,
# a researcher records facts and doesn't repeat them. Don't keep the two in sync.
RESEARCH_LEAD_PROMPT = """\
You lead a research run. The user message below is the brief. Researchers you start search the web and record \
facts into this run's notebook: each fact is one claim with a quote checked against the page it came from. You have \
no web tools. You write the report from those facts, one section at a time, with write_section. Nobody can answer a \
question from you: work with the brief you have, and list what you couldn't cover under Caveats.

## How to work
- **Split** the brief into sub-questions that don't overlap. Scale to the question: one researcher for a narrow \
fact, one per option for a comparison, three to five for a survey. Researchers spend from your budget.
- **Run a round in two replies.** First, spawn every researcher of the round with background=True, in one reply. \
Then collect every one of them in the next reply: subagent(1, wait=600), subagent(2, wait=600), …. Waiting costs \
nothing, and collecting them together means you read all the results at once. If a result says "still running", \
wait for it again. Results are never delivered to you any other way: a researcher you don't collect is cancelled \
when you finish.
- **Brief each researcher** on the objective and why it matters, what's in and out of scope, what the others cover, \
where to look, and what facts to return. It sees only the task you give it.
- **Write only what facts say.** Cite every factual sentence as [F12], or [F12, F13] for several; never cite a URL. \
Prefer ✓ facts; cite a ⚠ fact only if nothing better exists, and say so in Caveats. If a point needs a fact you \
don't have, spawn a researcher for it or list it under Caveats. Never fill it in from memory.
- **Draft as you go:** after the first round, write the Summary first, then the other sections, and rewrite them \
as later rounds come in. The draft is saved as you write it, so a run that stops early still has a report.
- **Before finishing,** call facts(unused_only=True), and use those facts or leave them out on purpose. Check that \
every Deliverable item in the brief has cited facts.

## The report
- Sections: Summary (3–5 bullets that answer the question), sections by theme or option, and Caveats (conflicting \
facts, old sources, ⚠ facts, Deliverable items not covered, assumptions made). Use ### for headings inside a \
section. Sources are rendered for you from the facts you cite.
- Final reply: one line saying the report is done. The report is the draft, not your reply.
"""

RESEARCHER_PROMPT = """\
You research one sub-question for a research lead: the user message below. You search the web, read pages, and \
record the facts the report may need with record_fact. The lead writes the report only from recorded facts. Nobody \
can answer a question from you.

## How to work
- Start broad with short queries, then narrow. Prefer primary sources (official docs, papers, original data) over \
blogs and aggregators.
- Record a fact for anything the report may need: numbers, dates and names, but also definitions, how things \
work, trade-offs and caveats. One claim per fact.
- Record as you read: after each fetch_page, record the facts that page supports before you fetch the next one. \
You have a limited number of rounds, and only recorded facts reach the lead.
- The quote is at least one full sentence, copied verbatim and without cuts from what fetch_page returned for that \
URL. It is checked against that text. A search snippet isn't a source: fetch the page first. If what you need is \
past the part shown, fetch again with start. Add the page's publication date if it shows one.
- If a fact comes back NOT verified, re-read the page and record it again with the exact quote, or leave it.
- Stop once the sub-question is answered, or when new searches stop turning up anything new.

## Final reply
A short note: what you covered, what you couldn't find, where sources conflict, and leads worth following. Don't \
repeat the facts: the lead gets every fact you recorded in full.
"""

SUMMARY_CHARS = 3000  # the most of the report's Summary a result hands back

RESEARCH_DESCRIPTION = (
    "researches a scoped question on the web and writes a cited report file; give it the brief from the research "
    "skill and run it in the background"
)
RESEARCHER_DESCRIPTION = (
    "searches the web for one sub-question and records facts, with quotes checked against the pages, into this "
    "run's notebook; give it the sub-question, what's in and out of scope, what the others cover and what facts "
    "to return"
)


def researcher_agent(
    notebook: Notebook, node: AgentNode, model: str = DEFAULT_MODEL, client: StreamingBackend | None = None
) -> Agent:
    """Returns a researcher: web search, plus the notebook's researcher tools.

    Args:
        notebook: The run's notebook.
        node: Its node in the lead's tree.
        model: OpenRouter model id.
        client: Where requests are sent; OpenRouter by default.
    """
    return Agent(
        model,
        system_prompt=RESEARCHER_PROMPT,
        tools={"web_search": web.web_search, **notebook.tools_for(node.path)},
        node=node,
        max_tool_rounds=15,
        compact_at=None,
        client=client,
        stream=True,
    )


def research_lead_agent(
    notebook: Notebook, node: AgentNode, model: str = DEFAULT_MODEL, client: StreamingBackend | None = None
) -> Agent:
    """Returns the lead: it reads the facts, writes the draft and starts researchers.

    It has no web tools.

    Args:
        notebook: The run's notebook.
        node: Its node.
        model: OpenRouter model id, for it and its researchers.
        client: Where requests are sent; OpenRouter by default.
    """
    # max_tool_rounds=40: the default 10 runs out on 3 spawns + 3 collects + 4 sections.
    return Agent(
        model,
        system_prompt=RESEARCH_LEAD_PROMPT,
        tools={"facts": notebook.facts, "write_section": notebook.write_section, "read_draft": notebook.read_draft},
        subagent_profiles={"researcher": researcher_profile(notebook, model, client)},
        node=node,
        max_tool_rounds=40,
        compact_at=None,
        client=client,
        stream=True,
    )


def researcher_profile(
    notebook: Notebook, model: str = DEFAULT_MODEL, client: StreamingBackend | None = None
) -> Profile:
    """Returns the lead's profile for researchers that all write into `notebook`.

    Args:
        notebook: The lead's notebook.
        model: OpenRouter model id.
        client: Where requests are sent; OpenRouter by default.
    """

    def start(task: str, node: AgentNode) -> Run:
        agent = researcher_agent(notebook, node, model, client)
        return Run.start(agent, task, report=lambda: researcher_result(agent, notebook))

    return Profile(RESEARCHER_DESCRIPTION, start, max_cost=0.10, max_cost_cap=0.25)


def research_profile(
    model: str = DEFAULT_MODEL, client: StreamingBackend | None = None, *, workspace: str | Path
) -> Profile:
    """Returns the research profile: a fresh folder and a lead for each run.

    Outside an agent, e.g. for an eval, a run is `research_profile(workspace=…).start(brief, AgentNode(…))`.

    Args:
        model: OpenRouter model id, for the lead and its researchers.
        client: Where requests are sent; OpenRouter by default.
        workspace: The folder sandboxed agents see as /workspace; the run folders go in its `research/`.
    """
    runs_dir = Path(workspace).resolve() / "research"  # resolved, so the `Report:` path it hands back is absolute

    def start(task: str, node: AgentNode) -> Run:
        notebook = Notebook.create(runs_dir, brief=task, path=node.path, model=model)
        lead = research_lead_agent(notebook, node, model, client)
        return Run.start(lead, task, report=lambda: research_result(lead, notebook))

    return Profile(RESEARCH_DESCRIPTION, start, max_cost=0.50, max_cost_cap=2.0)


def researcher_result(agent: Agent, notebook: Notebook) -> str:
    """Returns what the lead gets back from a researcher.

    That's every fact the researcher recorded, in full, then its last reply as the note.

    Args:
        agent: The researcher.
        notebook: The run's notebook.
    """
    mine = [f for f in read_facts(notebook.dir) if f.by == agent.node.path]
    return f"Facts recorded ({len(mine)}):\n{format_facts(mine)}\n\nNote:\n{last_reply(agent) or '(none)'}"


def research_result(lead: Agent, notebook: Notebook) -> str:
    """Finishes a run folder, and returns what the caller gets back.

    The caller gets the report's path, its Summary and the counts. It's the only code that finishes a run folder,
    so every run, stopped or not, leaves report.md, agents/ and budget.json.

    Args:
        lead: The run's lead.
        notebook: The run's notebook.
    """
    # Cancel and wait: a cancelled researcher still runs its last reply's tool calls, so its last record_facts
    # land before the report counts them.
    if lead.subagents:
        lead.subagents.close()
    report = render_report(notebook.dir)
    write_text(notebook.dir / "report.md", report)
    (notebook.dir / "agents").mkdir(exist_ok=True)
    for path, agent in agent_tree(lead).items():
        write_json(
            notebook.dir / "agents" / f"{path.replace('/', '-')}.json",
            {
                "path": path,
                "task": agent.node.task,
                "messages": [render_message(m) for m in agent.messages],
                "usage": [asdict(c) for c in agent.node.entries],
            },
        )
    write_json(
        notebook.dir / "budget.json",
        {"limit": lead.node.limit, "spent": lead.node.spent, "breakdown": lead.node.breakdown()},
    )
    return f"Report: {notebook.dir / 'report.md'}\n\n{summary_of(report)}\n\n{footer(notebook.dir)}"


def summary_of(report: str) -> str:
    """Returns the rendered report's Summary section, else its first section.

    With no draft, the first section holds the uncited facts. The section is cut at `SUMMARY_CHARS`.

    Args:
        report: The rendered report.
    """
    sections = parse_sections(report.rsplit("\n---\n", 1)[0])  # without the footer
    if not sections:
        return "(the report is empty: no facts and no draft)"
    title, body = next((s for s in sections if s[0].casefold() == "summary"), sections[0])
    if len(body) > SUMMARY_CHARS:
        body = body[:SUMMARY_CHARS] + "\n… (cut; the rest is in the report)"
    return f"## {title}\n{body}"
