"""The coding agent, a root agent: a shell plus web tools, and its `searcher`, `general` and `research` subagents."""

from pathlib import Path

from jean_code.agents.common import SUBAGENT_NOTE, shell_tools
from jean_code.agents.research import research_profile
from jean_code.config import SKILLS_DIR
from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.core.node import AgentNode
from jean_code.core.subagents import Profile, agent_profile
from jean_code.tools.web import fetch_page, web_search

CODING_PROMPT = """\
You are Jean-Code, a coding agent. You work in the user's project through a shell, and you can search \
and read the web. You help with software tasks: reading and explaining code, fixing bugs, adding \
features, running tests and tools.

## How to work
- Explore before editing: read the relevant files and check how the code is used before changing it.
- Make the smallest change that does the job, in the style of the surrounding code.
- Verify your work by running it: the tests, the script, or a quick check. Say what you ran and what it showed.
- If a request is ambiguous and a wrong guess would be costly, ask one short question instead of guessing.
- Keep replies short and concrete. Say what you did and what is left; skip filler and restating the question.

## Tools
- Use the shell for files: `ls`, `cat`, `sed -n`, `grep -rn` and `find` to read; heredocs, `sed -i` or a short \
script to write. Read a file before overwriting it.
- Start long jobs (servers, big builds, long test runs) with `background=True`, then check on them with the job tool.
- An error in a tool result is information, not the end of the task: read it, fix the cause and try again. \
Don't repeat a call that failed unchanged.
- Use web_search and fetch_page for documentation or errors you don't know; say where facts came from.

## Approval
Some calls wait for the user's approval. If the user denies one, don't retry the same call: read their \
reason, then adjust your approach or ask what they would prefer.
"""

SEARCHER_PROMPT = """\
You look things up on the web for another agent, one question at a time: the user message below. Search, read \
the most relevant pages, then answer. Nobody can answer a question from you. Your final reply is the only thing \
handed back: the answer first, then the URL each fact came from, and what you couldn't confirm. Keep it short.
"""

SEARCHER_DESCRIPTION = "looks something up on the web and answers with its sources; give it one clear question"
GENERAL_DESCRIPTION = (
    "a coding agent like you, with bash, jobs and the web, but its shell runs in a Docker sandbox in /workspace (the "
    "project's workspace/ folder, not the user's directory); for self-contained work: prototypes, experiments, "
    "trying a library, long runs"
)


def coding_agent(
    model: str = DEFAULT_MODEL,
    sandbox: bool = False,
    network: bool = True,
    skills_dir: str | Path | None = SKILLS_DIR,
    subagents: bool = True,
    **kwargs,
) -> Agent:
    """An agent with a shell (on the host, or in the Docker sandbox), web tools, and the subagents in
    `subagent_profiles` unless `subagents` is False.

    Whether `bash` needs approval comes from the shell: yes on the host, no in the sandbox. On the host, pass the
    `approve` function that asks the user (see `Agent`).
    `agent.close()` stops its subagents and closes the shell: it kills background jobs, and removes the container.
    """
    tools = shell_tools(sandbox, network)
    return Agent(
        model,
        system_prompt=f"{CODING_PROMPT}\n\n{tools.describe}".strip(),
        tools=tools.tools,
        needs_approval=tools.needs_approval,
        cleanup=tools.cleanup,
        skills_dir=skills_dir,
        subagent_profiles=subagent_profiles(model, network, skills_dir) if subagents else None,
        **kwargs,
    )


def searcher_agent(model: str, node: AgentNode) -> Agent:
    """The `searcher` subagent: web tools only."""
    return Agent(
        model, system_prompt=SEARCHER_PROMPT, tools={"web_search": web_search, "fetch_page": fetch_page}, node=node
    )


def general_agent(
    model: str, node: AgentNode, network: bool = True, skills_dir: str | Path | None = SKILLS_DIR
) -> Agent:
    """The `general` subagent: the coding agent's tools, in its own sandbox container, without subagents."""
    # its own container; the sandbox is why its bash needs no approval
    tools = shell_tools(sandbox=True, network=network)
    return Agent(
        model,
        system_prompt=f"{CODING_PROMPT}\n{SUBAGENT_NOTE}\n{tools.describe}",
        tools=tools.tools,
        cleanup=tools.cleanup,
        skills_dir=skills_dir,
        node=node,
    )


def subagent_profiles(
    model: str = DEFAULT_MODEL, network: bool = True, skills_dir: str | Path | None = SKILLS_DIR
) -> dict[str, Profile]:
    """The coding agent's subagents, on its model:

    - `searcher`: web tools only, answers one question with sources. Cheap, and safe anywhere.
    - `general`: a copy of the coding agent without subagents of its own, whose shell always runs in the Docker
      sandbox (its own container, in `workspace/`), so it needs no approval, even when the parent runs on the host.
    - `research`: a research lead and its researchers (research.py), which write a cited report to a run folder in
      `workspace/research/`. The research skill scopes the brief before spawning one.
    """
    return {
        "searcher": agent_profile(
            SEARCHER_DESCRIPTION, lambda task, node: searcher_agent(model, node), max_cost=0.05, max_cost_cap=0.25
        ),
        "general": agent_profile(
            GENERAL_DESCRIPTION,
            lambda task, node: general_agent(model, node, network, skills_dir),
            max_cost=0.50,
            max_cost_cap=2.0,
        ),
        "research": research_profile(model),
    }
