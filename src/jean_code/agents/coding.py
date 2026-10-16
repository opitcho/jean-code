"""The coding agent both front-ends run: a shell plus web tools, and its subagents."""

from pathlib import Path

from jean_code.agents.experts import expert_profiles
from jean_code.agents.prompts import SEARCHER_PROMPT, SUBAGENT_NOTE, SYSTEM_PROMPT
from jean_code.agents.research import research_profile
from jean_code.config import SKILLS_DIR
from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.core.node import AgentNode
from jean_code.core.subagents import Profile, agent_profile
from jean_code.tools.shell import DockerShell, LocalShell
from jean_code.tools.web import fetch_page, web_search


def coding_agent(
    model: str = DEFAULT_MODEL,
    sandbox: bool = False,
    network: bool = True,
    skills_dir: str | Path | None = SKILLS_DIR,
    subagents: bool = True,
    max_tool_rounds: int = 40,  # the default 10 runs out halfway through a build: 5 steps, each spawned and audited
    **kwargs,
) -> Agent:
    """An agent with a shell (on the host, or in the Docker sandbox), web tools, and the subagents in
    `subagent_profiles` unless `subagents` is False.

    Whether `bash` needs approval comes from the shell: yes on the host, no in the sandbox. On the host, pass the
    `approve` function that asks the user (see `Agent`).
    `agent.close()` stops its subagents and closes the shell: it kills background jobs, and removes the container.
    """
    shell = DockerShell(network=network) if sandbox else LocalShell()
    return Agent(
        model,
        system_prompt=f"{SYSTEM_PROMPT}\n\n{shell.describe()}".strip(),
        tools={"bash": shell.bash, "job": shell.job, "web_search": web_search, "fetch_page": fetch_page},
        needs_approval={"bash"} if shell.requires_approval else set(),
        cleanup=[shell.close],
        skills_dir=skills_dir,
        subagent_profiles=subagent_profiles(model, network, skills_dir) if subagents else None,
        max_tool_rounds=max_tool_rounds,
        **kwargs,
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
    - the experts (experts.py): `planner`, `pm`, `architect`, `engineer` and `auditor`, each in its own sandbox
      container like `general`. Skills such as `build` put them together.
    """
    return {
        "searcher": agent_profile(
            "looks something up on the web and answers with its sources; give it one clear question",
            lambda task, node: searcher_agent(model, node),
            max_cost=0.05,
            max_cost_cap=0.25,
        ),
        "general": agent_profile(
            "a coding agent like you, with bash, jobs and the web, but its shell runs in a Docker sandbox in "
            "/workspace (the project's workspace/ folder, not the user's directory); for self-contained work: "
            "prototypes, experiments, trying a library, long runs",
            lambda task, node: general_agent(model, node, network, skills_dir),
            max_cost=0.50,
            max_cost_cap=2.0,
        ),
        "research": research_profile(model),
    } | expert_profiles(model, network)


def searcher_agent(model: str, node: AgentNode) -> Agent:
    """The `searcher` subagent: web tools only."""
    return Agent(
        model, system_prompt=SEARCHER_PROMPT, tools={"web_search": web_search, "fetch_page": fetch_page}, node=node
    )


def general_agent(
    model: str, node: AgentNode, network: bool = True, skills_dir: str | Path | None = SKILLS_DIR
) -> Agent:
    """The `general` subagent: the coding agent's tools, in its own sandbox container, without subagents."""
    shell = DockerShell(network=network)  # its own container; the sandbox is why its bash needs no approval
    return Agent(
        model,
        system_prompt=f"{SYSTEM_PROMPT}\n{SUBAGENT_NOTE}\n{shell.describe()}",
        tools={"bash": shell.bash, "job": shell.job, "web_search": web_search, "fetch_page": fetch_page},
        cleanup=[shell.close],
        skills_dir=skills_dir,
        node=node,
    )
