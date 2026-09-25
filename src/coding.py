"""The coding agent both front-ends run: a shell plus web tools, and two kinds of subagent."""

from pathlib import Path

from agent import Agent, DEFAULT_MODEL
from config import SKILLS_DIR
from prompts import SEARCHER_PROMPT, SUBAGENT_NOTE, SYSTEM_PROMPT
from shell import DockerShell, LocalShell
from subagents import Profile, agent_profile
from tools import fetch_page, web_search
from usage import Budget


def coding_agent(
    model: str = DEFAULT_MODEL,
    sandbox: bool = False,
    network: bool = True,
    skills_dir: str | Path | None = SKILLS_DIR,
    subagents: bool = True,
    **kwargs,
) -> Agent:
    """An agent with a shell (on the host, or in the Docker sandbox), web tools, and the `searcher` and `general`
    subagents (see `subagent_profiles`) unless `subagents` is False.

    Whether `bash` needs approval comes from the shell: yes on the host, no in the sandbox.
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
        **kwargs,
    )


def subagent_profiles(model: str = DEFAULT_MODEL, network: bool = True,
                      skills_dir: str | Path | None = SKILLS_DIR) -> dict[str, Profile]:
    """The coding agent's subagents, on its model:

    - `searcher`: web tools only, answers one question with sources. Cheap, and safe anywhere.
    - `general`: a copy of the coding agent without subagents of its own, whose shell always runs in the Docker
      sandbox (its own container, in `workspace/`), so it needs no approval, even when the parent runs on the host.
    """
    return {
        "searcher": agent_profile(
            "looks something up on the web and answers with its sources; give it one clear question",
            lambda task, budget: searcher_agent(model, budget), max_cost=0.05, max_cost_cap=0.25),
        "general": agent_profile(
            "a coding agent like you, with bash, jobs and the web, but its shell runs in a Docker sandbox in "
            "/workspace (the project's workspace/ folder, not the user's directory); for self-contained work: "
            "prototypes, experiments, trying a library, long runs",
            lambda task, budget: general_agent(model, budget, network, skills_dir), max_cost=0.50, max_cost_cap=2.0),
    }


def searcher_agent(model: str, budget: Budget) -> Agent:
    return Agent(model, system_prompt=SEARCHER_PROMPT,
                 tools={"web_search": web_search, "fetch_page": fetch_page}, budget=budget)


def general_agent(model: str, budget: Budget, network: bool = True,
                  skills_dir: str | Path | None = SKILLS_DIR) -> Agent:
    shell = DockerShell(network=network)  # its own container; the sandbox is why its bash needs no approval
    return Agent(
        model,
        system_prompt=f"{SYSTEM_PROMPT}\n{SUBAGENT_NOTE}\n{shell.describe()}",
        tools={"bash": shell.bash, "job": shell.job, "web_search": web_search, "fetch_page": fetch_page},
        cleanup=[shell.close],
        skills_dir=skills_dir,
        budget=budget,
    )
