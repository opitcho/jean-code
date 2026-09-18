"""The coding agent both front-ends run: a shell plus web tools."""

from pathlib import Path

from agent import Agent, DEFAULT_MODEL
from config import SKILLS_DIR
from prompts import SYSTEM_PROMPT
from shell import DockerShell, LocalShell
from tools import fetch_page, web_search


def coding_agent(
    model: str = DEFAULT_MODEL,
    sandbox: bool = False,
    network: bool = True,
    skills_dir: str | Path | None = SKILLS_DIR,
    **kwargs,
) -> Agent:
    """An agent with a shell (on the host, or in the Docker sandbox) and web tools.

    Whether `bash` needs approval comes from the shell: yes on the host, no in the sandbox.
    `agent.close()` closes the shell: it kills background jobs, and removes the container.
    """
    shell = DockerShell(network=network) if sandbox else LocalShell()
    return Agent(
        model,
        system_prompt=f"{SYSTEM_PROMPT}\n\n{shell.describe()}".strip(),
        tools={"bash": shell.bash, "job": shell.job, "web_search": web_search, "fetch_page": fetch_page},
        needs_approval={"bash"} if shell.requires_approval else set(),
        cleanup=[shell.close],
        skills_dir=skills_dir,
        **kwargs,
    )
