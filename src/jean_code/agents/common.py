"""Shared by the agent modules: the subagent note, and the shell-plus-web toolset.

Prompts across `agents/` are static text only (no timestamps or other per-call data), so the prompt prefix stays
cached.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jean_code.tools.shell import DockerShell, LocalShell
from jean_code.tools.web import fetch_page, web_search

# Every subagent: the task another agent gave it is its only user message.
SUBAGENT_NOTE = """\
## You are a subagent
Another agent started you on one task: the user message below. Nobody reads your messages until you finish, \
and nobody can answer a question, so don't ask any: make a reasonable assumption and say what it was. Your \
final reply is the only thing handed back, so make it complete on its own: what you did, what you found, and \
the paths of any files you wrote.
"""


@dataclass(frozen=True)
class Toolset:
    """Tools for an `Agent`, with what goes with them.

    Attributes:
        tools: The tools, by the name the model calls them.
        needs_approval: The names of the tools whose calls need approval.
        cleanup: What to run to release what the tools hold.
        describe: One line for the system prompt, saying where commands run.
    """

    tools: dict[str, Callable]
    needs_approval: set[str]
    cleanup: list[Callable[[], None]]
    describe: str


def shell_tools(sandbox: bool = True, network: bool = True, *, workspace: str | Path) -> Toolset:
    """Returns `bash` and `job` on a shell, plus `web_search` and `fetch_page`.

    `bash` needs approval only on the host.

    Args:
        sandbox: Run the shell in its own Docker container instead of on the host.
        network: Whether the container gets a network.
        workspace: The folder the container mounts at /workspace; unused on the host.
    """
    shell = DockerShell(workspace, network) if sandbox else LocalShell()
    return Toolset(
        tools={"bash": shell.bash, "job": shell.job, "web_search": web_search, "fetch_page": fetch_page},
        needs_approval={"bash"} if shell.requires_approval else set(),
        cleanup=[shell.close],
        describe=shell.describe(),
    )
