"""Expert subagents: a planner, a product manager, an architect, an engineer and an auditor.

Each one is a role prompt on the `general` subagent's tools: its own sandbox container in `/workspace`, so files it
writes there are shared with the other experts and with the user. They don't know about any particular pipeline: the
brief names what to read and what to write, and for the auditor which rules to check. A skill decides how to put them
together (`skills/build` is one way).
"""

from jean_code.agents.prompts import (
    ARCHITECT_PROMPT,
    AUDITOR_PROMPT,
    ENGINEER_PROMPT,
    PLANNER_PROMPT,
    PM_PROMPT,
    SUBAGENT_NOTE,
)
from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.core.node import AgentNode
from jean_code.core.subagents import Profile, agent_profile
from jean_code.tools.shell import DockerShell
from jean_code.tools.web import fetch_page, web_search

EXPERTS = {  # name → (listing line, role prompt, max_cost, max_cost_cap)
    "planner": (
        "turns a request into a plan: goal, scope, assumptions, approach, risks",
        PLANNER_PROMPT,
        0.25,
        0.75,
    ),
    "pm": (
        "turns a plan into a PRD: numbered requirements with acceptance criteria",
        PM_PROMPT,
        0.25,
        0.75,
    ),
    "architect": (
        "turns a PRD into a spec: design, interfaces, tasks tied to requirement ids",
        ARCHITECT_PROMPT,
        0.50,
        1.5,
    ),
    "engineer": (
        "implements a spec, with tests, and runs them",
        ENGINEER_PROMPT,
        1.0,
        3.0,
    ),
    "auditor": (
        "checks a file or folder against the rules in its brief and writes a PASS/FAIL verdict file; give it the "
        "rules, the output to check, its inputs and where to write the verdict",
        AUDITOR_PROMPT,
        0.20,
        0.50,
    ),
}


def expert_profiles(model: str = DEFAULT_MODEL, network: bool = True) -> dict[str, Profile]:
    """One profile per entry of `EXPERTS`, on `model`."""
    return {
        # prompt=prompt binds each prompt now: a closure would see only the loop's last one
        name: agent_profile(
            description,
            lambda task, node, prompt=prompt: expert_agent(prompt, model, node, network),
            max_cost=max_cost,
            max_cost_cap=cap,
        )
        for name, (description, prompt, max_cost, cap) in EXPERTS.items()
    }


def expert_agent(role_prompt: str, model: str, node: AgentNode, network: bool = True) -> Agent:
    """An expert: `general_agent`'s tools in its own sandbox container, with `role_prompt` and no skills."""
    shell = DockerShell(network=network)  # its own container; the sandbox is why its bash needs no approval
    # max_tool_rounds=40: the default 10 runs out on an engineer writing a project and running its tests.
    return Agent(
        model,
        system_prompt=f"{role_prompt}\n{SUBAGENT_NOTE}\n{shell.describe()}",
        tools={"bash": shell.bash, "job": shell.job, "web_search": web_search, "fetch_page": fetch_page},
        cleanup=[shell.close],
        node=node,
        max_tool_rounds=40,
    )
