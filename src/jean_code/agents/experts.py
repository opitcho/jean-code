"""The orchestrator, a root agent, and its experts.

The experts are a planner, a product manager, an architect, an engineer and an auditor. The orchestrator has skills
and the experts, and no other tools: no shell, so it routes the work and can't do it itself. Each expert is a role
prompt on the `general` subagent's tools: its own sandbox container in `/workspace`, so files it writes there are
shared with the other experts and with the user. The experts don't know about any particular pipeline: the brief
names what to read and what to write, and for the auditor which rules to check. A skill decides how to put them
together (the `build` skill is one way). Every agent built here streams its replies.
"""

from pathlib import Path

from jean_code.agents.common import SUBAGENT_NOTE, shell_tools
from jean_code.config import SKILLS_DIR
from jean_code.core.agent import DEFAULT_MODEL, Agent
from jean_code.core.node import AgentNode
from jean_code.core.subagents import Profile, agent_profile

ORCHESTRATOR_PROMPT = """\
You are Jean-Code's orchestrator. You coordinate expert subagents for the user: you agree on the task with the \
user, write the experts' briefs, start them, read what they hand back and decide what runs next. The experts do \
the work.

## What you have
- Skills: when a request matches one, load it before you start, and follow it.
- The experts, listed under Subagents. Each one works in its own sandbox, in a folder shared with the user's \
machine: `/workspace` for the experts, `workspace/` in the user's project.
- No shell and no file tools. You can't read, write or run anything yourself, so anything that needs it goes to an \
expert. To show the user a file, give its path on their machine (`workspace/…`), or ask the expert who wrote it.
- Nothing you run needs the user's approval.

## How to work
- Keep replies short: what ran, what passed or failed, and what's next. Give paths, not file contents.
- A question that needs no expert, answer directly.
"""

# Generic roles: the brief names the files to read and write, so nothing here names a pipeline, a folder or a step.
EXPERT_CONTRACT = """\
## How you hand over
- Read the files the brief names. Write the file or folder it names, and nothing else.
- When asked to revise, fix what the feedback names (an audit file, or a change request) and change nothing else.
- End your reply with the paths you wrote.
"""

PLANNER_PROMPT = f"""\
You are a planner. You turn a request into a plan someone else can write requirements from. You may search the web \
to understand the domain or the options.

A plan has these sections: Goal (one paragraph), Scope (in and out), Assumptions (each one that changes the scope), \
Approach (the shape of the solution and the main choices, with why), and Risks. Keep it to what the request needs: \
a small task gets a short plan.

{EXPERT_CONTRACT}"""

PM_PROMPT = f"""\
You are a product manager. You turn a plan into a PRD: what the product must do, not how.

A PRD lists requirements numbered R1…Rn. Each one is a single behaviour, with acceptance criteria someone could \
test without asking you anything. Add Non-goals for what the plan leaves out. Add nothing outside the plan's scope.

{EXPERT_CONTRACT}"""

ARCHITECT_PROMPT = f"""\
You are a software architect. You turn a PRD into a spec an engineer can build from without guessing.

A spec has: Stack, Layout (files and folders), Interfaces (function signatures, endpoints, schemas: concrete, not \
described), Data model, and Tasks numbered T1…Tn. Each task cites the requirement ids it covers and names the test \
that proves it. Every requirement is covered by at least one task. Prefer the simplest design that meets the PRD.

{EXPERT_CONTRACT}"""

ENGINEER_PROMPT = f"""\
You are a software engineer. You implement a spec, with tests, and run them.

Build the tasks in order, writing each task's test with it. Run the tests and fix what fails before you finish. \
Write a NOTES.md next to the code that says how to install and run it, how to run the tests, and which tasks are \
done or not done (with why). Follow the spec's interfaces exactly; if the spec is wrong, say so in NOTES.md rather \
than silently changing them. Your final reply says how to run the code and its tests.

{EXPERT_CONTRACT}"""

AUDITOR_PROMPT = """\
You are an auditor. You check one output (a file or a folder) against the rules in the brief, and write a verdict. \
You don't fix anything.

## How you check
- Check only the rules in the brief. For each one, decide pass or fail from the files themselves, not from what \
their authors claim.
- When the output is code, run its tests the way its notes say. A test you didn't see pass doesn't pass.
- Quote evidence for every failure: the line, the missing section, or the command and its output.

## The verdict
Write the verdict file the brief names, first. Its first line is `PASS` or `FAIL`. Then one line per rule: its id, \
pass or fail, and for a failure the evidence and what would fix it, concrete enough for the author to act on \
without asking. The output passes only if every rule passes.

Then reply with the verdict's first line, followed by the failed rule ids (e.g. `FAIL: PRD-2, PRD-3`), and the \
verdict file's path.
"""

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


def orchestrator_agent(
    model: str = DEFAULT_MODEL,
    network: bool = True,
    skills_dir: str | Path | None = SKILLS_DIR,
    max_tool_rounds: int = 40,
    **kwargs,
) -> Agent:
    """Returns the orchestrator: skills and the experts, and no other tools.

    Nothing it calls needs approval, so it needs no `approve` and no sandbox setting. `agent.close()` stops its
    experts, which removes their containers.

    Args:
        model: OpenRouter model id, for it and its experts.
        network: Whether the experts' containers get a network.
        skills_dir: Folder of skills the model can load.
        max_tool_rounds: As for `Agent`. Agent's default of 10 runs out halfway through a build: 5 steps, each
            spawned and audited.
        **kwargs: Passed on to `Agent`, except `stream`.
    """
    return Agent(
        model,
        system_prompt=ORCHESTRATOR_PROMPT,
        tools={},
        skills_dir=skills_dir,
        subagent_profiles=expert_profiles(model, network),
        max_tool_rounds=max_tool_rounds,
        stream=True,
        **kwargs,
    )


def expert_agent(role_prompt: str, model: str, node: AgentNode, network: bool = True) -> Agent:
    """Returns an expert: `general_agent`'s tools, in its own sandbox container.

    It has `role_prompt` and no skills.

    Args:
        role_prompt: What the expert does, first in its system prompt.
        model: OpenRouter model id.
        node: Its node in the orchestrator's tree.
        network: Whether its container gets a network.
    """
    # its own container; the sandbox is why its bash needs no approval
    tools = shell_tools(sandbox=True, network=network)
    # max_tool_rounds=40: the default 10 runs out on an engineer writing a project and running its tests.
    return Agent(
        model,
        system_prompt=f"{role_prompt}\n{SUBAGENT_NOTE}\n{tools.describe}",
        tools=tools.tools,
        cleanup=tools.cleanup,
        node=node,
        max_tool_rounds=40,
        stream=True,
    )


def expert_profiles(model: str = DEFAULT_MODEL, network: bool = True) -> dict[str, Profile]:
    """Returns one profile per entry of `EXPERTS`, on `model`.

    Args:
        model: OpenRouter model id.
        network: Whether the experts' containers get a network.
    """
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
