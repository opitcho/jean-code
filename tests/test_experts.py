"""Tests for the expert profiles (experts.py) and the `build` skill that puts them together.

Nothing here starts a container or calls a model. Each test names the bug it would catch.
"""

import re
from types import MethodType

import pytest

from jean_code.agents.coding import coding_agent, subagent_profiles
from jean_code.agents.experts import EXPERTS, expert_agent
from jean_code.config import SKILLS_DIR
from jean_code.core.node import AgentNode
from jean_code.core.skills import Skills
from jean_code.tools.shell import DockerShell


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # builds a client; nothing here sends a request


def test_the_coding_agent_lists_the_experts():
    # Catches experts that exist but never reach the orchestrator, or that replace the existing profiles.
    assert {"planner", "pm", "architect", "engineer", "auditor", "searcher", "general", "research"} <= set(
        subagent_profiles("m")
    )


@pytest.mark.parametrize("name", list(EXPERTS))
def test_experts_run_bash_in_docker_without_approval(name):
    # Catches an expert that Run.start refuses (host bash needs approval) or that runs on the host.
    _, prompt, max_cost, _ = EXPERTS[name]
    node = AgentNode().child(max_cost, "1")
    agent = expert_agent(prompt, "m", node, network=False)
    try:
        assert agent.needs_approval == set()
        bash = agent.tool_map["bash"]
        assert isinstance(bash, MethodType) and isinstance(bash.__self__, DockerShell)
        assert agent.node is node  # spawn's check that the profile used its node
        assert prompt in agent.messages[0].content and "## You are a subagent" in agent.messages[0].content
    finally:
        agent.close()


@pytest.mark.parametrize("name", list(EXPERTS))
def test_experts_have_no_skills_or_subagents(name):
    # Catches an expert that loads a skill meant for the orchestrator, or spawns experts of its own.
    agent = expert_agent(EXPERTS[name][1], "m", AgentNode(1.0), network=False)
    try:
        assert agent.skills.catalog == {} and agent.subagents is None
    finally:
        agent.close()


def test_the_build_skill_has_rules_for_every_step_it_audits():
    # Catches a skill that sends the auditor a step whose rules rules.md doesn't have, e.g. after renaming a step.
    skills = Skills(SKILLS_DIR)
    assert skills.catalog["build"].description
    steps = set(re.findall(r"`([A-Z]+)-\*`", skills.load_skill("build")))
    rules = skills.read_skill_file("build", "rules.md")
    assert steps == {"PLAN", "PRD", "SPEC", "IMPL", "SWEEP"}
    for step in steps:
        assert re.search(rf"\*\*{step}-\d+\*\*", rules), step


def test_experts_have_room_for_a_whole_step():
    # Catches the default 10 rounds: an engineer writing a project and running its tests stops halfway.
    agent = expert_agent(EXPERTS["engineer"][1], "m", AgentNode(1.0), network=False)
    try:
        assert agent.max_tool_rounds == 40
    finally:
        agent.close()


def test_the_orchestrator_has_room_for_a_whole_build():
    # Catches the default 10 rounds on the coding agent: a build (5 steps, each spawned and audited) stops halfway.
    agent = coding_agent(skills_dir=None, approve=lambda name, arguments: None)
    try:
        assert agent.max_tool_rounds == 40
    finally:
        agent.close()
