"""Tests for the orchestrator and its experts (experts.py), and the `build` skill that puts them together.

Nothing here starts a container or calls a model. Each test names the bug it would catch.
"""

import re
from types import MethodType

import pytest

from jean_code.agents.coding import subagent_profiles
from jean_code.agents.experts import EXPERTS, expert_agent, orchestrator_agent
from jean_code.config import SKILLS_DIR
from jean_code.core.node import AgentNode
from jean_code.core.skills import Skills
from jean_code.tools.shell import DockerShell


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # builds a client; nothing here sends a request


def test_the_coding_agent_has_no_experts(tmp_path):
    # Catches the experts leaking back into the coding agent: builds belong to the orchestrator, which has no shell.
    assert set(subagent_profiles("m", workspace=tmp_path)) == {"searcher", "general", "research"}


def test_the_orchestrator_has_no_tools_of_its_own():
    # Catches a shell, file or web tool on the orchestrator: with one, it does the experts' work instead of routing it.
    agent = orchestrator_agent("m")
    try:
        assert set(agent._tools) == {"load_skill", "read_skill_file", "spawn", "subagent", "resume"}
        assert agent.needs_approval == set()
    finally:
        agent.close()


def test_the_orchestrator_lists_the_experts_and_research():
    # Catches experts that exist but never reach the orchestrator's prompt, or a research question it can only
    # answer from the model's memory.
    agent = orchestrator_agent("m", skills_dir=None)
    try:
        prompt = agent.messages[0].content
        assert all(f"- {name}: " in prompt for name in [*EXPERTS, "research"])
        assert "load_skill" not in agent._tools  # no skills, and no crash without them
    finally:
        agent.close()


def test_the_orchestrator_and_its_experts_stream(tmp_path):
    # Catches an expert whose reply lands whole: nothing shows while it works, and a stop can't cut it off.
    for agent in [orchestrator_agent("m", skills_dir=None)] + [
        expert_agent(prompt, "m", AgentNode(1.0), network=False, workspace=tmp_path)
        for _, prompt, _, _ in EXPERTS.values()
    ]:
        assert agent.stream
        agent.close()


@pytest.mark.parametrize("name", list(EXPERTS))
def test_experts_run_bash_in_docker_without_approval(name, tmp_path):
    # Catches an expert that Run.start refuses (host bash needs approval) or that runs on the host.
    _, prompt, max_cost, _ = EXPERTS[name]
    node = AgentNode().add_agent(max_cost)
    agent = expert_agent(prompt, "m", node, network=False, workspace=tmp_path)
    try:
        assert agent.needs_approval == set()
        bash = agent.tool_map["bash"]
        assert isinstance(bash, MethodType) and isinstance(bash.__self__, DockerShell)
        assert agent.node is node  # spawn's check that the profile used its node
        assert prompt in agent.messages[0].content and "## You are a subagent" in agent.messages[0].content
    finally:
        agent.close()


@pytest.mark.parametrize("name", list(EXPERTS))
def test_experts_have_no_skills_or_subagents(name, tmp_path):
    # Catches an expert that loads a skill meant for the orchestrator, or spawns experts of its own.
    agent = expert_agent(EXPERTS[name][1], "m", AgentNode(1.0), network=False, workspace=tmp_path)
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


def test_experts_have_room_for_a_whole_step(tmp_path):
    # Catches the default 10 rounds: an engineer writing a project and running its tests stops halfway.
    agent = expert_agent(EXPERTS["engineer"][1], "m", AgentNode(1.0), network=False, workspace=tmp_path)
    try:
        assert agent.settings.max_tool_rounds == 40
    finally:
        agent.close()


def test_the_orchestrator_has_room_for_a_whole_build():
    # Catches the default 10 rounds: a build (4 experts, a sweep, and reruns after a failed sweep) stops halfway.
    agent = orchestrator_agent("m", skills_dir=None)
    try:
        assert agent.settings.max_tool_rounds == 40
    finally:
        agent.close()


def test_the_orchestrators_subagents_use_its_workspace(tmp_path):
    # Catches an expert mounting another folder, or research writing where the experts can't read it.
    agent = orchestrator_agent("m", skills_dir=None, workspace=tmp_path)
    try:
        for profile in agent.subagents.profiles.values():  # what spawn starts from
            node = AgentNode(1.0)
            node.cancel()  # makes no call: the run stops at once
            run = profile.start("t", node)
            assert run.wait(5)
            if bash := run.agent.tool_map.get("bash"):
                assert isinstance(bash, MethodType) and isinstance(shell := bash.__self__, DockerShell)
                assert shell.workdir == tmp_path.resolve()
        assert len(list((tmp_path / "research").iterdir())) == 1
    finally:
        agent.close()
