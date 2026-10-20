"""Tests for the coding agent's wiring: its subagent profiles, and Docker shells sharing a workspace.

Nothing here starts a container or calls a model. Each test names the bug it would catch.
"""

from types import MethodType

import pytest

from jean_code.agents.coding import coding_agent, general_agent, searcher_agent, subagent_profiles
from jean_code.core.node import AgentNode
from jean_code.core.subagents import Run
from jean_code.tools.shell import DockerShell


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # builds a client; nothing here sends a request


def yes(name: str, arguments: dict) -> None:
    """An `approve` that lets every call run: the host shell's `bash` needs one."""


def test_the_coding_agent_offers_its_subagents():
    # Catches profiles that exist but never reach the model: no listing, or no tools.
    agent = coding_agent(skills_dir=None, approve=yes)
    try:
        prompt = agent.messages[0].content
        assert "## Subagents" in prompt and "- searcher: " in prompt and "- general: " in prompt
        assert "- research: " in prompt
        assert {"spawn", "subagent"} <= set(agent._tools)
    finally:
        agent.close()


def test_subagents_have_no_subagents_of_their_own(tmp_path):
    # Catches unbounded recursion: a general subagent that can spawn more general subagents.
    for build in (
        lambda b: searcher_agent("m", b),
        lambda b: general_agent("m", b, skills_dir=None, workspace=tmp_path),
    ):
        agent = build(AgentNode(1.0))
        assert agent.subagents is None
        agent.close()


def test_the_coding_agent_and_its_subagents_stream(tmp_path):
    # Catches a subagent whose reply lands whole: nothing shows while it works, and a stop can't cut it off.
    builds = (
        lambda b: coding_agent(skills_dir=None, approve=yes, node=b),
        lambda b: searcher_agent("m", b),
        lambda b: general_agent("m", b, skills_dir=None, workspace=tmp_path),
    )
    for build in builds:
        agent = build(AgentNode(1.0))
        assert agent.stream
        agent.close()


def test_the_general_subagent_runs_bash_in_docker_without_approval(tmp_path):
    # Catches a general subagent that Run.start refuses (host bash needs approval) or that runs on the host.
    node = AgentNode().add_agent(0.5)
    agent = general_agent("m", node, skills_dir=None, workspace=tmp_path)
    try:
        assert agent.needs_approval == set() and "bash" in agent.tool_map
        bash = agent.tool_map["bash"]
        assert isinstance(bash, MethodType) and isinstance(bash.__self__, DockerShell)
        assert agent.node is node  # spawn's check that the profile used its node
        assert "## You are a subagent" in agent.messages[0].content
    finally:
        agent.close()


def test_profile_limits(tmp_path):
    profiles = subagent_profiles("m", workspace=tmp_path)
    assert (profiles["searcher"].max_cost, profiles["searcher"].cap) == (0.05, 0.25)
    assert (profiles["general"].max_cost, profiles["general"].cap) == (0.50, 2.0)


def test_the_coding_agents_sandboxes_use_its_workspace(tmp_path):
    # Catches its own sandbox or `general`'s mounting another folder, or research writing where they can't read it.
    agent = coding_agent(sandbox=True, skills_dir=None, workspace=tmp_path)
    try:
        bash = agent.tool_map["bash"]
        assert isinstance(bash, MethodType) and isinstance(shell := bash.__self__, DockerShell)
        assert shell.workdir == tmp_path.resolve()
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


def test_a_searcher_run_starts():
    # Catches a profile whose agent Run.start refuses.
    agent = searcher_agent("m", AgentNode(1.0))
    agent.node.cancel()  # makes no call: the run stops at once
    assert Run.start(agent, "q").wait(5)


def test_docker_shells_in_one_workspace_keep_separate_jobs(tmp_path):
    # Catches an agent and its subagent both starting job "1" and overwriting each other's files.
    a, b = DockerShell(workdir=tmp_path), DockerShell(workdir=tmp_path)
    assert a.jobs_dir != b.jobs_dir and a._jobs_path() != b._jobs_path()
    assert a.jobs_dir.parent == b.jobs_dir.parent == tmp_path / ".jobs"
    assert a._jobs_path() == f"/workspace/.jobs/{a.jobs_dir.name}"
    a.close()
    assert not a.jobs_dir.exists() and b.jobs_dir.exists()
    b.close()


def test_closing_a_docker_shell_forgets_its_jobs(tmp_path):
    # Catches a resumed sandbox numbering its jobs on from ones whose container and files are gone.
    shell = DockerShell(workdir=tmp_path)
    shell.jobs["1"] = 0  # as bash(background=True) records a job
    shell.close()
    assert shell.jobs == {}
