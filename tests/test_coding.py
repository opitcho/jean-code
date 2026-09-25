"""Tests for the coding agent's wiring: its subagent profiles, and Docker shells sharing a workspace.

Nothing here starts a container or calls a model. Each test names the bug it would catch.
"""

import pytest

from coding import coding_agent, general_agent, searcher_agent, subagent_profiles
from shell import DockerShell
from subagents import Run
from usage import Budget


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # builds a client; nothing here sends a request


def test_the_coding_agent_offers_both_subagents():
    # Catches profiles that exist but never reach the model: no listing, or no tools.
    agent = coding_agent(skills_dir=None)
    try:
        prompt = agent.messages[0]["content"]
        assert "## Subagents" in prompt and "- searcher: " in prompt and "- general: " in prompt
        assert {"spawn", "subagent"} <= set(agent._tools())
    finally:
        agent.close()
    assert coding_agent(skills_dir=None, subagents=False).subagents is None


def test_subagents_have_no_subagents_of_their_own():
    # Catches unbounded recursion: a general subagent that can spawn more general subagents.
    for build in (lambda b: searcher_agent("m", b), lambda b: general_agent("m", b, skills_dir=None)):
        agent = build(Budget(1.0))
        assert agent.subagents is None
        agent.close()


def test_the_general_subagent_runs_bash_in_docker_without_approval():
    # Catches a general subagent that Run.start refuses (host bash needs approval) or that runs on the host.
    budget = Budget().child(0.5, "1")
    agent = general_agent("m", budget, skills_dir=None)
    try:
        assert agent.needs_approval == set() and "bash" in agent.tool_map
        assert isinstance(agent.tool_map["bash"].__self__, DockerShell)
        assert agent.budget is budget  # spawn's check that the profile used its node
        assert "## You are a subagent" in agent.messages[0]["content"]
    finally:
        agent.close()


def test_profile_limits():
    profiles = subagent_profiles("m")
    assert (profiles["searcher"].max_cost, profiles["searcher"].cap) == (0.05, 0.25)
    assert (profiles["general"].max_cost, profiles["general"].cap) == (0.50, 2.0)


def test_a_searcher_run_starts():
    # Catches a profile whose agent Run.start refuses.
    agent = searcher_agent("m", Budget(1.0))
    agent.budget.cancel()  # makes no call: the run stops at once
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
