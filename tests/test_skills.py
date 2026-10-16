"""Tests for `Skills`: discovery, the listing, the two tools and the `/name` commands, then how `Agent` wires them in.

Each test names the bug it would catch.
"""

import json

import pytest

from conftest import FakeClient, response
from jean_code.core.agent import Agent
from jean_code.core.skills import Skills
from jean_code.core.tool_schema import is_tool


@pytest.fixture
def skills_dir(tmp_path):
    """Two skills: `demo` (named by its frontmatter, with a reference file) and `plain` (named by its folder)."""
    demo = tmp_path / "demo-folder"
    (demo / "reference").mkdir(parents=True)
    (demo / "SKILL.md").write_text("---\nname: demo\ndescription: A demo.\n---\nDEMO BODY\n")
    (demo / "reference" / "notes.md").write_text("NOTES")
    (demo / ".hidden").write_text("HIDDEN")
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "SKILL.md").write_text("PLAIN BODY\n")
    (tmp_path / "no-skill").mkdir()  # no SKILL.md: not a skill
    (tmp_path / "secret.txt").write_text("SECRET")  # outside every skill folder
    return tmp_path


def test_discovery_names_skills_by_frontmatter_then_folder(skills_dir):
    # Catches a folder name winning over the frontmatter, or a folder without SKILL.md listed as a skill.
    skills = Skills(skills_dir)
    assert sorted(skills.catalog) == ["demo", "plain"]
    assert skills.catalog["demo"].path == skills_dir / "demo-folder"


def test_prompt_lists_names_and_descriptions_and_is_empty_without_skills(skills_dir):
    prompt = Skills(skills_dir).prompt()
    assert prompt.startswith("## Skills\n")
    assert "- demo: A demo." in prompt and "- plain: " in prompt
    assert "DEMO BODY" not in prompt
    assert Skills(None).prompt() == ""


def test_tools_are_marked_and_absent_without_skills(skills_dir):
    # Catches tools the agent would refuse as unmarked, or tools offered with nothing to load.
    tools = Skills(skills_dir).tools()
    assert sorted(tools) == ["load_skill", "read_skill_file"]
    assert all(is_tool(fn) for fn in tools.values())
    assert Skills(None).tools() == {}


def test_load_skill_returns_the_body_every_time_and_records_the_name_once(skills_dir):
    # Catches an "already loaded" refusal, which would block a reload after compaction summarized the body away.
    skills = Skills(skills_dir)
    first = skills.load_skill("demo")
    assert first.startswith("DEMO BODY")
    assert "- reference/notes.md" in first and ".hidden" not in first
    assert skills.load_skill("demo") == first
    assert skills.loaded == ["demo"]


def test_load_skill_refuses_an_unknown_name_and_lists_the_known_ones(skills_dir):
    with pytest.raises(ValueError, match="unknown skill 'nope'; available: demo, plain"):
        Skills(skills_dir).load_skill("nope")


def test_read_skill_file_reads_bundled_files_only(skills_dir):
    # Catches a path that escapes the skill's folder.
    skills = Skills(skills_dir)
    assert skills.read_skill_file("demo", "reference/notes.md") == "NOTES"
    for path in ("../secret.txt", "missing.md", "reference"):
        with pytest.raises(ValueError, match="is not a file in skill 'demo'"):
            skills.read_skill_file("demo", path)


def test_commands_load_their_skill_and_ignore_the_rest_of_the_line(skills_dir):
    # Catches the late-binding closure bug: every command loading the last skill.
    skills = Skills(skills_dir)
    commands = skills.commands()
    assert sorted(commands) == ["demo", "plain"]
    assert commands["demo"]("convert 5 miles").startswith("DEMO BODY")
    assert commands["plain"]("").startswith("PLAIN BODY")
    assert skills.loaded == ["demo", "plain"]


# ---------------------------------------------------------------- wired into Agent


def load_call(id: str, name: str) -> dict:
    """A `load_skill` tool call; conftest's `call` can't take an argument called `name`."""
    return {"id": id, "type": "function", "function": {"name": "load_skill", "arguments": json.dumps({"name": name})}}


def test_agent_runs_load_skill_calls_and_returns_errors_to_the_model(skills_dir):
    # Catches the skill tools missing from the agent's tools, or an unknown name crashing the turn.
    client = FakeClient(
        [
            response(tool_calls=[load_call("c1", "demo"), load_call("c2", "nope")]),
            response("done"),
        ]
    )
    agent = Agent(client=client, skills_dir=skills_dir)
    assert agent.run_turn("hi") == "answered"
    results = [m.content for m in agent.messages if m.role == "tool"]
    assert results[0].startswith("DEMO BODY")
    assert results[1].startswith("Error: ValueError: unknown skill 'nope'")
    assert agent.skills.loaded == ["demo"]
    assert "## Skills" in client.requests[0]["messages"][0]["content"]


def test_a_slash_command_puts_the_body_in_the_user_message(skills_dir):
    agent = Agent(client=FakeClient([response("ok")]), skills_dir=skills_dir)
    agent.run_turn("/demo convert 5 miles")
    user = [m for m in agent.messages if m.role == "user"][-1].content
    assert user.startswith('/demo convert 5 miles\n\n<command name="demo">\nDEMO BODY')
    assert agent.skills.loaded == ["demo"]


def test_an_agent_without_skills_has_no_skill_tools_listing_or_commands():
    client = FakeClient([response("ok")])
    agent = Agent(client=client, system_prompt="sys")
    agent.run_turn("/demo hi")
    request = client.requests[0]
    assert "tools" not in request
    assert request["messages"][0]["content"] == "sys"
    assert request["messages"][-1]["content"] == "/demo hi"
    assert agent.commands == {}
