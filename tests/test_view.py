"""Tests for `AgentView`: the state a tree's events fold into, and what a subscriber sees.

Each test names the bug it would catch.
"""

from conftest import agent_added
from jean_code.core.events import (
    Charged,
    Compaction,
    Delivered,
    Delta,
    GroupAdded,
    LimitSet,
    MessageAdded,
    RunEnded,
    SkillLoaded,
    TurnEnded,
    TurnStarted,
)
from jean_code.core.llm import AssistantMessage, UserMessage
from jean_code.core.usage import ModelCall
from jean_code.core.view import AgentView, GroupState, TreeState, apply, snapshot

# ---------------------------------------------------------------- the reducer


def call_for(path: str, cost: float) -> ModelCall:
    return ModelCall("gen", None, "m", 0, cost, path)


def test_the_reducer_splices_a_compaction_and_keeps_the_replaced_items():
    # Catches compacted messages dropped from the timeline, or the summary shown after the turns that followed it.
    state = TreeState()
    apply(state, agent_added("main"))
    for text in ("t1", "t2", "t3", "t4"):
        apply(state, MessageAdded(path="main", message=UserMessage(content=text), context_tokens=10))
    summary = UserMessage(content="SUMMARY")
    apply(
        state,
        Compaction(
            path="main", status="compacted", start=1, removed=2, summary=summary, tokens_before=40, context_tokens=20
        ),
    )
    main = state.agents["main"]
    assert [(i.message.content, i.in_context) for i in main.items] == [
        ("t1", True),
        ("t2", False),
        ("t3", False),
        ("SUMMARY", True),
        ("t4", True),
    ]
    assert [m.content for m in main.messages()] == ["t1", "SUMMARY", "t4"]
    assert main.context_tokens == 20 and len(main.compactions) == 1


def test_the_reducer_drops_a_draft_once_its_reply_lands():
    # Catches streamed text shown twice: as the draft and as the reply it became.
    state = TreeState()
    apply(state, agent_added("main"))
    apply(state, TurnStarted(path="main", text="go"))
    apply(state, Delta(path="main", kind="reasoning", text="hm"))
    apply(state, Delta(path="main", kind="content", text="he"))
    apply(state, Delta(path="main", kind="content", text="llo"))
    assert state.agents["main"].draft == {"reasoning": "hm", "content": "hello"} and state.agents["main"].running
    apply(state, MessageAdded(path="main", message=AssistantMessage(content="hello"), context_tokens=5))
    assert state.agents["main"].draft == {}
    apply(state, Charged(path="main", call=call_for("main", 0.1)))
    apply(state, TurnEnded(path="main", end="answered", context_tokens=5))
    assert not state.agents["main"].running and state.agents["main"].end == "answered"


def test_a_limit_change_reaches_its_agent_or_group_and_one_with_no_state_is_ignored():
    # Catches a raised budget the inspector never shows, or a crash on the limit of a spawn whose start failed.
    state = TreeState()
    apply(state, agent_added("main", limit=1.0))
    apply(state, GroupAdded(path="main/1", profile="pair", task="compare", limit=0.5))
    assert state.agents["main"].limit == 1.0
    apply(state, LimitSet(path="main", limit=2.0))
    apply(state, LimitSet(path="main/1", limit=0.7))
    apply(state, LimitSet(path="main/2", limit=0.5))
    assert state.agents["main"].limit == 2.0 and state.groups["main/1"].limit == 0.7
    assert list(state.agents) == ["main"] and list(state.groups) == ["main/1"]


def test_an_agents_call_counts_toward_its_group_and_the_root():
    # Catches a group whose spend stays at zero, or a call counted toward a group it isn't under.
    state = TreeState()
    apply(state, agent_added("main"))
    apply(state, GroupAdded(path="main/1", profile="pair", task="compare", limit=None))
    apply(state, GroupAdded(path="main/2", profile="pair", task="other", limit=None))
    apply(state, agent_added("main/1/1"))
    apply(state, Charged(path="main/1/1", call=call_for("main/1/1", 0.25)))
    assert state.groups["main/1"] == GroupState("main/1", "pair", "compare", None, spent=0.25)
    assert state.agents["main"].spent == 0.25 and state.groups["main/2"].spent == 0


def test_a_snapshot_keeps_its_groups_when_a_later_event_changes_them():
    # Catches a client's copy whose group spend moves under it.
    state = TreeState()
    apply(state, agent_added("main"))
    apply(state, GroupAdded(path="main/1", profile="pair", task="compare", limit=None))
    apply(state, agent_added("main/1/1"))
    copy = snapshot(state)
    apply(state, Charged(path="main/1/1", call=call_for("main/1/1", 0.25)))
    apply(state, LimitSet(path="main/1", limit=1.0))
    assert copy.groups["main/1"] == GroupState("main/1", "pair", "compare", None)


def test_an_agent_keeps_what_it_was_built_with():
    # Catches the inspector's tools, approval list or skill catalog missing from the view.
    state = TreeState()
    tools = [{"type": "function", "function": {"name": "rm"}}]
    apply(state, agent_added("main", tools=tools, needs_approval=["rm"], skills=["demo", "pdf"]))
    main = state.agents["main"]
    assert (main.tools, main.needs_approval, main.skills, main.loaded_skills) == (tools, ["rm"], ["demo", "pdf"], [])
    apply(state, SkillLoaded(path="main", name="pdf"))
    assert state.agents["main"].loaded_skills == ["pdf"]


def test_waiting_counts_ended_results_not_yet_delivered_per_parent():
    # Catches a result counted under the wrong parent, counted after delivery, or not counted again once resumed.
    state = TreeState()
    for path in ("main", "main/1", "main/2", "main/1/1"):
        apply(state, agent_added(path))
    apply(state, RunEnded(path="main/1", status="done"))
    apply(state, RunEnded(path="main/1/1", status="stopped", reason="budget spent"))
    assert (state.waiting("main"), state.waiting("main/1")) == (1, 1)
    apply(state, Delivered(path="main/1"))
    assert state.waiting("main") == 0
    apply(state, RunEnded(path="main/1", status="done"))  # a resumed run ended again
    assert state.waiting("main") == 1 and state.agents["main/1"].outcome.status == "done"


def test_a_subscriber_sees_each_event_after_its_snapshot_once():
    # Catches an event the snapshot already holds delivered again, or one between them lost.
    view = AgentView()
    view.listen(agent_added("main"))
    with view.subscribe() as (snapshot, events):
        view.listen(MessageAdded(path="main", message=UserMessage(content="hi"), context_tokens=1))
        assert list(snapshot.agents) == ["main"] and snapshot.agents["main"].items == []
        assert events.get(timeout=0).type == "message"
    assert view._subscribers == set()
