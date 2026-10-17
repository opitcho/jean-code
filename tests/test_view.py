"""Tests for `AgentView`: the state a tree's events fold into, and what a subscriber sees.

Each test names the bug it would catch.
"""

from jean_code.core.events import AgentAdded, Charged, Compaction, Delta, MessageAdded, TurnEnded, TurnStarted
from jean_code.core.llm import AssistantMessage, UserMessage
from jean_code.core.usage import ModelCall
from jean_code.core.view import AgentView, TreeState, apply

# ---------------------------------------------------------------- the reducer


def call_for(path: str, cost: float) -> ModelCall:
    return ModelCall("gen", None, "m", 0, cost, path)


def test_the_reducer_splices_a_compaction_and_keeps_the_replaced_items():
    # Catches compacted messages dropped from the timeline, or the summary shown after the turns that followed it.
    state = TreeState()
    apply(state, AgentAdded(path="main", profile="", task="", model="m"))
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
    assert [main.item(uid).message.content for uid in main.context] == ["t1", "SUMMARY", "t4"]
    assert main.context_tokens == 20 and len(main.compactions) == 1


def test_the_reducer_drops_a_draft_once_its_reply_lands():
    # Catches streamed text shown twice: as the draft and as the reply it became.
    state = TreeState()
    apply(state, AgentAdded(path="main", profile="", task="", model="m"))
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


def test_a_subscriber_sees_each_event_after_its_snapshot_once():
    # Catches an event the snapshot already holds delivered again, or one between them lost.
    view = AgentView()
    view.listen(AgentAdded(path="main", profile="", task="", model="m"))
    with view.subscribe() as (snapshot, events):
        view.listen(MessageAdded(path="main", message=UserMessage(content="hi"), context_tokens=1))
        assert list(snapshot.agents) == ["main"] and snapshot.agents["main"].items == []
        assert events.get(timeout=0).type == "message"
    assert view._subscribers == set()
