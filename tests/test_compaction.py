"""Tests for context compaction: `Agent.compact()` and the end-of-turn trigger.

Each test names the bug it would catch. The scripted usage decides when the context is over `compact_at`;
the real estimator decides each part's share of it.

The standard session: keep_first=2, keep_last=2, compact_at=1000. Turns 1-4 are measured small; turn 5's reply
measures 5000 tokens, so compaction runs when turn 5 ends. Turns 1-2 are the head (then the marker),
turn 3 is the middle, turns 4-5 are the tail. Turn 3's reply carries BULK (~4700 estimated tokens), so the
middle is most of the context and compacting it helps.
"""

import pytest

import agent as agent_module
from agent import Agent, estimate_tokens, is_synthetic
from conftest import FakeClient, call, response, usage
from prompts import COMPACTION_PROMPT, EARLIER_SUMMARY_NOTE, HEAD_MARKER_TAG, SUMMARY_TAG


def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def rm(path: str) -> str:
    """Remove a file.

    Args:
        path: The file.
    """
    return f"removed {path}"


BULK = " filler" * 2000
SMALL = usage(prompt=100, completion=5, cost=0.001)
BIG = usage(prompt=5000, completion=5, cost=0.001)


def summary(text="SUMMARY TEXT", **kwargs) -> dict:
    return response(text, usage=usage(prompt=99_999, completion=300, cost=0.002), **kwargs)


def make_agent(responses, **kwargs) -> tuple[Agent, FakeClient]:
    client = kwargs.pop("client", None) or FakeClient(responses)
    kwargs = {"system_prompt": "sys", "tools": {"add": add, "rm": rm}, "compact_at": 1000,
              "keep_first": 2, "keep_last": 2, **kwargs}
    return Agent(client=client, **kwargs), client


def run_small_turns(agent: Agent, n: int, start: int = 1, bulky: int = 3) -> None:
    """Turns measured small; turn number `bulky` gets a long reply."""
    for i in range(start, start + n):
        agent.client.responses.append(response(f"reply {i}" + (BULK if i == bulky else ""), usage=SMALL))
        agent.run_turn(f"turn {i}")


def is_summary_request(request: dict) -> bool:
    return request["messages"][-1]["content"].startswith(COMPACTION_PROMPT[:40])


def assert_valid(messages: list[dict]) -> None:
    """Every tool result follows the assistant message holding its call, past only sibling results, and every
    call before the last assistant message has a result."""
    for i, m in enumerate(messages):
        if m["role"] != "tool":
            continue
        j = i - 1
        while messages[j]["role"] == "tool":
            j -= 1
        assert messages[j]["role"] == "assistant", f"tool result at {i} doesn't follow a call"
        assert m["tool_call_id"] in {c["id"] for c in messages[j].get("tool_calls", [])}
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    for m in messages[:-1]:
        for c in m.get("tool_calls", []):
            assert c["id"] in answered, f"call {c['id']} has no result"


def compacted_session(**kwargs) -> tuple[Agent, FakeClient]:
    """Turns 1-4 small, turn 5 big, then the summary: one compaction when turn 5 ends."""
    agent, client = make_agent([], **kwargs)
    run_small_turns(agent, 4)
    client.responses += [response("reply 5", usage=BIG), summary()]
    agent.run_turn("turn 5")
    return agent, client


# ---------------------------------------------------------------- the summary request


def test_summary_request_extends_the_cached_prefix():
    # Catches a changed prefix (rebuilt tools, a trimmed transcript): a full-price 256k-token read on every
    # compaction, and without `tools` DeepSeek silently drops the reasoning the summary needs.
    agent, client = compacted_session(reasoning="low")
    last_step, request = client.requests[-2], client.requests[-1]
    assert is_summary_request(request)
    assert request["messages"][: len(last_step["messages"])] == last_step["messages"]
    added = request["messages"][len(last_step["messages"]):]
    assert [m["role"] for m in added] == ["assistant", "user"]  # turn 5's reply, then the instruction
    assert added[0]["content"] == "reply 5"
    for key in ("model", "tools", "reasoning"):
        assert request[key] == last_step[key]
    # tool_choice "none" makes DeepSeek drop the tools, and with them the reasoning and the cache hit
    assert "tool_choice" not in request


def test_instruction_names_the_region():
    # Catches a prompt that points the model at the wrong turns.
    agent, client = compacted_session()
    instruction = client.requests[-1]["messages"][-1]["content"]
    assert HEAD_MARKER_TAG in instruction
    assert "2nd-to-last user message before this one" in instruction
    assert '"turn 4"' in instruction
    assert EARLIER_SUMMARY_NOTE not in instruction


# ---------------------------------------------------------------- the rewritten transcript


def test_transcript_is_head_marker_summary_tail():
    # Catches the wrong turns removed, or the summary missing its wrapper.
    agent, _ = compacted_session()
    contents = [m["content"] for m in agent.messages]
    assert contents[:5] == ["sys", "turn 1", "reply 1", "turn 2", "reply 2"]
    assert contents[5].startswith(HEAD_MARKER_TAG)
    assert contents[6].startswith(SUMMARY_TAG) and "SUMMARY TEXT" in contents[6]
    assert "no new requests" in contents[6]
    assert contents[7:] == ["turn 4", "reply 4", "turn 5", "reply 5"]
    assert agent.head_end == 6


def test_kept_messages_are_the_same_objects():
    # Catches copies: app.py matches messages by id(), so it would show everything as compacted and duplicated.
    agent, _ = compacted_session()
    before = agent.compactions[0]["before"]
    assert all(a is b for a, b in zip(agent.messages[:6], before[:6]))
    assert all(a is b for a, b in zip(agent.messages[-4:], before[-4:]))


def test_tool_calls_stay_with_their_results():
    # Catches a cut inside a turn: the API rejects the next request with a 400, and every one after it.
    agent, client = make_agent([])
    run_small_turns(agent, 2)
    for i in (3, 4):  # several rounds and parallel calls per turn
        client.responses += [
            response(tool_calls=[call("add", a=1, b=2), call("add", a=3, b=4)], usage=SMALL),
            response(tool_calls=[call("add", a=5, b=6)], usage=SMALL),
            response(f"reply {i}" + (BULK if i == 3 else ""), usage=SMALL),
        ]
        agent.run_turn(f"turn {i}")
    client.responses += [response(tool_calls=[call("add", a=7, b=8)], usage=SMALL),
                         response("reply 5", usage=BIG), summary()]
    agent.run_turn("turn 5")
    assert agent.compactions[-1]["status"] == "compacted"
    assert_valid(agent.messages)
    assert sum(m["role"] == "tool" for m in agent.messages) == 3 + 1  # turn 4's three results and turn 5's one


def test_context_history_keeps_each_earlier_context():
    # Catches the pre-compaction context being lost, or modified in place.
    agent, _ = compacted_session()
    history = agent.context_history
    assert len(history) == 2
    assert [m["content"][:7] for m in history[0]][6:8] == ["turn 3", "reply 3"]
    assert history[1] is agent.messages


# ---------------------------------------------------------------- when it runs


def test_not_inside_a_tool_loop():
    # Catches compaction in the middle of a turn, which cuts the reasoning trace of a task in progress.
    agent, client = make_agent([])
    run_small_turns(agent, 4)
    client.responses += [
        response(tool_calls=[call("add", a=1, b=2)], usage=BIG),  # over the threshold mid-turn
        response(tool_calls=[call("add", a=3, b=4)], usage=BIG),
        response("reply 5", usage=BIG),
        summary(),
    ]
    agent.run_turn("turn 5")
    kinds = [is_summary_request(r) for r in client.requests[-4:]]
    assert kinds == [False, False, False, True]


def test_not_while_a_call_waits_for_approval():
    # Catches a summary request ending in a call without a result (400).
    agent, client = make_agent([], needs_approval={"rm"})
    run_small_turns(agent, 4)
    client.responses += [response(tool_calls=[call("rm", path="x")], usage=BIG)]
    agent.run_turn("turn 5")
    assert agent.pending_calls and not any(map(is_summary_request, client.requests))
    assert agent.compact(force=True).startswith("refused")
    assert not any(map(is_summary_request, client.requests))
    client.responses += [response("reply 5", usage=BIG), summary()]
    agent.approve()  # the turn ends now, so it compacts
    assert is_summary_request(client.requests[-1])
    assert_valid(agent.messages)


def test_not_after_an_interrupt():
    # Catches a paid summary call right after the user said stop.
    agent, client = make_agent([])
    run_small_turns(agent, 4)

    class Interrupting(FakeClient):
        def complete(self, request):
            agent.interrupt()
            return super().complete(request)

    agent.client = Interrupting([response("reply 5", usage=BIG), summary()])
    agent.run_turn("turn 5")
    assert len(agent.client.requests) == 1 and agent.compactions == []


def test_interrupt_during_the_summary_call_cancels_it():
    # Catches a summary applied after the user interrupted it.
    agent, client = make_agent([])
    run_small_turns(agent, 4)

    class InterruptingSummary(FakeClient):
        def complete(self, request):
            if is_summary_request(request):
                agent.interrupt()
            return super().complete(request)

    agent.client = InterruptingSummary([response("reply 5", usage=BIG), summary()])
    before = agent.messages
    agent.run_turn("turn 5")
    assert agent.messages is before and agent.compactions[-1]["status"] == "failed: interrupted"


@pytest.mark.parametrize("turns", [2, 3, 4])
def test_nothing_to_compact_without_a_middle(turns):
    # Catches a paid call with nothing to summarize: N+M turns or fewer have no middle.
    agent, client = make_agent([])
    run_small_turns(agent, turns)
    assert agent.compact(force=True) == "nothing to compact"
    assert not any(map(is_summary_request, client.requests))


def test_an_earlier_summary_alone_is_not_compacted_again():
    # Catches summarizing the summary, a full-context call every time a turn ends over the threshold.
    # Right after a compaction the middle is only the summary (turns 4-5 are the tail).
    agent, client = compacted_session()
    assert agent.compact(force=True) == "nothing to compact"
    assert sum(map(is_summary_request, client.requests)) == 1


def test_skipped_when_the_kept_turns_alone_are_over_the_threshold(monkeypatch):
    # Catches compacting when it can't help, then again at every turn: one full-context call each time.
    agent, client = make_agent([], compact_at=1000)
    run_small_turns(agent, 4)
    monkeypatch.setattr(agent_module, "estimate_tokens", lambda value: 200)  # head + tail alone ~ 2000
    client.responses += [response("reply 5", usage=BIG)]
    agent.run_turn("turn 5")
    assert agent.compactions[-1]["status"].startswith("skipped")
    assert not any(map(is_summary_request, client.requests))


def test_an_estimator_that_runs_high_does_not_block_compaction(monkeypatch):
    # Catches the kept part judged by the raw estimate: the estimator errs high by design, so real sessions
    # would skip compactions that help (seen in .experiments/output/repl_compaction_run.txt: ~2228 "kept"
    # against a measured context of ~1.8k).
    monkeypatch.setattr(agent_module, "estimate_tokens", lambda value: 5 * estimate_tokens(value))
    agent, _ = compacted_session()
    assert agent.compactions[-1]["status"] == "compacted"


def test_below_threshold_does_nothing_unless_forced():
    # Catches /compact doing nothing, or automatic compaction below the threshold.
    agent, client = make_agent([])
    run_small_turns(agent, 5)
    assert agent.compactions == [] and agent.compact() == "below threshold"
    client.responses.append(summary())
    assert agent.compact(force=True) == "compacted"


def test_after_compaction_the_old_measurement_is_scaled_down():
    # Catches the measurement not being reset (compaction in a loop), the summary call's 99,999-token prompt
    # taken for the new context, or a raw estimate that reports the context growing
    # (.experiments/output/repl_compaction_run3_raw_after.txt: "~2.1k -> ~2.7k", then 1.9k measured).
    agent, client = compacted_session()
    record = agent.compactions[-1]
    tools = estimate_tokens(client.requests[-1]["tools"])
    new = tools + sum(estimate_tokens(m) for m in agent.messages)
    old = tools + sum(estimate_tokens(m) for m in record["before"])
    assert agent.context_tokens == record["tokens_after"] == round(record["tokens_before"] * new / old)
    assert agent.context_tokens < agent.compact_at


# ---------------------------------------------------------------- failures leave the transcript alone


@pytest.mark.parametrize("reply, reason", [
    (RuntimeError("HTTP 502"), "RuntimeError"),
    (summary(""), "empty"),
    (summary(tool_calls=[call("add", a=1, b=2)]), "called a tool"),
    ({**summary(), "choices": [{**summary()["choices"][0], "finish_reason": "length"}]}, "cut off"),
    ({"id": "gen-x", "usage": usage(cost=0.002)}, "KeyError"),
])
def test_a_failed_summary_changes_nothing(reply, reason):
    # Catches an empty, truncated or missing summary replacing turns of work.
    agent, client = make_agent([])
    run_small_turns(agent, 4)
    client.responses += [response("reply 5", usage=BIG), reply]
    agent.run_turn("turn 5")
    before = agent.messages
    assert agent.compactions[-1]["status"].startswith("failed") and reason in agent.compactions[-1]["status"]
    assert agent.messages is before and [m["content"] for m in before][-2:] == ["turn 5", "reply 5"]
    assert agent.context_history == [before]


def test_second_compaction_folds_the_first_summary_in():
    # Catches summaries piling up, or the head moving.
    agent, client = compacted_session()
    head = agent.messages[:6]
    run_small_turns(agent, 2, start=6, bulky=6)  # turn 6 ends up in the middle with the first summary
    client.responses += [response("reply 8", usage=BIG), summary("SECOND")]
    agent.run_turn("turn 8")
    assert EARLIER_SUMMARY_NOTE in client.requests[-1]["messages"][-1]["content"]
    assert all(a is b for a, b in zip(agent.messages[:6], head))
    synthetic = [m["content"] for m in agent.messages if is_synthetic(m)]
    assert len(synthetic) == 2 and "SECOND" in synthetic[1] and "SUMMARY TEXT" not in synthetic[1]
    assert [m["content"] for m in agent.messages[7:]] == ["turn 7", "reply 7", "turn 8", "reply 8"]
    assert len(agent.context_history) == 3


# ---------------------------------------------------------------- budget, skills, hard limit


def test_summary_call_is_billed_and_respects_the_budget():
    # Catches the most expensive call of a session left out of the budget, or made after it's spent.
    agent, _ = compacted_session()
    assert agent.usage[-1]["kind"] == "compaction"
    assert agent.total_cost == pytest.approx(5 * 0.001 + 0.002)

    agent, client = make_agent([], max_cost=0.005)
    run_small_turns(agent, 4)
    client.responses += [response("reply 5", usage=BIG)]  # the 5th $0.001: the budget is spent
    agent.run_turn("turn 5")
    assert not any(map(is_summary_request, client.requests))


def test_a_skill_summarized_away_can_be_loaded_again(tmp_path):
    # Catches "already loaded" for instructions the model can no longer see.
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "SKILL.md").write_text("---\nname: demo\ndescription: A demo.\n---\nDEMO BODY\n")
    agent, client = make_agent([], skills_dir=tmp_path)
    run_small_turns(agent, 2)
    client.responses.append(response("reply 3" + BULK, usage=SMALL))
    agent.run_turn("/demo")  # turn 3: its body goes into the user message, which ends up in the middle
    assert agent.loaded_skills == ["demo"]
    run_small_turns(agent, 1, start=4)
    client.responses += [response("reply 5", usage=BIG), summary()]
    agent.run_turn("turn 5")
    assert agent.loaded_skills == []
    assert "DEMO BODY" in agent.load_skill("demo")


def test_hard_limit_stops_the_turn():
    # Catches a turn running on until the provider rejects the request mid-turn.
    agent, client = make_agent([response(tool_calls=[call("add", a=1, b=2)], usage=usage(prompt=1900))],
                               compact_at=None, context_window=2000)
    agent.run_turn("go")
    assert len(client.requests) == 1
    assert agent.interrupted and "nearly full" in agent.stop_reason


def test_compact_at_must_leave_room():
    # Catches a threshold so close to the window that the summary call itself can't fit.
    with pytest.raises(ValueError):
        Agent(client=FakeClient([]), compact_at=600_000)
