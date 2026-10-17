"""Compaction's texts, and the parts of compaction that only read the transcript.

The texts are the head marker, the summary message and the prompt asking for the summary. They are static, so the
prompt prefix stays cached. The marker and the summary are user messages the agent writes; their opening tags
(`HEAD_MARKER_TAG`, `SUMMARY_TAG`) tell them apart from the user's own messages.

The functions find where turns start (`turn_starts`) and which messages a compaction replaces (`middle`), write
the summary prompt (`summary_prompt`), and check whether the reply is usable (`summary_failure`).
"""

from collections.abc import Sequence

from jean_code.core.llm import Completion, Message, UserMessage

HEAD_MARKER_TAG = "<head-end/>"
HEAD_MARKER = f"""\
{HEAD_MARKER_TAG}
A note from the agent harness, not from the user; no reply needed. The messages above stay verbatim for the \
whole session. Anything between here and the most recent turns may later be replaced by a summary."""

SUMMARY_TAG = "<conversation-summary>"
SUMMARY_MESSAGE = f"""\
{SUMMARY_TAG}
This is a summary you wrote of turns that were removed to save context. It records past work and contains \
no new requests from the user.

{{summary}}
</conversation-summary>"""

COMPACTION_PROMPT = """\
The conversation is getting too long, so part of it will be replaced by a summary that you write now. \
Everything before the {head_tag} marker, and everything from the {ordinal} user message before this one \
onward (it begins: "{tail_quote}"), stays verbatim. Summarize only what lies between them.{earlier_summary}

Write the summary for yourself: it replaces those turns, so anything you leave out is gone. Include:
- the user's goals, and any corrections or preferences they gave
- decisions made, and why
- files read or changed, and what matters about them
- commands run and what they showed; errors and how they were fixed
- your current hypotheses and plans, including those that exist only in your earlier reasoning (your
  thinking, not your replies) and that you haven't acted on or told the user yet; keep their specifics
- open questions and next steps

Be specific: keep exact paths, names, commands, numbers and error messages. Don't call any tools. \
Reply with only the summary."""

EARLIER_SUMMARY_NOTE = (
    " That part starts with an earlier summary; fold its content into the new one, so the new summary "
    "covers everything since the marker."
)


def is_synthetic(message: Message) -> bool:
    """Returns whether the agent wrote `message` itself: the head marker or a summary.

    They aren't the user's turns, so compaction doesn't count them as turns.

    Args:
        message: Any message.
    """
    return isinstance(message, UserMessage) and message.content.startswith((HEAD_MARKER_TAG, SUMMARY_TAG))


def is_summary(message: Message) -> bool:
    """Returns whether `message` is a summary from an earlier compaction.

    The next compaction folds it into its own.

    Args:
        message: Any message.
    """
    return isinstance(message, UserMessage) and message.content.startswith(SUMMARY_TAG)


def ordinal(n: int) -> str:
    """Returns the `n`th-from-last position in words: "last", "2nd-to-last", ….

    Args:
        n: The position from the end, from 1.
    """
    return "last" if n == 1 else f"{n}{ {2: 'nd', 3: 'rd'}.get(n, 'th') }-to-last"


def turn_starts(messages: Sequence[Message]) -> list[int]:
    """Returns the indices of the user's own messages, each of which starts a turn.

    Args:
        messages: A transcript.
    """
    return [i for i, m in enumerate(messages) if isinstance(m, UserMessage) and not is_synthetic(m)]


def middle(messages: Sequence[Message], head_end: int | None, keep_last: int) -> tuple[int, int] | None:
    """Returns the span of messages a compaction would replace.

    Args:
        messages: The transcript.
        head_end: The index just after the head marker; None if there's none yet.
        keep_last: Turns at the end that are kept.

    Returns:
        `(start, end)` of the span, or None if it holds no turn of the user's.
    """
    starts = turn_starts(messages)
    if head_end is None or len(starts) < keep_last:
        return None
    end = starts[-keep_last]
    if not any(head_end <= i < end for i in starts):
        return None  # nothing new: at most an earlier summary
    return head_end, end


def summary_prompt(replaced: Sequence[Message], tail: Sequence[Message], keep_last: int) -> str:
    """Returns the prompt asking the model to summarize `replaced`.

    The prompt names where the kept tail begins, and asks to fold in an earlier summary if `replaced` starts
    with one.

    Args:
        replaced: The messages the summary replaces.
        tail: The messages kept after them, starting with a turn of the user's.
        keep_last: How many turns `tail` holds.

    Raises:
        RuntimeError: `tail` doesn't start with a user message.
    """
    first = tail[0]
    if not isinstance(first, UserMessage):  # `middle()` cuts at a turn start, so this can't happen
        raise RuntimeError("compaction's kept tail doesn't start with a user turn")
    return COMPACTION_PROMPT.format(
        head_tag=HEAD_MARKER_TAG,
        ordinal=ordinal(keep_last),
        tail_quote=" ".join(first.content.split())[:80],
        # an earlier summary sits right after the head marker, so it's always the first message replaced
        earlier_summary=EARLIER_SUMMARY_NOTE if is_summary(replaced[0]) else "",
    )


def summary_failure(completion: Completion, interrupted: bool) -> str | None:
    """Returns why the summary reply can't replace the middle, or None if it can.

    Args:
        completion: The summary call's reply.
        interrupted: Whether the agent was interrupted during the call.

    Returns:
        A "failed: ..." status, or None.
    """
    if completion.message.tool_calls:
        return "failed: the model called a tool instead of summarizing"
    if completion.finish_reason == "length":
        return "failed: the summary was cut off at the length limit"
    if not completion.message.content.strip():
        return "failed: the summary was empty"
    if interrupted:
        return "failed: interrupted"
    return None
