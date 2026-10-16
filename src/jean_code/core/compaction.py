"""Compaction's messages and markers, and the parts of compaction that only read the transcript: where the turns
start, which messages a compaction replaces, the prompt asking for the summary, and whether the reply is usable.

`Agent.compact` does the rest: the summary call, and swapping in the new transcript. The texts here are static,
so the prompt prefix stays cached. The marker and the summary are user messages the agent writes; their opening
tags tell them apart from the user's own messages.
"""

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
    """True for the user messages the agent writes itself: the head marker and summaries. They aren't the user's
    turns, so compaction doesn't count them as turns."""
    return isinstance(message, UserMessage) and message.content.startswith((HEAD_MARKER_TAG, SUMMARY_TAG))


def is_summary(message: Message) -> bool:
    """True for a summary from an earlier compaction, which the next compaction folds into its own."""
    return isinstance(message, UserMessage) and message.content.startswith(SUMMARY_TAG)


def ordinal(n: int) -> str:
    return "last" if n == 1 else f"{n}{ {2: 'nd', 3: 'rd'}.get(n, 'th') }-to-last"


def turn_starts(messages: list[Message]) -> list[int]:
    """Indices of the user's own messages, each of which starts a turn."""
    return [i for i, m in enumerate(messages) if isinstance(m, UserMessage) and not is_synthetic(m)]


def middle(messages: list[Message], head_end: int | None, keep_last: int) -> tuple[int, int] | None:
    """(start, end) of the messages a compaction would replace, or None if they hold no turn of the user's."""
    starts = turn_starts(messages)
    if head_end is None or len(starts) < keep_last:
        return None
    end = starts[-keep_last]
    if not any(head_end <= i < end for i in starts):
        return None  # nothing new: at most an earlier summary
    return head_end, end


def summary_prompt(replaced: list[Message], tail: list[Message], keep_last: int) -> str:
    """The prompt asking the model to summarize `replaced`, naming where the kept tail begins."""
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
    """Why the summary call's reply can't replace the middle, as a "failed: ..." status, or None if it can."""
    if completion.message.tool_calls:
        return "failed: the model called a tool instead of summarizing"
    if completion.finish_reason == "length":
        return "failed: the summary was cut off at the length limit"
    if not completion.message.content.strip():
        return "failed: the summary was empty"
    if interrupted:
        return "failed: interrupted"
    return None
