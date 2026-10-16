"""Compaction's messages and markers. Static text only, so the prompt prefix stays cached.

The marker and the summary are user messages the agent writes (see `Agent.compact`); their opening tags tell them
apart from the user's own messages.
"""

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
