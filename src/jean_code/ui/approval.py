"""Asking the user before a tool call runs, when the turn and the front-end are on different threads.

The agent calls `Approvals.ask` on the turn's thread (it's what a front-end passes to `Agent` as `approve`) and
waits there. The front-end sees the call in `waiting` from its own thread, asks the user, and calls `answer`.
"""

import threading


def denied(reason: str = "") -> str:
    """The result the model gets for a call the user refused."""
    return f"Denied by user. {reason}".strip()


class Approvals:
    """One question at a time: the call the turn is asking about, and the user's answer to it."""

    def __init__(self):
        self.waiting: tuple[str, dict] | None = None  # the call being asked about: (tool name, arguments)
        self._answered = threading.Event()
        self._refusal: str | None = None

    def ask(self, name: str, arguments: dict) -> str | None:
        """Wait for `answer` and return what it was given: None to run the call, or the result to use instead."""
        self._answered.clear()
        self.waiting = (name, arguments)
        self._answered.wait()
        return self._refusal

    def answer(self, refusal: str | None = None) -> None:
        """Let the waiting call run, or give the model `refusal` as its result instead. Nothing if no call waits."""
        if self.waiting is None:
            return
        self._refusal = refusal
        self.waiting = None
        self._answered.set()
