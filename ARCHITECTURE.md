# Architecture notes

- **Most classes take their collaborators through dependency injection**, so each subcomponent can be tested
  on its own with a fake in its place. For example, `Agent` takes its `client` (a fake model in the tests) and its
  `node` (its place in the work tree) as constructor arguments instead of building them itself.
- **OpenRouter backend.** Every model call goes through one chat-completions client (`core/llm.py`), so any
  OpenRouter model works.
- **Streaming or plain completions.** `ChatClient.complete` returns the whole reply at once;
  `StreamingClient.stream` yields it as it's written, then the same parsed `Completion`. Every agent the
  builders in `agents/` make streams, subagents included; `Agent` itself streams only with `stream=True`.
- **Logging subscribes to the view; the core writes no files.** A `Recorder` (`logs.py`) subscribes to a tree's
  `AgentView` before the agent is built and writes every event, on its own thread, to
  `logs/processes/<run>/chat-<n>/`, one folder per agent nested like the tree (`1-research/2-researcher/`), and
  each model call to `logs/usage.jsonl`. `replay` folds a trace back into the state the view held.
- **Clients read only the event stream.** A UI drives the tree through a `Driver`'s commands and reads everything
  it shows from `driver.view`, which is built only from events; the driver doesn't hand out its agent, and
  `SLF001` holds `ui/` to that. The tests check `replay(trace) == view.state()`, so all of the state is in the
  events.
- **Parsing at the boundary.** Responses are parsed into typed messages as they come in (`core/llm.py`), so a
  malformed one raises `MalformedResponse` at once instead of failing later deep in the agent. Malformed tool
  arguments go back to the model as the tool's result, so it can retry.
- **Errors go back to the model.** A failing tool call or subagent comes back to the model as its result, so it can
  adapt. The model gets the error as one line; the full stack goes in the trace. Each agent records every exception
  its code catches as an `ErrorCaught` event in its own `events.jsonl`: where it swallows one (a tool's error
  result, a failed summary), or where one leaves it (`run_turn`, `compact`, `close`). Its callers record nothing.
