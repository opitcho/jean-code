# Architecture notes


- **Most classes take their collaborators through dependency injection**, so each subcomponent can be tested
  on its own with a fake in its place. For example, `Agent` takes its `client` (a fake model in the tests) and its
  `node` (its place in the agent tree) as constructor arguments instead of building them itself.
- **OpenRouter backend.** Every model call goes through one chat-completions client (`core/llm.py`), so any
  OpenRouter model works.
- **Streaming or plain completions.** `ChatClient.complete` returns the whole reply at once;
  `StreamingClient.stream` yields it as it's written, then the same parsed `Completion`. Only the agent the user
  talks to streams; subagents don't.
- **Logging subscribes to the view; the core writes no files.** A `Recorder` (`logs.py`) subscribes to a tree's
  `AgentView` before the agent is built and writes every event, on its own thread, to
  `logs/processes/<run>/chat-<n>/`, one folder per agent nested like the tree (`1-research/2-researcher/`), and
  each model call to `logs/usage.jsonl`. `replay` folds a trace back into the state the view held.
- **Parsing at the boundary.** Responses are parsed into typed messages as they come in (`core/llm.py`), so a
  malformed one raises `MalformedResponse` at once instead of failing later deep in the agent. Malformed tool
  arguments go back to the model as the tool's result, so it can retry.
- **Error Tracebacks** - Tool calls or subagents, errors fed back into context for model driving to be able to adapt.


- **Two Approaches**:

- Decoupling? --> What if harness fails?
