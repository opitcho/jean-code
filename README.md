# Jean-Code

A fully fledged agent, in your terminal or your browser, with subagents, tool calling, skills, context compaction,
cost limits and tracing of every agent through an event stream. Every model call goes through
[OpenRouter](https://openrouter.ai), with DeepSeek V4.1 Flash by default.

![The Streamlit UI: a research turn's timeline (thinking, a skill load, web searches) and the reply, with the tree of research subagents in the sidebar](docs/screenshot.png)

## Getting started

You need [uv](https://docs.astral.sh/uv/) and an [OpenRouter key](https://openrouter.ai/keys) with some credits.

```sh
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env

# Browser UI, at http://localhost:8501
uv run --env-file .env streamlit run src/jean_code/ui/app.py

# or the terminal
uv run --env-file .env jean-code
```

For the Docker sandbox and the `general` subagent, build the image once:

```sh
docker build -t jean-code-sandbox sandbox/
```

## Features

| Feature | What it does |
|---|---|
| Shell | Runs commands on your machine with your approval, or in a Docker sandbox without asking |
| Web | Searches the web and reads pages |
| Subagents | Hands work to a web researcher, or a coding agent in its own sandbox |
| Research reports | Runs researchers in the background and writes a cited report, each quote checked against the page it came from |
| Expert builds | An orchestrator takes a project through a PRD, a spec and an implementation, then an auditor checks every step |
| Skills | Learns a task from a folder in `skills/` |
| Cost control | Shows what a session costs and stops at a limit you set |
| Context compaction | Summarizes the middle of a long conversation once it fills the context, keeping the first and last turns; `/compact` does it on demand |
| Interruption | Handles user interruption, and supports model steering |
| Error recovery | A failing tool, a malformed call or a crashed subagent comes back to the model as a result, so it can adapt instead of the turn ending |
| Live view | Shows every thought, tool call and result, for every agent |
| Complete tracing | Writes every agent's events to `logs/`, one file per agent nested like the tree, with each error's full traceback; a trace replays into exactly what the UI showed |
