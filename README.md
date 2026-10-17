# Jean-Code

A minimal coding agent for educational purposes, in your terminal or your browser. Built on
[OpenRouter](https://openrouter.ai), with DeepSeek V4.1 Flash by default.

![The Streamlit UI: the timeline of a turn, with a tool call open in the inspector and two subagents in the sidebar](docs/screenshot.png)

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
| Research reports | Runs researchers in the background and writes a cited report |
| Expert builds | An orchestrator takes a project through plan, PRD, spec and implementation |
| Skills | Learns a task from a folder in `skills/` |
| Cost control | Shows what a session costs and stops at a limit you set |
| Live view | Shows every thought, tool call and result, for every agent |
