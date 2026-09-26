# jean-code

A small coding agent that works in your shell and in your browser. Built on [OpenRouter](https://openrouter.ai) (DeepSeek V4.1 Flash by default) - Supports all the goodies you're used to in agents: bash, tools, web-browsing, subagents, etc.

![The Streamlit UI: the timeline of a turn, with a tool call open in the inspector and two subagents in the sidebar](docs/screenshot.png)

## Running it

You need [uv](https://docs.astral.sh/uv/), and Docker if you want the sandbox.

**1. Add your OpenRouter key.** Create a key at <https://openrouter.ai/keys>, then put it in a `.env` file at the
repo root (it is gitignored):

```sh
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env
```

**2. Start a front-end.** `uv run` installs the dependencies on first use.

```sh
# Streamlit UI
uv run --env-file .env streamlit run src/app.py

# Terminal REPL
uv run --env-file .env python src/repl.py
```

REPL options: `--sandbox` runs commands in Docker, `--no-network` cuts the container off from the network,
`--model` takes any OpenRouter model id, `--reasoning low|medium|high`, `--max-cost` stops the session at a USD
limit, and `--compact-at` sets the token count that triggers compaction. Type `/cost` to see what the session has
spent, and `exit` to quit.

**3. (Optional) Build the sandbox image:**

```sh
docker build -t jean-code-sandbox sandbox/
# if your user id isn't 1000:
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t jean-code-sandbox sandbox/
```

## Key features

- **Runs commands for you**, on your machine with your approval, or in a Docker sandbox without asking.
- **Searches the web** and reads pages when it needs documentation or answers.
- **Delegates to subagents**: a web researcher, and a coding agent that experiments in the sandbox while the main
  one keeps working.
- **Skills**: drop instructions and reference files into `skills/` to teach it how to do a task.
- **Long sessions**: when the conversation gets too long, it summarizes the older turns and carries on.
- **Cost control**: shows what each session costs, and stops at a spending limit you set.
- **See what it's doing**: the UI shows every thought, tool call and result live, for the main agent and each subagent.

## Layout

```
src/
  agent.py      the agent loop: tool calls, approvals, compaction
  coding.py     the coding agent and its subagent profiles
  shell.py      host and Docker shell backends
  subagents.py  starting, waiting for and managing subagent runs
  skills.py     skill discovery
  tools.py      web tools
  llm.py        OpenRouter client, and tool schemas built from functions
  usage.py      cost records, logging and budgets
  repl.py       terminal front-end
  app.py        Streamlit front-end
skills/         skill folders
sandbox/        Dockerfile for the sandbox
```

Run the tests with `uv run pytest`.
