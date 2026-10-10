# agent-responder

A lightweight agentic harness for personal use, primarily to act as a
automated responder to triggers. Obviously more polished harnesses exist
that can serve that purpose but they all have tended to be susceptible to
feature bloat and instability from vibe-coding.

Additionally, I just wanted to know that I could.

This harness has basic python defined tools that can be called by supporting
models.


## Quickstart

```sh
uv sync --group dev        # create .venv, install deps, write uv.lock
cp .env.example .env       # then set AGENT_API_KEY (or export OPENAI_API_KEY)

uv run agent tools         # list the built-in example tools
uv run agent chat "What is 2 + 3?"
uv run agent chat          # interactive REPL
uv run agent web             # edit the config in a local web UI
```

Run the checks:

```sh
uv run pytest -q
uv run ruff check .
uv run mypy
```


## Basic Overview

`Agent.run(user_input)` adds the prompt to a created long-running `Conversation`, then loops:

1. Send `system` + history to the model (along with tool schemas).
2. If the model requests tool calls, run each tool, append the results, and loop.
3. If the model returns a plain message, append it and return it as the answer.

The loop stops when the model stops calling tools or `max_steps` is reached.
Tool errors are returned to the model as strings rather than raising, so the
agent can recover.
Each chat session is appended to a JSONL log under the active agent's persona
directory (`personas/<name>/sessions/`), capturing the user prompts, assistant
replies, tool calls, and tool results.

## Configuration

Settings load with this precedence: **environment variable > config file >
built-in default**. The file is optional.

- **Config file**: `$AGENT_RESPONDER_HOME/config.yaml` (default home
  `$HOME/.agent-responder/config.yaml`). See `config.yaml.example` for a template.
- **Env vars**: `AGENT_PROVIDER`, `AGENT_MODEL`, `AGENT_BASE_URL`,
  `AGENT_API_KEY` (or `OPENAI_API_KEY`), `AGENT_TIMEOUT`.
  `AGENT_RESPONDER_HOME` overrides the config home directory.
- **Env interpolation**: config values may use `$NAME` or `${NAME}`; set
  variables are substituted, unset ones are left as-is.

The file is sectioned:

- `providers:` — named endpoints. Each has `base_url`, `api_key`, `timeout`,
  and a `models` list (the models you can switch among).
- `agents:` — a list of named agents. Each picks a `provider` and `model`,
  plus an optional `tools` list (opt-in: omit for all built-in tools, `[]` for
  none, or name a subset) and an optional `persona` (a key into `personas:`).
- `personas:` — names a persona. Each named persona gets a directory under
  `AGENT_RESPONDER_HOME` -- `personas/<name>/` -- holding its `SOUL.md` plus a
  `sessions/` directory of conversation logs. The `SOUL.md` value seeds that file
  on first run and becomes the system prompt for any agent that references it.
- `interaction:` — names which agent each interaction uses; `chat.agent` is
  the agent the `agent chat` loop runs.

`Config.active_agent()` resolves the agent the loop uses (the one named by
`interaction.chat.agent`, else the first agent, else a default), and
`Config.endpoint()` resolves that into the effective endpoint.
New features add their own section (see `Config` in `config.py`).

## Web config UI

`agent web` serves a small local web page for viewing and editing the config
file (no extra dependencies; binds to `127.0.0.1` by default):

```sh
uv run agent web                 # http://127.0.0.1:8765/
uv run agent web --port 9000     # custom port
uv run agent web --host 0.0.0.0  # expose beyond localhost
```

The page has a navigation sidebar (currently just **Config**) and a top bar
showing the config file path with **Save** and **Refresh** buttons to the right.
Beneath it, providers (each with base URL, API key, timeout, and an add/remove
able list of models), agents (each with name, provider, model, opt-in tools, and
persona), personas (each with its `SOUL.md` and other file contents), and the
interaction (which agent the chat loop uses) are edited via form fields. Edits
write back to `config.yaml` (env refs like `$NAME` / `${NAME}` are preserved),
and on-disk changes are reflected automatically. A running `agent chat` session
is unaffected -- changes apply to the next session.

## Extending it

- **Add a tool**: implement `name` / `description` / `parameters` (JSON Schema)
  / `async run(**kwargs) -> str`, then `registry.register(MyTool())`.
- **Use a different model**: implement the `LLMClient` protocol
  (`async chat(...) -> LLMResponse`, `async close()`) and pass it to `Agent`.
  Point the default client at any OpenAI-compatible endpoint via
  `AGENT_BASE_URL` / `AGENT_MODEL`.
