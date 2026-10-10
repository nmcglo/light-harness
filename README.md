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
- `agent:` — picks the `provider` and `model` the loop uses, plus an optional
  `tools` list (opt-in): omit for all built-in tools, `[]` for none, or name a
  subset to enable.

`Config.endpoint()` resolves those into the effective endpoint for the loop.
New features add their own section (see `Config` in `config.py`).

## Extending it

- **Add a tool**: implement `name` / `description` / `parameters` (JSON Schema)
  / `async run(**kwargs) -> str`, then `registry.register(MyTool())`.
- **Use a different model**: implement the `LLMClient` protocol
  (`async chat(...) -> LLMResponse`, `async close()`) and pass it to `Agent`.
  Point the default client at any OpenAI-compatible endpoint via
  `AGENT_BASE_URL` / `AGENT_MODEL`.
