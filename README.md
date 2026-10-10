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

## Extending it

- **Add a tool**: implement `name` / `description` / `parameters` (JSON Schema)
  / `async run(**kwargs) -> str`, then `registry.register(MyTool())`.
- **Use a different model**: implement the `LLMClient` protocol
  (`async chat(...) -> LLMResponse`, `async close()`) and pass it to `Agent`.
  Point the default client at any OpenAI-compatible endpoint via
  `AGENT_BASE_URL` / `AGENT_MODEL`.
