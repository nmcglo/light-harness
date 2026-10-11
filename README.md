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
cp .env.example .env       # secrets referenced from config.yaml as $NAME; `agent` loads ./.env
cp config.yaml.example ~/.agent-responder/config.yaml   # then edit

uv run agent tools         # list the built-in example tools
uv run agent chat "What is 2 + 3?"
uv run agent chat          # interactive REPL
uv run agent web             # edit the config in a local web UI
uv run agent webhooks          # serve the configured webhook listeners
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

Settings come from the config file, falling back to built-in defaults. There
are no per-field environment overrides: each agent resolves its own provider,
model, base URL and API key, so agents on different providers never share
credentials. Secrets are pulled from the environment by interpolation (below).

- **Config file**: `$AGENT_RESPONDER_HOME/config.yaml` (default home
  `$HOME/.agent-responder/config.yaml`). See `config.yaml.example` for a template.
- **Env var**: `AGENT_RESPONDER_HOME` overrides the config home directory.
- **`.env`**: the CLI loads `./.env` (`KEY=VALUE` lines) at startup. Variables
  already set in the environment win over the file.
- **Errors**: a config file that is not valid YAML (or whose top level is not a
  mapping) aborts the command with an error. Individual malformed entries
  (an agent without a name, a webhook without a `/` path, ...) are dropped and
  reported as `Config warning:` lines on stderr.
- **Env interpolation**: config values may use `$NAME` or `${NAME}`; set
  variables are substituted, unset ones are left as-is.

The file is sectioned:

- `providers:` — named endpoints. Each has `base_url`, `api_key`, `timeout`,
  and a `models` list (the models you can switch among).
- `agents:` — a list of named agents. Each picks a `provider` and `model`,
  plus an optional `tools` list (opt-in: omit for every available built-in tool,
  `[]` for none, or name a subset), an optional `workspace`, and an optional
  `persona` (a key into `personas:`). The file tools (`read_file`, `write_file`,
  `edit_file`) are only available to an agent with a `workspace`: an absolute
  directory they are confined to. `../` traversal, absolute paths outside it, and
  symlinks pointing out of it are refused, and a workspace that contains (or sits
  inside) `AGENT_RESPONDER_HOME` is rejected because that holds `config.yaml` with
  your keys. A bad workspace disables the file tools and prints a `Config warning`.
  Writes are atomic and files over 10 MB are refused. Anything *inside* the
  workspace is still fair game, so keep it narrow and prefer `tools: [read_file]`
  for agents that don't need to write.
- `personas:` — names a persona. Each named persona gets a directory under
  `AGENT_RESPONDER_HOME` -- `personas/<name>/` -- holding its `SOUL.md` plus a
  `sessions/` directory of conversation logs. The `SOUL.md` value seeds that file
  on first run and becomes the system prompt for any agent that references it.
- `interaction:` — names which agent each interaction uses; `chat.agent` is
  the agent the `agent chat` loop runs.
- `webhooks:` — names HTTP paths that trigger a specific agent. Each entry has a
  `path`, an `agent` (a name under `agents:`), an optional `secret` that, when
  set, requires `Authorization: Bearer <secret>`, and an optional `system_prompt`
  that overrides the agent's default system prompt. Served by `agent webhooks`.

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

**Access control.** The config holds API keys and webhook secrets, so the API is
not open to arbitrary web pages even though CORS defaults to `*`:

- Every `/api/*` request needs the per-launch token in an `X-Agent-Token` header.
  The token is embedded in the served page (which is sent with `Cache-Control:
  no-store` and no CORS headers, so other sites cannot read it) and printed at
  startup for scripts. It changes on every launch.
- The `Host` header must be `localhost`, `127.0.0.1`, `::1` or the bind host
  (blocks DNS rebinding). This check is skipped when binding `0.0.0.0` / `::`.
- `Access-Control-Allow-Origin` on `/api/*` is `*` by default; set
  `--cors-origin https://ui.example` to restrict it, or `--cors-origin ""` to
  disable CORS. Note that with `*` the token is the only thing protecting the API:
  anyone who obtains it (e.g. from the startup output) can call it from any origin.

## Webhook listeners

`agent webhooks` serves the webhooks named in the `webhooks:` config section on a
local HTTP server (no extra dependencies; binds to `127.0.0.1:8787` by default).
Each POST to a configured `path` runs that webhook's `agent` with the parsed
request body (JSON is re-serialized; non-JSON is passed through as text) and
returns the agent's answer. A webhook with a `secret` requires the request to
carry `Authorization: Bearer <secret>`; others are unauthenticated.
A webhook may carry an optional `system_prompt`, which overrides the agent's
default (persona-based) system prompt for that webhook's runs.

```sh
uv run agent webhooks                # http://127.0.0.1:8787/
uv run agent webhooks --port 9001    # custom port
uv run agent webhooks --host 0.0.0.0 # expose beyond localhost
```

## Extending it

- **Add a tool**: implement `name` / `description` / `parameters` (JSON Schema)
  / `async run(arguments: dict[str, Any]) -> str` (the arguments arrive as one
  dict), then `registry.register(MyTool())`. Before `run` is called, the registry
  checks the arguments against `parameters` (required keys and scalar types) and
  returns a descriptive error to the model on a mismatch; each call is also limited
  to 60 seconds. Tools registered this way are used programmatically -- the
  `tools:` config list only selects among the built-ins.
- **Use a different model**: implement the `LLMClient` protocol
  (`async chat(...) -> LLMResponse`, `async close()`) and pass it to `Agent`.
  The default client talks to any OpenAI-compatible endpoint; set it per
  provider with `base_url` in `config.yaml`.
