# light-harness

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
cp .env.example .env       # secrets referenced from config.yaml as $NAME; `agent` loads ./.env (and ~/.light-harness/.env)
cp config.yaml.example ~/.light-harness/config.yaml   # then edit

uv run agent tools         # list the built-in example tools
uv run agent chat "What is 2 + 3?"
uv run agent chat          # interactive REPL
uv run agent web             # edit the config in a local web UI
uv run agent webhooks          # serve the configured webhook listeners
```

In the REPL, replies stream in as they are generated and render as Markdown, each tool
call shows as a dim `⚙ name(args) → result` line, and a rule with the elapsed time
closes each turn. Input history persists in
`$LIGHT_HARNESS_HOME/history` (arrow keys, Ctrl-R), Alt-Enter inserts a newline, and
Ctrl-C clears the typed text (or leaves the chat when the prompt is empty); during a
reply it stops that reply without leaving the chat (the partial reply stays in
the conversation). Commands start with `/` (Tab completes them): `/help`, `/clear`,
`/tools`, `/exit` (or `/quit`). Anything else is sent to the model.

The UI degrades gracefully: with piped input/output or `TERM=dumb` it prints plain text
(no spinner, panel or rule), on a non-UTF-8 terminal it swaps its symbols for ASCII, and
`NO_COLOR` turns colors off. Failures name the likely fix (bad API key, rate limit,
unreachable endpoint, timeout).

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

- **Config file**: `$LIGHT_HARNESS_HOME/config.yaml` (default home
  `$HOME/.light-harness/config.yaml`). See `config.yaml.example` for a template.
- **Env var**: `LIGHT_HARNESS_HOME` overrides the config home directory.
- **`.env`**: the CLI loads `./.env` and then `$LIGHT_HARNESS_HOME/.env`
  (`KEY=VALUE` lines) at startup. Variables already set in the environment win
  over both files, and `./.env` wins over the home file; `./.env` may itself set
  `LIGHT_HARNESS_HOME`.
- **Errors**: a config file that is not valid YAML (or whose top level is not a
  mapping) aborts the command with an error. Individual malformed entries
  (an agent without a name, a webhook without a `/` path, ...) are dropped and
  reported as `Config warning:` lines on stderr.
- **Env interpolation**: config values may use `$NAME` or `${NAME}`; set
  variables are substituted, unset ones are left as-is -- except in credentials,
  where that would be unsafe: a provider `api_key` that references an unset
  variable is cleared (so `agent chat` reports "No API key"), and a webhook whose
  `secret` references one is disabled. Each case prints a `Config warning:`.

The file is sectioned:

- `providers:` — named endpoints. Each has `base_url`, `api_key`, `timeout`,
  and a `models` list (the models you can switch among).
- `agents:` — a list of named agents. Each picks a `provider` and `model`,
  plus an optional `tools` list (opt-in: omit for every available built-in tool,
  `[]` for none, or name a subset), an optional `workspace`, and an optional
  `persona` (a key into `personas:`). The file tools (`read_file`, `write_file`,
  `edit_file`) are only available to an agent with a `workspace`: an absolute
  directory they are confined to. `../` traversal, absolute paths outside it, and
  symlinks pointing out of it are refused, as are files with more than one hard
  link (another name may sit outside the workspace), and a workspace that contains (or sits
  inside) `LIGHT_HARNESS_HOME` is rejected because that holds `config.yaml` with
  your keys. A bad workspace disables the file tools and prints a `Config warning`.
  A `tools:` name that is not a built-in prints a `Config warning` and is ignored.
  Writes are atomic and files over 10 MB are refused. Only regular files are
  reachable: a FIFO, socket or device inside the workspace is refused, because
  reading one would block the tool. Anything else *inside* the workspace is still
  fair game, so keep it narrow and prefer `tools: [read_file]` for agents that
  don't need to write.
- **Forgejo tools** (`src/light_harness/tools/forgejo/`) are built in like the others:
  available to every agent unless its `tools:` list names a subset. They call a
  Forgejo instance named per call: `forgejo_pr` (PR details as JSON), `forgejo_pr_diff`
  (diff text), `forgejo_pr_comment` (post a comment), `forgejo_pr_review` (approve /
  request changes / comment, with optional inline comments on `path` +
  `new_line`/`old_line`), `forgejo_pr_comment_edit` and `forgejo_pr_comment_delete` (both
  by `comment_id`, which must belong to the named PR) and `forgejo_pr_review_delete` (by
  `review_id`). `forgejo_pr_create` opens a PR (`title`, `head`, optional `base` --
  default branch if omitted -- and `body`; set labels/assignees afterwards with
  `forgejo_issue_update`), `forgejo_pr_close` closes one without merging, and
  `forgejo_pr_merge` merges it (`method` merge/rebase/rebase-merge/squash/
  fast-forward-only, optional `title`, `message`, `delete_branch`, `head_commit_id`).
  The per-PR tools take a PR `url`
  or `instance_url` + `repo` + `pr_number`, and an API `token` (required for everything
  that writes or deletes). Issues: `forgejo_issue_create` (`title`, optional `body`,
  `labels` by name, `assignees`), `forgejo_issue_list` (`state`, `labels`, `query`,
  `limit`/`page`; short summaries), `forgejo_issue` (one issue, optionally with its
  comments), `forgejo_issue_update` (title, body, `state` open/closed, and `assignees` /
  `labels`, which replace the current ones), `forgejo_issue_comment`, and
  `forgejo_issue_comment_edit` / `forgejo_issue_comment_delete` (by `comment_id`, which
  must belong to the named issue). The repo-level
  ones take a repo `url` or `instance_url` + `repo`; the per-issue ones an issue `url` or
  `instance_url` + `repo` + `issue_number`. `forgejo_clone` runs `git clone` of a repo
  (`url`, or `instance_url` + `repo`) into `path`, with optional `token`, `branch` and
  `depth`; like the file tools it exists only for an agent with a `workspace`, and `path`
  must resolve
  inside it (the destination must be missing or empty). The token is passed to git as a
  header, never written into the clone's `.git/config`. To keep an agent away from them,
  give it a `tools:` list without them. (The earlier `extra_tools` /
  `disable_extra_tools` keys are no longer read and print a warning.)
- `personas:` — names a persona. Each named persona gets a directory under
  `LIGHT_HARNESS_HOME` -- `personas/<name>/` -- holding its `SOUL.md` plus a
  `sessions/` directory of conversation logs. The `SOUL.md` value seeds that file
  on first creation only; from then on the file is the source of truth for the
  system prompt of every agent that references the persona. Edit the prompt by
  editing `personas/<name>/SOUL.md` -- or through `agent web`, whose persona
  field reads and writes that file.
- `interaction:` — names which agent each interaction uses; `chat.agent` is
  the agent the `agent chat` loop runs.
- `webhooks:` — names HTTP paths that trigger a specific agent. Each entry has a
  `path`, an `agent` (a name under `agents:`), an optional `secret` that, when
  set, requires `Authorization: Bearer <secret>`, and an optional `system_prompt`
  that overrides the agent's default system prompt. Served by `agent webhooks`.
- `webhook_server:` — tunes `agent webhooks`: `host`, `port`, extra
  `allowed_hosts`, and `output_log` (when true, each accepted webhook call
  logs itself and names its run's session log file on the server's stdout;
  `--output-log` / `--no-output-log` override it).

`Config.active_agent()` resolves the agent the loop uses (the one named by
`interaction.chat.agent`, else the first agent, else a default), and
`Config.endpoint()` resolves that into the effective endpoint.
New features add their own section (see `Config` in `config.py`).

## Web config UI

`--webui` on `agent chat` or `agent webhooks` also serves a small local web page
for viewing and editing the config file (no extra dependencies; binds to
`127.0.0.1` by default):

```sh
uv run agent chat --webui                      # http://127.0.0.1:8765/
uv run agent webhooks --webui --webui-port 9000  # custom port
uv run agent webhooks --webui --webui-host 0.0.0.0  # expose beyond localhost
```

The page has a navigation sidebar with **Providers**, **Agents**, **Personas**,
**Chat** and **Webhooks** pages, and a top bar showing the config file path with
**Save** and **Refresh** buttons to the right (they apply to every page).
Providers (each with base URL, API key, timeout, and an add/remove
able list of models), agents (each with name, provider, model, opt-in tools, and
persona), personas (each with its `SOUL.md`: the field shows the persona's stored
`personas/<name>/SOUL.md` -- the prompt the agent actually uses -- and saving it
rewrites that file; any other file a persona has is kept as it is), the
chat page (which agent the chat loop uses), and webhooks (path, agent, optional
secret and system prompt) are edited via form fields. Edits
write back to `config.yaml` (env refs like `$NAME` / `${NAME}` are preserved), and
on-disk changes are reflected automatically. Only
the sections a request actually names are rewritten: a request that names none
of them is refused, so no save can drop the sections it never touched. A running
`agent chat` session is unaffected -- changes apply to the next session.

**Access control.** The config holds API keys and webhook secrets, so the API is
not open to arbitrary web pages even though CORS defaults to `*`:

- Every `/api/*` request needs the per-launch token in an `X-Agent-Token` header.
  The token is embedded in the served page (which is sent with `Cache-Control:
  no-store` and no CORS headers, so other sites cannot read it) and printed at
  startup for scripts. It changes on every launch. Every `/api/*` answer -- which
  carries the config's API keys and webhook secrets -- is sent with `Cache-Control:
  no-store` as well, so nothing in between can keep a copy.
- The `Host` header must name this server: `localhost` or any `*.localhost`,
  `127.0.0.1`, `::1`, the bind host, an IP address, or a name passed with
  `--allowed-host` (repeat it for each name the server is reached by -- a mDNS
  name, the name a reverse proxy forwards). This blocks DNS rebinding, and it
  applies to a `0.0.0.0` / `::` bind too: that is the bind a browser can be talked
  into reaching under a hostname an attacker controls.
- `Access-Control-Allow-Origin` on `/api/*` is `*` by default; set
  `--cors-origin https://ui.example` to restrict it, or `--cors-origin ""` to
  disable CORS. The value must be `*` or a bare origin (`https://host[:port]`: no
  path, no `user:pass@`, no whitespace) because it is echoed into a response
  header verbatim -- anything else is refused at startup instead of becoming an
  extra header on every `/api/*` response. Note that with `*` the token is the
  only thing protecting the API: anyone who obtains it (e.g. from the startup
  output) can call it from any origin.

## Webhook listeners

`agent webhooks` serves the webhooks named in the `webhooks:` config section on a
local HTTP server (no extra dependencies; binds to `127.0.0.1:8787` by default).
Each POST to a configured `path` runs that webhook's `agent` with the parsed
request body (JSON is re-serialized; non-JSON is passed through as text). The
server acknowledges immediately with `200 {"agent": ..., "message": "OK"}` once
the agent has been built, and runs it in the background: the agent's answer is
not returned to the caller (a failed run is only printed to stderr), and a run
that cannot start gets a 500. Concurrent runs are capped; POSTs over the cap
get 429. A webhook with a `secret` requires the request to
carry `Authorization: Bearer <secret>`; others are unauthenticated.

Every request must also carry a `Host` header that names the server -- a loopback
name, an IP address, the bind host, or a name passed with `--allowed-host` (the
name a reverse proxy forwards). A request under any other hostname gets 403 and
no agent is run for it.

A webhook may carry an optional `system_prompt`, which overrides the agent's
default (persona-based) system prompt for that webhook's runs.

With `--output-log` (or `output_log: true` under `webhook_server:`), the server
logs each webhook call whose run started -- time, path, agent, and the session
log file that will hold that run's chat log (relative to the harness home, so
`personas/<persona>/sessions/<file>.jsonl`). `--no-output-log` turns it back
off. Rejected requests (401, 403, 404, 429) and calls whose agent cannot start
(500) log nothing.

At startup the server prints warnings on stderr naming the webhooks that cannot work as
configured: the unauthenticated ones, and any webhook whose agent cannot get the tools its
config implies -- an omitted `tools` list means every built-in, so a webhook whose agent has
no `workspace` loses `read_file`, `write_file` and `edit_file`, and a webhook naming an agent
that does not exist cannot run at all.

```sh
uv run agent webhooks                # http://127.0.0.1:8787/
uv run agent webhooks --port 9001    # custom port
uv run agent webhooks --host 0.0.0.0 # expose beyond localhost
uv run agent webhooks --allowed-host hooks.example # behind a proxy at that name
uv run agent webhooks --output-log # log each call and its session log file
```

## Extending it

- **Add a tool**: implement `name` / `description` / `parameters` (JSON Schema)
  / `async run(arguments: dict[str, Any]) -> str` (the arguments arrive as one
  dict), then `registry.register(MyTool())`. Before `run` is called, the registry
  checks the arguments against `parameters` (required keys and scalar types) and
  returns a descriptive error to the model on a mismatch; each call is also limited
  to 60 seconds -- but a timeout can only fire while the event loop is free, so a
  tool whose `run` blocks the loop synchronously cannot be interrupted: hand
  blocking work (disk, sockets) to a thread with `asyncio.to_thread`, as the
  built-in file tools do. A timed-out call is abandoned, not killed: its worker
  thread runs to completion in the background. Tools registered this way are used
  programmatically -- the `tools:` config list only selects among the built-ins.
- **Use a different model**: implement the `LLMClient` protocol
  (`async chat(...) -> LLMResponse`, `async close()`) and pass it to `Agent`.
  The default client talks to any OpenAI-compatible endpoint; set it per
  provider with `base_url` in `config.yaml`.
