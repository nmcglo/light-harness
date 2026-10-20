# light-harness

A lightweight agentic harness for personal use, mainly as an automated
responder to triggers (webhooks). More polished harnesses exist, but they tend
to suffer from feature bloat and instability from vibe-coding.

Also, I just wanted to know that I could.

Models that support tool calling can call the harness's built-in tools, which
are written in Python.

## Contents

- [Quickstart](#quickstart)
- [Chat REPL](#chat-repl)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Built-in tools](#built-in-tools)
- [Web config UI](#web-config-ui)
- [Webhook listeners](#webhook-listeners)
- [Extending it](#extending-it)

## Quickstart

```sh
uv sync --group dev                                  # create .venv, install deps, write uv.lock
cp .env.example .env                                 # secret values, referenced from config's secrets:
cp config.yaml.example ~/.light-harness/config.yaml  # then edit

uv run agent tools                  # list the built-in tools
uv run agent chat "What is 2 + 3?"  # one-shot
uv run agent chat                   # interactive REPL
uv run agent web                    # edit the config in a local web UI
uv run agent secrets list           # configured secrets and their policies (never values)
uv run agent webhooks               # serve the configured webhook listeners
```

Checks:

```sh
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Chat REPL

- Replies stream in as they are generated and render as Markdown.
- Each tool call shows as a dim `⚙ name(args) → result` line.
- A rule showing the elapsed time closes each turn.

| Key / command | Effect |
|---|---|
| ↑ / ↓, Ctrl-R | Input history (persisted in `$LIGHT_HARNESS_HOME/history`) |
| Alt-Enter | Insert a newline |
| Ctrl-C (while typing) | Clear the input; on an empty prompt, leave the chat |
| Ctrl-C (during a reply) | Stop the reply; the partial reply stays in the conversation |
| `/help`, `/clear`, `/tools`, `/system_prompt`, `/exit` (`/quit`) | Commands (Tab completes them); `/system_prompt` prints the system prompt the agent was given, including the secrets note |

Anything that isn't a command is sent to the model.

The output adapts to the terminal:

- Piped input/output or `TERM=dumb`: plain text, without the spinner, panel or rule.
- Non-UTF-8 terminal: ASCII symbols.
- `NO_COLOR`: no colors.

Error messages suggest the likely fix (bad API key, rate limit, unreachable
endpoint, timeout).

## How it works

`Agent.run(user_input)` adds the prompt to a long-running `Conversation`, then loops:

1. Send `system` + history (with tool schemas) to the model.
2. If the model requests tool calls, run each one, append the results, and loop.
3. If the model returns a plain message, append it and return it as the answer.

The loop ends when the model stops calling tools or `max_steps` is reached.
A tool error doesn't raise: it goes back to the model as a string, so the agent
can recover.

Each session is appended to a JSONL log at
`personas/<name>/sessions/` in the harness home. The log records the user
prompts, the assistant replies, and the tool calls and their results.

## Configuration

### Where settings come from

| Source | Notes |
|---|---|
| `$LIGHT_HARNESS_HOME/config.yaml` | Default home: `~/.light-harness`. Template: `config.yaml.example`. Missing settings use built-in defaults. |
| `LIGHT_HARNESS_HOME` | Overrides the home directory. |
| `.env` files | Loaded at startup: `./.env`, then `$LIGHT_HARNESS_HOME/.env` (`KEY=VALUE` lines). Precedence: real environment > `./.env` > home `.env`. `./.env` may set `LIGHT_HARNESS_HOME`. |

There are no per-field environment overrides. Each agent resolves its own
provider, model, base URL and API key, so agents on different providers never
share credentials.

**Env interpolation.** Config values may use `$NAME` or `${NAME}`. Variables
that are set get substituted; unset ones are left as-is, except in credentials
(see [Secrets](#secrets)):

- a provider `api_key` that doesn't resolve is cleared, so `agent chat`
  reports "No API key";
- a webhook `secret` that doesn't resolve disables that webhook.

Both cases print a `Config warning:`. An unresolved reference is never used
literally.

**Errors.** If the file isn't valid YAML, or its top level isn't a mapping, the
command aborts. Malformed entries (an agent without a name, a webhook path that
doesn't start with `/`, …) are dropped, and each prints a `Config warning:` line
to stderr.

### Sections

| Section | Purpose |
|---|---|
| `secrets:` | Named credentials: a `source` reference plus the policy for using it (see [Secrets](#secrets)). |
| `providers:` | Named endpoints: `base_url`, `api_key`, `timeout`, and a `models` list. |
| `agents:` | Named agents: `provider`, `model`, and the optional `tools`, `workspace`, `persona` and `secrets` bindings. |
| `personas:` | Named system prompts, each stored in its own directory (see below). |
| `interaction:` | Which agent each interaction uses; `chat.agent` is the agent for `agent chat`. |
| `webhooks:` | HTTP paths that trigger an agent: `path`, `agent`, plus the optional `secret` and `system_prompt`. |
| `webhook_server:` | `host`, `port`, extra `allowed_hosts`, and `output_log`. |

`Config.active_agent()` picks the agent the loop uses: `interaction.chat.agent`,
else the first agent, else a default. `Config.endpoint()` turns that agent into
the endpoint it calls. New features add their own section (see `Config` in
`config.py`).

### Agents

- **`tools`** (opt-in): omit it to get every available built-in, set `[]` for
  none, or list a subset. Unknown names print a `Config warning` and are ignored.
  The old `extra_tools` / `disable_extra_tools` keys are no longer read and
  print a warning.
- **`workspace`**: an absolute directory. The file tools and `forgejo_clone`
  are only available to agents that have one (see
  [Workspace sandbox](#workspace-sandbox)).
- **`persona`**: a key into `personas:`.
- **`secrets`**: optional purpose-to-secret bindings, e.g.
  `secrets: {forgejo: forgejo_main}`. The Forgejo tools then use that secret
  instead of the one found by host; its policy still applies.

The agent learns where it is authenticated from a note appended to its system
prompt (chat and webhooks alike). It lists, for each secret whose policy admits
this agent and interaction, the allowed hosts and the agent's tools that may
use it, and flags secrets that aren't set, need confirmation, or switch off
after untrusted content. Secret names and values are never included. Secrets
that need confirmation are left out for webhooks, which would deny them anyway.
The listed hosts are visible to anything that can read the conversation,
including injected prompts.

### Secrets

The model handles names; the harness resolves values inside trusted code at the
last moment and scrubs everything flowing back toward the model or the logs.
Config files hold references, never values:

```yaml
secrets:
  forgejo_main:
    source: env:FORGEJO_TOKEN        # or file:/path, or file:secrets/name (relative to home)
    allow:
      tools: [forgejo_pr, forgejo_pr_comment, forgejo_clone]
      hosts: [git.example.com]       # host, or host:port
      agents: [assistant]            # optional; omitted = any agent
      interactions: [chat, webhook]  # optional; omitted = both
    confirm: never                   # never | once-per-session | always
    expires: 2027-01-01              # optional
    deny_after_untrusted: false      # optional
  openai:
    source: env:OPENAI_API_KEY

providers:
  openai:
    api_key: secret:openai           # $OPENAI_API_KEY works too (registered as env:OPENAI_API_KEY)
```

- **References.** A provider `api_key` and a webhook `secret` take
  `secret:NAME` or `$NAME`. Either way the value is known to the redactor. A
  `source` that isn't `env:` / `file:` is dropped with a warning (it is never
  treated as a value), and a literal credential pasted into `api_key` or
  `secret` (an `sk-…`/`ghp_…`-shaped or long random-looking string) is rejected.
  Short literals like `dummy` still work for local servers.
- **Policy.** Default deny: a secret without an `allow` block, or whose `allow`
  lacks `tools` or `hosts`, is usable by no tool. Every listed condition must
  hold. Provider and webhook references are implicitly allowed for their own
  field. A denied tool call gets `use of secret 'x' was denied`, never the rule
  that failed.
- **`confirm`** asks on the terminal in an interactive chat. Webhooks (and
  piped chats) can't answer, so anything but `never` denies them.
- **`deny_after_untrusted`** denies the secret once untrusted content is in the
  conversation: always for webhook runs (the payload), and in chat after
  `forgejo_pr`, `forgejo_pr_diff`, `forgejo_issue` or `forgejo_issue_list`
  returned a result. `/clear` resets it.
- **Audit.** Each decision appends a line (time, secret, tool, host, agent,
  interaction, allowed/denied and why) to `$LIGHT_HARNESS_HOME/secrets-audit.log`.
  Values are never logged.
- **Redaction.** Every registered value -- raw, base64, URL-encoded and
  JSON-escaped -- is replaced with `[REDACTED:name]` in tool results, user
  input (with a warning), assistant replies, session logs, the webhook output
  log, and error messages. Before each model request the serialized payload is
  scanned; if a value is still in it, the request is aborted
  (`SecretLeakError`). Values shorter than 6 characters are not redacted.
- **Workspaces.** A workspace overlapping `LIGHT_HARNESS_HOME` is rejected. One
  that contains `./.env` or a `file:` secret prints a warning, and the file
  tools refuse those files.

```sh
uv run agent secrets list            # sources, set/unset, policies; never values
uv run agent secrets set forgejo_main  # hidden prompt (or stdin); writes <home>/secrets/forgejo_main (0600)
```

`agent secrets set` points `secrets.<name>.source` at the file it wrote and
keeps the entry's policy; when the source changes, `config.yaml` is rewritten
(as the web UI does, comments are not kept).

Limits: redaction only catches registered values, so the policy is the primary
control and redaction the fallback. Streamed reply tokens are shown as they
arrive, before redaction. A tool that can send arbitrary data anywhere (a
shell or fetch tool) would make any secret it is granted visible to the model.

### Personas

Each persona gets a directory, `$LIGHT_HARNESS_HOME/personas/<name>/`, which holds:

- `SOUL.md`: the system prompt for every agent that uses the persona. The
  `SOUL.md` value in the config only seeds this file the first time it's
  created. After that, the file is the source of truth: edit it directly or
  through `agent web`.
- `sessions/`: conversation logs.

## Built-in tools

### File tools

`read_file`, `write_file`, `edit_file`. These are only available to an agent
that has a `workspace`.

#### Workspace sandbox

The tools refuse:

- paths outside the workspace: `../` traversal, outside absolute paths, and
  symlinks that point out of it;
- files with more than one hard link, since another name for the file may be
  outside the workspace;
- anything that isn't a regular file (FIFOs, sockets, devices), because
  reading one would block the tool;
- files over 10 MB.

Writes are atomic. A workspace that contains, or is inside,
`LIGHT_HARNESS_HOME` is rejected, because the home holds `config.yaml` and the
file-backed secrets. An invalid workspace disables the file tools and prints a
`Config warning`. `./.env` and `file:` secrets are refused even inside a
workspace.

Everything else inside the workspace is reachable, so keep the workspace small.
Agents that don't need to write should get `tools: [read_file]`.

### Forgejo tools

These live in `src/light_harness/tools/forgejo/` and are available to every
agent unless its `tools:` list leaves them out. Each call names the Forgejo
instance to talk to. The API token is never an argument: the tool looks up the
secret whose `allow.hosts` lists the instance's host (or the agent's
`secrets.forgejo` binding) and checks its policy. Without a matching secret the
call is anonymous, so tools that write or delete fail. With a token, redirects
are not followed, so the header can't be forwarded to another host.

How to address the target:

- **PR tools**: a PR `url`, or `instance_url` + `repo` + `pr_number`.
- **Issue tools**: an issue `url`, or `instance_url` + `repo` + `issue_number`.
- **Repo-level tools**: a repo `url`, or `instance_url` + `repo`.

**Pull requests**

| Tool | Does |
|---|---|
| `forgejo_pr` | PR details as JSON |
| `forgejo_pr_diff` | Diff text |
| `forgejo_pr_create` | Open a PR: `title`, `head`, plus optional `base` (default branch if omitted) and `body` |
| `forgejo_pr_close` | Close without merging |
| `forgejo_pr_merge` | Merge. `method`: merge / rebase / rebase-merge / squash / fast-forward-only. Optional: `title`, `message`, `delete_branch`, `head_commit_id` |
| `forgejo_pr_comment` | Post a comment |
| `forgejo_pr_comment_edit` / `_delete` | Edit or delete a comment by `comment_id` (the comment must belong to that PR) |
| `forgejo_pr_review` | Approve, request changes, or comment. Optional inline comments on `path` + `new_line`/`old_line` |
| `forgejo_pr_review_delete` | Delete a review by `review_id` |

To set labels or assignees on a PR, use `forgejo_issue_update`.

**Issues**

| Tool | Does |
|---|---|
| `forgejo_issue_create` | Create: `title`, plus optional `body`, `labels` (by name) and `assignees` |
| `forgejo_issue_list` | Short summaries; filter by `state`, `labels`, `query`; paginate with `limit`/`page` |
| `forgejo_issue` | One issue, optionally with its comments |
| `forgejo_issue_update` | Title, body, `state` (open/closed). `assignees` and `labels` replace the current ones |
| `forgejo_issue_comment` | Post a comment |
| `forgejo_issue_comment_edit` / `_delete` | Edit or delete a comment by `comment_id` (the comment must belong to that issue) |

**Clone**

`forgejo_clone` runs `git clone` of a repo into `path`. Optional arguments:
`branch`, `depth`.

- Like the file tools, it is only available to agents with a `workspace`.
- `path` must resolve inside the workspace and must be missing or empty.
- The token goes to git as a header and is never written to the clone's
  `.git/config`.

## Web config UI

`agent web` serves a local page for viewing and editing the config. You can
also add `--webui` to `agent chat` or `agent webhooks` to serve it alongside
them. It needs no extra dependencies and binds to `127.0.0.1` by default.

```sh
uv run agent chat --webui                           # http://127.0.0.1:8765/
uv run agent webhooks --webui --webui-port 9000     # custom port
uv run agent webhooks --webui --webui-host 0.0.0.0  # expose beyond localhost
```

The sidebar has pages for **Providers**, **Secrets**, **Agents**, **Personas**,
**Chat** and **Webhooks**. The top bar shows the config path, with **Save** and
**Refresh** buttons that apply to every page.

- Saving writes back to `config.yaml` and keeps references (`$NAME`,
  `${NAME}`, `secret:NAME`) as written. Changes made on disk show up in the
  page automatically.
- The page shows references and policies, never resolved values: every `/api`
  response is redacted. A legacy literal `api_key` shows as `[REDACTED:…]` and
  is kept as-is when saved back unchanged.
- A value typed into a secret's **New value** field is written to
  `$LIGHT_HARNESS_HOME/secrets/<name>` (mode 0600) via `POST /api/secret`, and
  the secret's `source` is pointed at it. It never goes into `config.yaml`.
- The persona field edits `personas/<name>/SOUL.md` directly. Other files in
  the persona's directory are left alone.
- A save only rewrites the sections it names, and a request that names none is
  refused. A save can't drop sections it didn't touch.
- A running `agent chat` session keeps its old config; changes apply to the
  next session.

### Access control

The config holds API keys and webhook secrets, so the API is locked down even
though CORS defaults to `*`:

- **Token.** Every `/api/*` request needs the per-launch token in an
  `X-Agent-Token` header. The token is embedded in the page, which is served
  with no CORS headers so other sites can't read it. It is also printed at
  startup for scripts, and it changes on every launch. The page and all
  `/api/*` responses are sent with `Cache-Control: no-store`, so caches in
  between don't keep copies.
- **Host check.** The `Host` header must be `localhost` / `*.localhost`,
  `127.0.0.1`, `::1`, the bind host, an IP address, or a name passed with
  `--allowed-host` (repeat the flag for each name, e.g. an mDNS name or the name
  a reverse proxy forwards). This blocks DNS rebinding, and it applies to
  `0.0.0.0` / `::` binds too.
- **CORS.** `Access-Control-Allow-Origin` on `/api/*` defaults to `*`.
  `--cors-origin https://ui.example` restricts it; `--cors-origin ""` disables
  it. The value must be `*` or a bare origin (`https://host[:port]`: no path,
  no `user:pass@`, no whitespace), and anything else is refused at startup.
  With `*`, the token is the only protection: anyone who has it (e.g. from the
  startup output) can call the API from any origin.

## Webhook listeners

`agent webhooks` serves the `webhooks:` section on a local HTTP server. It needs
no extra dependencies and binds to `127.0.0.1:8787` by default.

```sh
uv run agent webhooks                               # http://127.0.0.1:8787/
uv run agent webhooks --port 9001                   # custom port
uv run agent webhooks --host 0.0.0.0                # expose beyond localhost
uv run agent webhooks --allowed-host hooks.example  # behind a proxy at that name
uv run agent webhooks --output-log                  # log each call and its session log file
```

A POST to a configured `path` runs that webhook's `agent` with the request body.
JSON bodies are re-serialized; anything else is passed through as text. The
webhook's `system_prompt`, if set, replaces the agent's persona prompt for these
runs.

| Response | When |
|---|---|
| `200 {"agent": ..., "message": "OK"}` | The agent was built. It runs in the background; its answer isn't returned, and a failed run is only printed to stderr |
| 401 | Missing or wrong `Authorization: Bearer <secret>` (only for webhooks with a `secret`) |
| 403 | `Host` isn't a loopback name, an IP address, the bind host, or an `--allowed-host` |
| 404 | No webhook configured at that path |
| 429 | Too many concurrent runs |
| 500 | The run couldn't start |

Webhooks without a `secret` are unauthenticated.

**Output log.** `--output-log` (or `output_log: true` under `webhook_server:`)
prints a line for each call whose run started: time, path, agent, and the
session log file for that run (relative to the home, e.g.
`personas/<persona>/sessions/<file>.jsonl`). Rejected requests and 500s are
not logged. `--no-output-log` overrides the config.

**Startup warnings** go to stderr for:

- unauthenticated webhooks;
- webhooks whose agent can't get the tools its config implies. An omitted
  `tools` list means every built-in, so an agent without a `workspace` loses
  the file tools;
- webhooks that name an agent that doesn't exist. These can't run at all.

## Extending it

### Add a tool

Implement `name`, `description`, `parameters` (JSON Schema) and
`async run(arguments: dict[str, Any]) -> str`, then call
`registry.register(MyTool())`.

- Before `run` is called, the registry checks the arguments against
  `parameters` (required keys and scalar types). On a mismatch, the model gets
  a descriptive error.
- Each call has a 60-second limit, but the timeout can only fire while the
  event loop is free. Run blocking work (disk, sockets) in a thread with
  `asyncio.to_thread`, as the built-in file tools do. A call that times out is
  abandoned, not killed: its worker thread keeps running in the background
  until it finishes.
- Tools registered this way are for programmatic use only. The `tools:` config
  list only chooses among the built-ins.

### Use a different model

Implement the `LLMClient` protocol (`async chat(...) -> LLMResponse`,
`async close()`) and pass it to `Agent`. The default client works with any
OpenAI-compatible endpoint; set the endpoint per provider with `base_url` in
`config.yaml`.
