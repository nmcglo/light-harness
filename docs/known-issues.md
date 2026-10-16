# Known issues

Analysis of `light-harness` at commit `f6d9be1` (2026-10-11). Read-only review; no
behaviour was changed.

**Update 2026-10-12:** the review's one high-severity finding, **B1**, **B2**, **C1**, **C2**, **D1**, **E2**, **L1**, **S1**, **S2**,
**S3**, **S4** and **P1** are fixed and no longer listed here — `src/light_harness/tools/edit_diff.py`
(regression tests in `tests/test_edit_file.py`), `read_file`'s `offset`/`limit` handling in
`src/light_harness/tools/builtin.py` (regression tests in `tests/test_read_file.py`), the
umask/root handling in `src/light_harness/tools/workspace.py` (regression tests in
`tests/test_workspace.py`), the persona `SOUL.md` round-trip: the web UI now reads and
writes `personas/<name>/SOUL.md`, the file the prompt is taken from, instead of only the
config value that seeds it (`src/light_harness/web.py`, `persona_soul_text` /
`write_persona_soul` in `src/light_harness/config.py`; regression tests in
`tests/test_web.py` and `tests/test_config.py`), and the header/caching/`Host` hardening of
both HTTP servers: `--cors-origin` is validated to `*` or a bare `http(s)://host[:port]`
before it is ever echoed into a header (`_validated_cors_origin` in `src/light_harness/web.py`),
every `/api/*` response now carries `Cache-Control: no-store`, and the `Host` allow-list is
shared by both servers (`host_allow_list` / `host_is_allowed` in `src/light_harness/httputil.py`)
instead of being skipped on a wildcard bind — the webhook server had none at all, and a name
the bind host does not cover now has to be named with the new `--allowed-host` option on
`agent web` / `agent webhooks` (regression tests in `tests/test_web.py` and
`tests/test_webhooks.py`; README access-control sections updated), and the 60-second tool
timeout: the file tools now hand their blocking disk work to a worker thread
(`asyncio.to_thread` in `src/light_harness/tools/builtin.py`), so `asyncio.wait_for` can fire
and the loop keeps running while a call is stalled, and `Workspace.resolve` refuses any
existing path that is not a regular file — the FIFO behind the original repro is rejected
before anything opens it (`src/light_harness/tools/workspace.py`; regression tests in
`tests/test_tool_timeout.py`; README caveats updated), and the **D1** documentation: the README
and `webhooks.py` docstring now say a webhook POST is acknowledged at once and the agent's answer
is not returned (the code was correct), and **B2**: `_prepare_edits` in
`src/light_harness/tools/builtin.py` now rejects a malformed `edit_file` entry with a message naming
it instead of dropping it (regression test in `tests/test_edit_file.py`), and **C2**: `write_file` and
`edit_file` now hold an exclusive `flock` on the file's directory (`file_lock` in
`src/light_harness/tools/workspace.py`) across the whole read → match → write, so concurrent runs —
threads or separate processes — no longer overwrite each other's edits (regression test in
`tests/test_edit_file.py`; advisory, so it only orders the harness's own writers, and edits to
different files in one directory take turns), and **L1**: `OpenAICompatibleClient.chat` validates the
response shape and raises `LLMError` (`src/light_harness/llm.py`) naming the problem — empty `choices`
(with the endpoint's `error` text if any), a non-JSON 200 body, a missing `message`, a tool call without
an id or function name — instead of `IndexError`/`KeyError`/`ValueError`; `agent chat` reports a failed
turn and keeps the REPL running, and a one-shot prompt prints the error and exits 1 (`src/light_harness/cli.py`);
the connection-error retry and exponential backoff now have tests (`tests/test_agent.py`). Transport and HTTP
status errors still surface as `httpx` exceptions, and **E2**: the "no changes made" guard in
`apply_edits` (`src/light_harness/tools/edit_diff.py`) now compares each matched span with its
replacement instead of comparing whole files, so a no-op edit that only matched fuzzily is refused instead
of silently ASCII-fying the line, and a unique exact match such as `"hello "` in `"hello \nhello\n"` is no
longer rejected as a fuzzy duplicate (regression tests in `tests/test_edit_file.py`); the
`_apply_preserving_lines` docstring now says touched lines take the fuzzy normalization. Two E2 claims did
not hold: the BOM is unchanged by an edit so its absence from the diff is correct, and
`detect_line_ending("a\rb\r\nc")` returns CRLF, and **S4**: `Workspace.resolve`
(`src/light_harness/tools/workspace.py`) now also matches a `deny` directory by identity
(`samefile` on every existing ancestor), so a different spelling of it on a case-insensitive or
normalization-insensitive filesystem is refused, and refuses an existing file with more than one hard
link, since its other names may be in a denied directory or outside the workspace (regression tests in
`tests/test_workspace.py`; the identity check is exercised with a symlink alias, not on a real
case-insensitive filesystem),
and **W2**: `GET /api/config` (`src/light_harness/web.py`) now answers 500 with a JSON `error` when the
config file is unreadable or not UTF-8 instead of raising inside the handler, and the page shows that
error instead of rendering `undefined` (regression test in `tests/test_web.py`). The other half of W2 is
deliberate, not a bug: `POST` already turned a `ConfigError` into a 400, and `GET` still returns the raw
text of a file with invalid YAML so the page can show what to fix, whereas the CLI aborts. Everything
below still describes
`f6d9be1`, and its `config.py`, `edit_diff.py`, `web.py` and `workspace.py` line numbers are
pre-fix. Current baseline: `uv run pytest -q` 195 passed, `ruff` and `mypy` clean.

**Baseline checks**

| Check | Result |
| --- | --- |
| `uv run ruff check .` | clean |
| `uv run pytest -q` | 103 passed |
| `uv run mypy` (documented command) | clean, 15 source files |
| `uv run mypy tests` | **31 errors in 4 test files** |

`pyproject.toml:45-49` sets `strict = true` with `packages = ["light_harness"]`, so the
documented `uv run mypy` never sees `tests/`. The strict posture over the test suite is
accidental rather than enforced (untyped defs, unannotated fixture parameters, bare `dict`,
`Optional` attribute access). Add an explicit `[[tool.mypy.overrides]] module = ["tests.*"]`
block, or annotate the fakes (`CapturingClient`, `FakeClient`, `AlwaysCalls`, `BrokenClient`)
so they satisfy `LLMClient` by declaration instead of by shape.

Severity: **high** = silent data loss or wrong result on a normal path; **medium** = reachable
failure under realistic conditions; **low** = polish or platform-conditional.
Verification: *verified* = reproduced against the live code; *inference* = read from the code.

## Index

| ID | Area | Issue | Severity | Verified |
| --- | --- | --- | --- | --- |
| X1 | `tests/` | Coverage gaps on the highest-risk paths | medium | inference |

## Low

- **Mixed line endings (was part of E2)** — `edit_file` normalizes the whole file to LF and writes it back
  with the single ending `detect_line_ending` found first, so a file mixing CRLF and LF (or lone `\r`)
  comes back uniform after any edit. Unchanged by the E2 fix; fixing it needs per-line ending tracking
  across edits that add or remove lines.
- `edit_file`'s read block (`357-368`) has no generic `except OSError`, so `ENXIO`/`ELOOP` falls through to the
  registry catch-all as `Error: OSError: ...` instead of the tool's own message; `WriteFileTool` does catch
  `OSError` (`259`).
- `atomic_write_bytes`' "a crash never leaves a truncated file" holds for truncation (rename is atomic) but not
  durability — no `fsync`. `WriteFileTool` reports characters (`261`) while the limit is bytes (`_MAX_FILE_BYTES`).
- `SessionLog.record` (`sessions.py:40`) has no lock; concurrent appends rely on OS atomicity.
- `state.py:23-34` (`add_user`, `add_assistant`, `add_tool`, `reset`) and `Agent.reset()` (`agent.py:34-36`) are
  unreachable API surface kept alive only by tests that do not exist.
- `cli.py:45` `load_dotenv()` reads `./.env` from the current working directory (as documented), not
  `LIGHT_HARNESS_HOME`; `config.py:145` writes parsed values into `os.environ`, so any test that reads env vars
  after `test_config.py:466-492` is order-dependent. `cli.py` has no tests at all.
- `generate_diff` runs `difflib.unified_diff` over up to 10 MB — `SequenceMatcher` is O(n²) worst
  case and `autojunk` can produce a misleading diff for large files. It now runs in the edit
  tool's worker thread, so it no longer stalls the event loop, but the write already happened
  before the diff (`381` then `386`), so a timeout reports failure for a change already on disk.

## Test-quality issues

- `test_config.py:147` is tautological: `assert with_ws == {t.name for t in builtin_tools(tmp_path)}` compares the
  implementation against itself.
- `test_webhooks.py:357-363` is a mock-echo: it asserts the fixture's own literal `"sp"` (`test_webhooks.py:90`),
  not `config.system_prompt_for`. `default_agent_factory` (`webhooks.py:79-108`) — workspace wiring, persona
  resolution, `ensure_persona`, session creation, the `except BaseException: session.close()` leak guard — has
  zero coverage because every server test injects its own factory.
- `test_webhooks.py:375` `test_webhook_rejects_oversized_body_without_reading_it` proves the agent did not run,
  not that bytes were not read. `test_workspace.py:100` `test_write_is_atomic_and_keeps_mode` never interrupts a
  write or races a reader; it asserts an incidental directory listing.
- Flaky/environment-sensitive: `urlopen` in `test_web.py:41,62` and `test_webhooks.py:113,118` honours `HTTP_PROXY`
  unless `no_proxy` covers `127.0.0.1` (other tests in the same file use raw sockets — inconsistent protection);
  `test_webhooks.py:370-372` assumes the whole status line lands in one `recv`; `join(timeout=5)` and
  `gate.wait(5)` timeouts are never asserted, so a wedged handler thread passes.
- Untested paths worth covering: `edit_file` uniqueness, overlap, fuzzy, CRLF/BOM/mixed endings, no-op, empty file;
  the `POST` body-error 400/413
  paths for the endpoint that writes the config file; `esc()` (the only thing preventing stored-XSS token theft,
  since config values are inserted via `innerHTML`); the wildcard-bind Host skip (`web.py:745-746`);
  symlink-loop and NUL paths in `Workspace.resolve`; `atomic_write_bytes`' failure path; `array`/`object`
  argument types (skipped by design, so `edit_file`'s `items.required` is never enforced) and unknown arguments;
  a tool raising an exception (`tools/__init__.py:91`); `max_concurrent_runs > 1` and slot release on a failing run;
  the `edit_file` result size limit (`builtin.py:379`).

## What is solid

Workspace confinement holds for the headline cases and is well tested: `../`, absolute paths, symlink escape,
the documented home rule in both directions, empty/NUL paths, double-slash roots, no symlink creation, and
config-layer rejection of bad workspace settings. Argument validation is correct including the `bool`-is-`int`
trap (`tools/__init__.py:40`). Size limits are checked before writing (`builtin.py:251`, `379`). No exec,
subprocess, or network primitives exist in the tools. The web UI access-control model — per-launch token, Host
allow-list, constant-time compare, complete HTML escaping, env-ref preservation, on-disk refresh, secrets failing
closed for unset env references — matches the README.
