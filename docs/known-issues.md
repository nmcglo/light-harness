# Known issues

Analysis of `light-harness` at commit `f6d9be1` (2026-10-11). Read-only review; no
behaviour was changed.

**Update 2026-10-12:** the review's one high-severity finding, **B1**, **S1**, **S2** and
**P1** are fixed and no longer listed here — `src/light_harness/tools/edit_diff.py`
(regression tests in `tests/test_edit_file.py`), `read_file`'s `offset`/`limit` handling in
`src/light_harness/tools/builtin.py` (regression tests in `tests/test_read_file.py`), the
umask/root handling in `src/light_harness/tools/workspace.py` (regression tests in
`tests/test_workspace.py`), and the persona `SOUL.md` round-trip: the web UI now reads and
writes `personas/<name>/SOUL.md`, the file the prompt is taken from, instead of only the
config value that seeds it (`src/light_harness/web.py`, `persona_soul_text` /
`write_persona_soul` in `src/light_harness/config.py`; regression tests in
`tests/test_web.py` and `tests/test_config.py`). Everything below still describes
`f6d9be1`, and its `config.py`, `edit_diff.py`, `web.py` and `workspace.py` line numbers are
pre-fix. Current baseline: `uv run pytest -q` 169 passed, `ruff` and `mypy` clean.

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
| S3 | `web.py` | CRLF injection through `--cors-origin`; secrets served without `Cache-Control` | medium | yes |
| D1 | `README.md`, `webhooks.py` | Docs say POST returns the agent's answer; code acks immediately | medium | yes |
| C1 | `tools/__init__.py` | 60 s tool timeout cannot interrupt sync-blocking tools | medium | yes |
| B2 | `tools/builtin.py` | Non-string `edit_file` entries silently dropped | low | yes |
| C2 | `tools/builtin.py` | No locking on read-modify-write | medium | inference |
| L1 | `llm.py` | Unguarded response shape (`choices[0]`, malformed tool_call) | medium | inference |
| E2 | `edit_diff.py` | No-op guard can never fire in fuzzy mode | medium | inference |
| S4 | `tools/workspace.py` | Deny check is lexical (case-insensitive FS) and blind to hardlinks | low | inference |
| W2 | `web.py` | Unguarded `read_text`; inconsistent `ConfigError` handling | low | inference |
| X1 | `tests/` | Coverage gaps on the highest-risk paths | medium | inference |

## Medium

### S3 — Header injection and caching on the config API

- `--cors-origin` is interpolated into `Access-Control-Allow-Origin` without validation. Verified: an
  origin containing `\r\n` produced an injected `Access-Control-Allow-Credentials: true` — credentialed
  cross-origin access to an API that returns API keys and webhook secrets.
- API responses containing secrets are sent with no `Cache-Control` — `no-store` is set only for the
  served page (`web.py:871`), not for `/api`. Verified.
- The Host allow-list (`web.py:745-746`) is skipped on wildcard binds, which the README documents as the
  DNS-rebinding posture for `--host 0.0.0.0`; combined with CORS `*` and secrets in JSON, that is the
  practical chain to token theft. The webhook server has no Host guard at all. Open (unauthenticated)
  webhooks are triggerable from any browser page; authenticated ones are not, because a simple cross-origin
  POST cannot carry `Authorization`.

### C1 — The 60-second tool timeout is unenforceable for the file tools

All three file tools are sync-blocking inside `async def run` (`open`, `read_bytes`, `mkstemp`), so
`asyncio.wait_for` (`tools/__init__.py:88`) cannot fire — the code documents this at `tools/__init__.py:48-49`, but
README's "Extending it" section states "each call is also limited to 60 seconds" without that caveat.
Verified: a FIFO inside the workspace reports size 0, passes `_check_size` (`builtin.py:83`), blocks
`f.read()` (`169`) forever, and the timeout never triggers; the event loop stalls and the run slot is held
permanently (429 for every subsequent request). README line 88 says anything inside the workspace is fair
game, so a model-created FIFO is reachable. Fix direction: run blocking tool bodies in an executor, or
validate `st_mode` before opening.

### C2 — No locking on read-modify-write

`builtin.py:359` read → `376` match → `381` write with no advisory lock, and `webhooks.py:86` builds a
fresh registry per request against the same `agent_cfg.workspace_path()`. Two concurrent runs (`agent chat`
plus a webhook, or two webhooks) editing the same file clobber each other, and the diff returned describes
stale content.

### L1 — LLM response shape is unguarded

`llm.py:169` `response.json()["choices"][0]["message"]` → `IndexError` when `choices` is empty (common with
guardrails); a non-JSON 200 body → `ValueError`; `170-176` `raw["id"]`, `raw["function"]["arguments"]` →
`KeyError` on a malformed tool call. `Agent.run` (`agent.py:38-72`) has no `try`, the webhook worker prints
to stderr after already acking 200 (`webhooks.py:171`), and `cli._run_repl` (`cli.py:178-186`) catches
only `EOFError/KeyboardInterrupt` — one transient failure ends the whole REPL. The transport-error retry
branch and the exponential backoff (`llm.py:145-152`) are untested: the existing test uses canned
`MockTransport` responses and `retry_delay=0`, so a change from exponential to constant delay, or a dropped
`if last: raise`, would pass. Connection failures are the most common real failure mode.

### E2 — The "no changes made" guard can never fire in fuzzy mode

`edit_diff.py:264-271` compares `base_content` (normalized) against `new_content`; `used_fuzzy` is only true
when the two views already differ (`239`), so a no-op edit modulo normalization is accepted and the file is
rewritten with the fuzzy-normalized version of every touched line (smart quotes → ASCII, trailing whitespace
stripped). Also `190-193`'s docstring promises preservation of unchanged-line bytes, but `213` rewrites the
whole touched line from the normalized view — the guarantee holds only outside the touched group. Fuzzy
uniqueness (`249`) counts in fuzzy space, so a genuinely unique exact match can be rejected: content
`"hello \nhello\n"` with `old_text="hello "` matches exactly once but counts as two. Line-ending handling
differs between the two paths (`374-377`): the exact path re-serializes every newline to the single detected
ending, so a mixed CRLF/LF file is converted wholesale, and `detect_line_endings` tie-breaking (`41-49`)
picks LF for `"a\rb\r\nc"`. The diff shown (`386`) is computed on normalized content, so it omits the BOM.

## Low

- **S4** — `workspace.py:47` `is_relative_to` compares `PurePath` parts as strings and `resolve()` does not
  canonicalize case, so on case-insensitive filesystems (macOS/Windows) `path="../.LIGHT-HARNESS/config.yaml"`
  is the same file as the denied home and its API keys are readable. Platform-conditional; no test. NFC/NFD
  decomposed names on APFS have the same shape. `resolve()` follows symlinks only, so a hardlink inside the
  workspace reads as "inside" and leaks content (writes are safe by accident — `os.replace` breaks the link).
- **W2** — `web.py:833` `path.read_text(encoding="utf-8")` is unguarded: a binary config file raises
  `UnicodeDecodeError` → no response, traceback in the server log. `ConfigError` is a `ValueError` subclass
  (`config.py:150`), so `web.py:822` turns a broken config into a 400, while the CLI aborts the command — the
  same input behaves differently in the two entry points.
- **B2** — `_prepare_edits` (`builtin.py:264-288`) drops entries whose `new_text` is not a string, so a request
  that *did* send an edit reports "requires at least one replacement". Verified.
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
- `generate_diff` runs `difflib.unified_diff` over up to 10 MB inside an async tool — `SequenceMatcher` is O(n²)
  worst case and `autojunk` can produce a misleading diff for large files. The write already happened before the
  diff (`381` then `386`), so a timeout reports failure for a change already on disk.

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
