"""Secret registry: policy, redaction boundaries, the pre-send guard, and the
Forgejo cutover (tokens come from the registry by host, never from the model)."""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from light_harness.agent import Agent
from light_harness.config import Config, Settings
from light_harness.llm import ChatMessage, LLMResponse, OpenAICompatibleClient, ToolCall
from light_harness.secrets import (
    Redactor,
    SecretContext,
    SecretDenied,
    SecretLeakError,
    SecretPolicy,
    SecretRegistry,
    SecretSpec,
    SecretValue,
    ToolSecrets,
)
from light_harness.sessions import SessionLog
from light_harness.tools import ToolRegistry, build_tool_registry

from ._helpers import EchoTool

TOKEN = "tok-9f8e7d6c5b4a"


def _registry(tmp_path: Path, audit: list[str] | None = None, **spec: Any) -> SecretRegistry:
    registry = SecretRegistry(tmp_path, audit=(audit.append if audit is not None else None))
    fields: dict[str, Any] = {
        "source": "env:T_FORGE_TOKEN",
        "allow": SecretPolicy(
            tools=["forgejo_pr"], hosts=["git.example.com"], agents=["bot"],
            interactions=["chat"],
        ),
        **spec,
    }  # fmt: skip
    registry.define("forge", SecretSpec(**fields))
    return registry


@pytest.fixture(autouse=True)
def _token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("T_FORGE_TOKEN", TOKEN)


def _ctx(**kwargs: Any) -> SecretContext:
    return SecretContext(**{"agent": "bot", "interaction": "chat", **kwargs})


@pytest.mark.parametrize(
    ("tool", "host", "context"),
    [
        ("forgejo_pr_merge", "git.example.com", _ctx()),
        ("forgejo_pr", "evil.example.com", _ctx()),
        ("forgejo_pr", "git.example.com", _ctx(agent="other")),
        ("forgejo_pr", "git.example.com", _ctx(interaction="webhook")),
    ],
    ids=["wrong-tool", "wrong-host", "wrong-agent", "wrong-interaction"],
)
def test_any_failed_condition_denies_without_saying_which(
    tmp_path: Path, tool: str, host: str, context: SecretContext
) -> None:
    audit: list[str] = []
    registry = _registry(tmp_path, audit)

    with pytest.raises(SecretDenied) as info:
        registry.get("forge", tool=tool, host=host, context=context)

    assert str(info.value) == "use of secret 'forge' was denied"
    assert TOKEN not in "".join(audit)
    assert "denied" in audit[-1]


def test_allowed_use_gets_a_value_that_does_not_print_itself(tmp_path: Path) -> None:
    audit: list[str] = []
    value = _registry(tmp_path, audit).get(
        "forge", tool="forgejo_pr", host="git.example.com:443", context=_ctx()
    )

    assert value.reveal() == TOKEN
    assert str(value) == repr(value) == f"{value}" == "[REDACTED:forge]"
    with pytest.raises(TypeError):
        json.dumps({"t": value})
    assert audit[-1].endswith("allowed") and TOKEN not in audit[-1]


def test_secret_without_allow_block_is_usable_by_no_tool(tmp_path: Path) -> None:
    registry = _registry(tmp_path, allow=None)
    with pytest.raises(SecretDenied):
        registry.get("forge", tool="forgejo_pr", host="git.example.com", context=_ctx())


def test_expiry_and_untrusted_content_deny(tmp_path: Path) -> None:
    expired = _registry(tmp_path, expires=date.today() - timedelta(days=2))
    with pytest.raises(SecretDenied):
        expired.get("forge", tool="forgejo_pr", host="git.example.com", context=_ctx())

    tainted = _registry(tmp_path, deny_after_untrusted=True)
    tainted.get("forge", tool="forgejo_pr", host="git.example.com", context=_ctx())
    with pytest.raises(SecretDenied):
        tainted.get(
            "forge", tool="forgejo_pr", host="git.example.com", context=_ctx(untrusted=True)
        )


def test_confirm_asks_in_chat_remembers_once_per_session_and_denies_webhooks(
    tmp_path: Path,
) -> None:
    allow = SecretPolicy(tools=["forgejo_pr"], hosts=["git.example.com"])
    asked: list[str] = []
    answers = iter([False, True])

    def confirm(name: str, tool: str, host: str) -> bool:
        asked.append(name)
        return next(answers)

    registry = _registry(tmp_path, allow=allow, confirm="once-per-session")
    context = _ctx(confirm=confirm)
    use = {"tool": "forgejo_pr", "host": "git.example.com"}
    with pytest.raises(SecretDenied):
        registry.get("forge", **use, context=context)  # user said no
    registry.get("forge", **use, context=context)  # user said yes
    registry.get("forge", **use, context=context)  # remembered, not asked again
    assert asked == ["forge", "forge"]

    webhook = _ctx(interaction="webhook", confirm=lambda *_a: True)
    with pytest.raises(SecretDenied):
        registry.get("forge", **use, context=webhook)


def test_redactor_catches_raw_base64_url_and_json_forms() -> None:
    secret = 'p@ss/w"rd+12345'
    redactor = Redactor({"api": secret})
    for leaked in (
        secret,
        base64.b64encode(secret.encode()).decode(),
        base64.b64encode(b"user:" + secret.encode()).decode(),  # basic-auth style
        "p%40ss%2Fw%22rd%2B12345",
        json.dumps(secret)[1:-1],
    ):
        out = redactor.redact(f"before {leaked} after")
        assert "[REDACTED:api]" in out, leaked
        assert leaked not in out


class _ScriptedClient:
    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = script
        self.seen: list[list[ChatMessage]] = []

    async def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.script.pop(0)

    async def close(self) -> None:
        return None


def test_tool_output_with_a_token_is_redacted_in_history_and_session_file(
    tmp_path: Path,
) -> None:
    leaked = f"raw={TOKEN} b64={base64.b64encode(TOKEN.encode()).decode()}"
    call = ToolCall(id="c1", name="echo", arguments={"text": leaked})
    client = _ScriptedClient(
        [
            LLMResponse(
                message=ChatMessage(role="assistant", content="", tool_calls=[call]),
                tool_calls=[call],
            ),
            LLMResponse(message=ChatMessage(role="assistant", content="done")),
        ]
    )
    tools = ToolRegistry()
    tools.register(EchoTool())
    redactor = Redactor({"forge": TOKEN})
    session = SessionLog(tmp_path / "s", meta={"persona": "p"}, redactor=redactor)
    agent = Agent(client, tools, session=session, redactor=redactor)  # type: ignore[arg-type]

    asyncio.run(agent.run("go"))
    session.close()

    tool_message = agent.conversation.messages[2]
    assert tool_message.content == "raw=[REDACTED:forge] b64=[REDACTED:forge]"
    # The model's second request saw only the redacted result.
    assert TOKEN not in json.dumps([m.model_dump() for m in client.seen[1]])
    log = next((tmp_path / "s").glob("*.jsonl")).read_text(encoding="utf-8")
    assert TOKEN not in log and base64.b64encode(TOKEN.encode()).decode() not in log
    assert [json.loads(line)["role"] for line in log.splitlines()[1:]] == [
        "user", "assistant", "tool", "assistant",
    ]  # fmt: skip


def test_pre_send_guard_blocks_a_payload_containing_a_value() -> None:
    sent: list[httpx.Request] = []
    http = httpx.AsyncClient(
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(500)),
    )
    client = OpenAICompatibleClient(
        Settings(), client=http, retry_delay=0, guard=Redactor({"forge": TOKEN})
    )
    try:
        with pytest.raises(SecretLeakError, match="forge") as info:
            asyncio.run(client.chat([ChatMessage(role="user", content=f"x {TOKEN} y")]))
    finally:
        asyncio.run(http.aclose())
    assert sent == []
    assert TOKEN not in str(info.value)


def test_no_forgejo_tool_schema_has_a_token_property(tmp_path: Path) -> None:
    for spec in build_tool_registry(None, tmp_path).tool_schemas():
        params = spec["function"]["parameters"]
        assert "token" not in params.get("properties", {}), spec["function"]["name"]
        assert "token" not in params.get("required", []), spec["function"]["name"]


def _forgejo(tmp_path: Path, tool: str, arguments: dict[str, Any]) -> tuple[str, list]:
    """Run a Forgejo tool against a mock server; returns (result, requests)."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "git.example.com" and request.url.path.endswith("/pulls/1"):
            return httpx.Response(302, headers={"Location": "https://evil.example.com/x"})
        return httpx.Response(200, json={"number": 2})

    original = httpx.AsyncClient

    def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return original(*args, transport=httpx.MockTransport(handler), **kwargs)

    secrets = ToolSecrets(_registry(tmp_path), _ctx())
    registry = build_tool_registry([tool], None, secrets)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx, "AsyncClient", patched)
        result = asyncio.run(registry.run(tool, arguments))
    return result, seen


def test_forgejo_token_is_looked_up_by_host_and_kept_on_it(tmp_path: Path) -> None:
    result, seen = _forgejo(
        tmp_path, "forgejo_pr", {"url": "https://git.example.com/o/r/pulls/2"}
    )
    assert json.loads(result) == {"number": 2}
    assert seen[0].headers["Authorization"] == f"token {TOKEN}"

    # Another host gets no token at all.
    _, seen = _forgejo(tmp_path, "forgejo_pr", {"url": "https://other.example.com/o/r/pulls/2"})
    assert "Authorization" not in seen[0].headers

    # A redirect is not followed while the header is attached.
    result, seen = _forgejo(
        tmp_path, "forgejo_pr", {"url": "https://git.example.com/o/r/pulls/1"}
    )
    assert result.startswith("Error") and "302" in result
    assert [r.url.host for r in seen] == ["git.example.com"]


def test_forgejo_tool_outside_the_policy_is_denied(tmp_path: Path) -> None:
    result, seen = _forgejo(
        tmp_path, "forgejo_pr_diff", {"url": "https://git.example.com/o/r/pulls/2"}
    )
    assert result == "Error: SecretDenied: use of secret 'forge' was denied"
    assert seen == []


def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Config:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("LIGHT_HARNESS_HOME", str(home))
    return Config.load()


def test_secret_references_resolve_and_unresolved_ones_are_never_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("T_OPENAI", "sk-from-env-0000")
    config = _load(
        tmp_path,
        monkeypatch,
        "secrets:\n"
        "  openai: {source: env:T_OPENAI}\n"
        "  missing: {source: env:T_NOT_SET}\n"
        "  pasted: {source: ghp_abcdefghijklmnopqrstuvwxyz0123}\n"
        "providers:\n"
        "  a: {api_key: 'secret:openai'}\n"
        "  b: {api_key: 'secret:missing'}\n"
        "  c: {api_key: 'secret:nope'}\n"
        "  d: {api_key: ghp_abcdefghijklmnopqrstuvwxyz0123}\n",
    )
    assert config.providers["a"].api_key == "sk-from-env-0000"
    assert [config.providers[p].api_key for p in "bcd"] == ["", "", ""]
    joined = "\n".join(config.warnings)
    assert "secrets.pasted: source must be env:NAME or file:PATH" in joined
    assert "literal values are not allowed" in joined
    assert "providers.b: api_key references secret 'missing', which is not set" in joined
    assert "providers.c: api_key references unknown secret 'nope'" in joined
    assert "providers.d: api_key looks like a literal credential" in joined
    assert "ghp_" not in joined
    # Provider keys are redacted like any other secret.
    assert config.redactor().redact("k=sk-from-env-0000") == "k=[REDACTED:openai]"


def test_file_secret_inside_a_workspace_is_reported_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "token.txt").write_text(TOKEN + "\n")
    config = _load(
        tmp_path,
        monkeypatch,
        f"secrets:\n  forge: {{source: 'file:{ws / 'token.txt'}'}}\n"
        f"agents:\n  - {{name: a, workspace: {ws}}}\n",
    )
    assert config.secrets.values() == {"forge": TOKEN}
    assert "contains credential file(s)" in "\n".join(config.warnings)
    agent = config.agents[0]
    tools = build_tool_registry(None, agent.workspace_path(), config.tool_secrets(agent, "chat"))
    result = asyncio.run(tools.run("read_file", {"path": "token.txt"}))
    assert result.startswith("Error") and TOKEN not in result


def test_secret_value_reveal_is_the_only_way_out() -> None:
    value = SecretValue("x", "hunter2-hunter2")
    assert "hunter2" not in f"{value!s} {value!r} {value:>40}"


def test_system_prompt_note_lists_only_usable_hosts_and_tools(tmp_path: Path) -> None:
    from light_harness.secrets import with_secret_note

    registry = SecretRegistry(tmp_path, audit=lambda _line: None)
    allow = {"tools": ["forgejo_pr", "forgejo_clone"], "hosts": ["git.example.com"]}
    registry.define("forge", SecretSpec(source="env:T_FORGE_TOKEN", allow=SecretPolicy(**allow)))
    registry.define(
        "other_agent",
        SecretSpec(
            source="env:T_FORGE_TOKEN",
            allow=SecretPolicy(tools=["forgejo_pr"], hosts=["secret.example"], agents=["x"]),
        ),
    )
    registry.define(
        "unset",
        SecretSpec(
            source="env:T_NOT_SET",
            allow=SecretPolicy(tools=["forgejo_pr"], hosts=["dev.example"]),
        ),
    )
    registry.define(
        "asks", SecretSpec(source="env:T_FORGE_TOKEN", confirm="always", allow=SecretPolicy(
            tools=["forgejo_pr"], hosts=["ask.example"])),
    )

    chat = ToolSecrets(registry, _ctx())
    prompt = with_secret_note("base", chat, ["forgejo_pr", "current_time"])
    assert prompt.startswith("base\n\nAuthenticated access:")
    assert "- hosts git.example.com: forgejo_pr\n" in prompt + "\n"  # clone not available
    assert "dev.example: forgejo_pr (credential not set" in prompt
    assert "ask.example: forgejo_pr (the user is asked" in prompt
    assert "secret.example" not in prompt  # policy excludes this agent
    assert TOKEN not in prompt and "forge" not in prompt.replace("forgejo", "")

    webhook = ToolSecrets(registry, _ctx(interaction="webhook"))
    assert "ask.example" not in with_secret_note("b", webhook, ["forgejo_pr"])
    assert with_secret_note("b", chat, ["current_time"]) == "b"
