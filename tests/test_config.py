"""Tests for config loading: providers, agent selection, and precedence."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_responder.config import (
    AgentConfig,
    Config,
    config_file_path,
    config_home,
    ensure_persona,
    load_yaml,
    resolve_env,
)
from agent_responder.tools import build_tool_registry, builtin_tools

_ALL_VARS = (
    "AGENT_MODEL",
    "AGENT_PROVIDER",
    "AGENT_BASE_URL",
    "AGENT_API_KEY",
    "AGENT_TIMEOUT",
    "OPENAI_API_KEY",
)

CFG = """
    providers:
      local:
        base_url: http://local/v1
        api_key: local-key
        timeout: 15
        models:
          - qwen3.8:27b
      openai:
        base_url: https://api.openai.com/v1
        api_key: openai-key
        models:
          - gpt-4o-mini
    agents:
      - name: assistant
        provider: local
        model: qwen3.8:27b
    interaction:
      chat:
        agent: assistant
"""


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)


def _write_cfg(home: Path, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(textwrap.dedent(text))


def test_agent_selects_provider_and_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, CFG)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    e = Config.load().endpoint()
    assert (e.model, e.base_url, e.api_key, e.timeout) == (
        "qwen3.8:27b",
        "http://local/v1",
        "local-key",
        15.0,
    )


def test_switching_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, CFG)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    cfg = Config.load()
    cfg.agents = [AgentConfig(name="assistant", provider="openai", model="gpt-4o-mini")]
    e = cfg.endpoint()
    assert (e.model, e.base_url, e.api_key) == (
        "gpt-4o-mini",
        "https://api.openai.com/v1",
        "openai-key",
    )


def test_env_provider_and_model_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, CFG)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGENT_PROVIDER", "openai")
    monkeypatch.setenv("AGENT_MODEL", "gpt-4o")
    e = Config.load().endpoint()
    assert e.model == "gpt-4o"  # env model override
    assert e.base_url == "https://api.openai.com/v1"  # env switched provider
    assert e.api_key == "openai-key"


def test_env_field_override_beats_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, CFG)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGENT_BASE_URL", "http://env/v1")
    e = Config.load().endpoint()
    assert e.base_url == "http://env/v1"  # env wins over provider base_url
    assert e.api_key == "local-key"  # no env -> provider value


def test_defaults_when_no_file_or_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    e = Config.load().endpoint()
    assert (e.model, e.base_url, e.api_key, e.timeout) == (
        "qwen3.8:27b",
        "http://127.0.0.1:8000/v1",
        "",
        60.0,
    )


def test_openai_api_key_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    assert Config.load().endpoint().api_key == "fallback-key"


def test_missing_provider_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, "agents:\n  - name: a\n    provider: nope\n    model: some-model\n")
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    e = Config.load().endpoint()
    assert e.model == "some-model"
    assert e.base_url == "http://127.0.0.1:8000/v1"  # unknown provider -> defaults


def test_malformed_providers_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, "providers: [not-a-dict]\nagents:\n  - name: a\n    model: m\n")
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    e = Config.load().endpoint()
    assert e.model == "m"
    assert e.base_url == "http://127.0.0.1:8000/v1"


def test_extra_provider_keys_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = (
        "providers:\n  local:\n    base_url: http://x/v1\n    unknown: 1\n"
        "agents:\n  - name: a\n    model: m\n"
    )
    _write_cfg(tmp_path, cfg)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    e = Config.load().endpoint()
    assert e.base_url == "http://x/v1"
    assert e.model == "m"


def test_config_home_respects_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    assert config_home() == tmp_path
    assert config_file_path() == tmp_path / "config.yaml"


def test_config_home_defaults_to_home_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_RESPONDER_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_home() == tmp_path / ".agent-responder"


def test_load_yaml_missing_returns_empty(tmp_path: Path) -> None:
    assert load_yaml(tmp_path / "nope.yaml") == {}


def test_build_registry_default_all() -> None:
    names = build_tool_registry(None).names()
    assert set(names) == {t.name for t in builtin_tools()}


def test_build_registry_subset() -> None:
    assert set(build_tool_registry(["echo", "add"]).names()) == {"echo", "add"}


def test_build_registry_empty() -> None:
    assert build_tool_registry([]).names() == []


def test_build_registry_ignores_unknown() -> None:
    assert build_tool_registry(["echo", "bogus"]).names() == ["echo"]


def test_agent_tools_opt_in_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(
        tmp_path,
        "agents:\n  - name: a\n    model: m\n    tools:\n      - echo\n      - add\n",
    )
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    cfg = Config.load()
    assert cfg.active_agent().tools == ["echo", "add"]
    assert set(build_tool_registry(cfg.active_agent().tools).names()) == {"echo", "add"}


def test_agent_tools_default_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, "agents:\n  - name: a\n    model: m\n")
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    assert Config.load().active_agent().tools is None


def test_resolve_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "secret")
    assert resolve_env("$MY_KEY") == "secret"
    assert resolve_env("${MY_KEY}") == "secret"


def test_resolve_env_unset_leaves_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNSET_A", raising=False)
    monkeypatch.delenv("UNSET_B", raising=False)
    assert resolve_env("$UNSET_A") == "$UNSET_A"
    assert resolve_env("${UNSET_B}") == "${UNSET_B}"


def test_resolve_env_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOST", "example.com")
    monkeypatch.setenv("PORT", "8080")
    assert resolve_env("http://$HOST:$PORT/v1") == "http://example.com:8080/v1"
    assert resolve_env("http://${HOST}:${PORT}/v1") == "http://example.com:8080/v1"


def test_resolve_env_nonvar_and_plain() -> None:
    assert resolve_env("$5") == "$5"
    assert resolve_env("plain text") == "plain text"
    assert resolve_env("") == ""


def test_file_env_value_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(
        tmp_path,
        "providers:\n  local:\n    api_key: $MY_API_KEY\n"
        "agents:\n  - name: a\n    model: m\n",
    )
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    monkeypatch.setenv("MY_API_KEY", "resolved-key")
    assert Config.load().endpoint().api_key == "resolved-key"


def test_file_env_unset_left_literal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(
        tmp_path,
        "providers:\n  local:\n    api_key: $NO_SUCH_KEY\n"
        "agents:\n  - name: a\n    model: m\n",
    )
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    assert Config.load().endpoint().api_key == "$NO_SUCH_KEY"


def test_file_env_value_in_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = "providers:\n  local:\n    models:\n      - $MODEL_NAME\n"
    _write_cfg(tmp_path, cfg)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    monkeypatch.setenv("MODEL_NAME", "big-model")
    assert Config.load().providers["local"].models == ["big-model"]


def test_env_override_beats_file_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(
        tmp_path,
        "providers:\n  local:\n    api_key: $MY_API_KEY\n"
        "agents:\n  - name: a\n    model: m\n",
    )
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    monkeypatch.setenv("MY_API_KEY", "from-file-env")
    monkeypatch.setenv("AGENT_API_KEY", "from-env-override")
    assert Config.load().endpoint().api_key == "from-env-override"


def test_active_agent_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_text = (
        "agents:\n"
        "  - name: first\n"
        "    provider: local\n"
        "    model: a-model\n"
        "  - name: second\n"
        "    provider: openai\n"
        "    model: b-model\n"
        "interaction:\n"
        "  chat:\n"
        "    agent: second\n"
    )
    _write_cfg(tmp_path, cfg_text)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    active = Config.load().active_agent()
    assert active.name == "second"
    assert active.model == "b-model"


def test_active_agent_first_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_text = (
        "agents:\n"
        "  - name: first\n"
        "    provider: local\n"
        "    model: a-model\n"
        "interaction:\n"
        "  chat:\n"
        "    agent: missing\n"
    )
    _write_cfg(tmp_path, cfg_text)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    assert Config.load().active_agent().name == "first"


def test_active_agent_empty_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, "providers:\n  local:\n    base_url: http://x/v1\n")
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    active = Config.load().active_agent()
    assert active.name == "default"
    assert active.provider == "local"


def test_system_prompt_for_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_text = (
        "agents:\n"
        "  - name: assistant\n"
        "    provider: local\n"
        "    model: m\n"
        "    persona: helpful\n"
        "personas:\n"
        "  helpful:\n"
        "    SOUL.md: You are a kind helper.\n"
    )
    _write_cfg(tmp_path, cfg_text)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    cfg = Config.load()
    assert cfg.system_prompt_for(cfg.active_agent()) == "You are a kind helper."


def test_system_prompt_for_no_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(tmp_path, "agents:\n  - name: assistant\n    provider: local\n    model: m\n")
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    cfg = Config.load()
    assert cfg.system_prompt_for(cfg.active_agent()) == "You are a helpful assistant."


def test_system_prompt_for_unknown_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_text = (
        "agents:\n"
        "  - name: assistant\n"
        "    provider: local\n"
        "    model: m\n"
        "    persona: ghost\n"
    )
    _write_cfg(tmp_path, cfg_text)
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    cfg = Config.load()
    assert cfg.system_prompt_for(cfg.active_agent()) == "You are a helpful assistant."


def test_parse_personas() -> None:
    from agent_responder.config import _parse_personas

    raw = {
        "a": {"SOUL.md": "Hello", "other.txt": "x"},
        "b": "not-a-dict",
        "c": {"num": 3, "txt": "y"},
    }
    parsed = _parse_personas(raw)
    assert parsed == {
        "a": {"SOUL.md": "Hello", "other.txt": "x"},
        "c": {"num": "3", "txt": "y"},
    }
    assert _parse_personas(None) == {}


def test_parse_interaction() -> None:
    from agent_responder.config import _parse_interaction

    assert _parse_interaction({"chat": {"agent": "x"}}).chat.agent == "x"
    assert _parse_interaction(None).chat.agent == "default"
    assert _parse_interaction({"chat": "wrong"}).chat.agent == "default"


def test_config_load_creates_persona_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(
        tmp_path,
        "personas:\n  helpful:\n    SOUL.md: You are a kind helper.\n",
    )
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    Config.load()
    persona_dir = tmp_path / "personas" / "helpful"
    assert (persona_dir / "SOUL.md").read_text(encoding="utf-8") == "You are a kind helper."
    assert (persona_dir / "sessions").is_dir()


def test_system_prompt_is_read_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cfg(
        tmp_path,
        "personas:\n  helpful:\n    SOUL.md: You are a kind helper.\n",
    )
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    agent = AgentConfig(persona="helpful")
    assert Config.load().system_prompt_for(agent) == "You are a kind helper."

    # Editing the on-disk SOUL.md changes the prompt: disk is the source of truth.
    soul = tmp_path / "personas" / "helpful" / "SOUL.md"
    soul.write_text("You are a sassy helper.", encoding="utf-8")
    assert Config.load().system_prompt_for(agent) == "You are a sassy helper."


def test_ensure_persona_does_not_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_RESPONDER_HOME", str(tmp_path))
    _clear_env(monkeypatch)
    ensure_persona("custom", seed="initial seed")
    soul = tmp_path / "personas" / "custom" / "SOUL.md"
    assert soul.read_text(encoding="utf-8") == "initial seed"
    assert (tmp_path / "personas" / "custom" / "sessions").is_dir()

    # A later ensure with a different seed must not overwrite the existing file.
    ensure_persona("custom", seed="second seed")
    assert soul.read_text(encoding="utf-8") == "initial seed"
