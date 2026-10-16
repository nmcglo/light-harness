"""Smoke tests for the ``agent`` command line: startup, errors, and a chat turn."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from light_harness import cli
from light_harness.llm import ChatMessage, LLMResponse

runner = CliRunner()

CONFIG = """\
providers:
  local:
    base_url: http://llm.test/v1
    api_key: $TEST_KEY
    models: [m]
agents:
  - name: a
    provider: local
    model: m
    tools: [echo]
interaction:
  chat:
    agent: a
"""


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An empty harness home and cwd, with the environment restored afterwards
    (the CLI's .env loading writes to ``os.environ``)."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with mock.patch.dict(os.environ):
        os.environ["LIGHT_HARNESS_HOME"] = str(home)
        os.environ.pop("TEST_KEY", None)
        yield home


def test_tools_lists_the_configured_agents_tools(home: Path) -> None:
    (home / "config.yaml").write_text(CONFIG, encoding="utf-8")

    result = runner.invoke(cli.app, ["tools"])

    assert result.exit_code == 0, result.output
    assert "echo" in result.output
    assert "current_time" not in result.output


def test_broken_config_aborts_with_an_error(home: Path) -> None:
    (home / "config.yaml").write_text("- not\n- a mapping\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["tools"])

    assert result.exit_code == 1
    assert "top level must be a mapping" in result.output


def test_home_env_file_supplies_the_api_key(home: Path) -> None:
    """``$TEST_KEY`` in the config is only set because ``<home>/.env`` is read."""
    (home / "config.yaml").write_text(CONFIG, encoding="utf-8")
    (home / ".env").write_text("TEST_KEY=secret\n", encoding="utf-8")
    sent: list[list[ChatMessage]] = []

    class FakeClient:
        def __init__(self, settings: object) -> None:
            self.api_key = getattr(settings, "api_key", None)

        async def chat(
            self, messages: list[ChatMessage], tools: list[dict] | None = None, on_token=None
        ) -> LLMResponse:
            sent.append(messages)
            return LLMResponse(
                message=ChatMessage(role="assistant", content=f"key={self.api_key}")
            )

        async def close(self) -> None:
            return None

    with mock.patch.object(cli, "OpenAICompatibleClient", FakeClient):
        result = runner.invoke(cli.app, ["chat", "hello"])

    assert result.exit_code == 0, result.output
    assert "key=secret" in result.output
    assert sent and sent[0][-1].content == "hello"


def test_chat_without_an_api_key_exits_with_a_hint(home: Path) -> None:
    (home / "config.yaml").write_text(CONFIG, encoding="utf-8")

    result = runner.invoke(cli.app, ["chat", "hello"])

    assert result.exit_code == 1
    assert "No API key found" in result.output


def test_a_failed_one_shot_prompt_reports_an_error_and_exits_1(home: Path) -> None:
    (home / "config.yaml").write_text(CONFIG, encoding="utf-8")
    (home / ".env").write_text("TEST_KEY=secret\n", encoding="utf-8")

    class FailingClient:
        def __init__(self, settings: object) -> None:
            pass

        async def chat(
            self, messages: list[ChatMessage], tools: object = None, on_token=None
        ) -> LLMResponse:
            raise RuntimeError("endpoint exploded")

        async def close(self) -> None:
            return None

    with mock.patch.object(cli, "OpenAICompatibleClient", FailingClient):
        result = runner.invoke(cli.app, ["chat", "hello"])

    assert result.exit_code == 1
    assert "endpoint exploded" in result.output


def _capture_webhook_output_log(home: Path, seen: list[bool]):
    """Patch ``serve_webhooks`` so the command returns and records the flag."""

    def fake_serve(config: cli.Config, **kwargs: object) -> None:
        seen.append(config.webhook_server.output_log)

    return mock.patch.object(cli, "serve_webhooks", fake_serve)


def test_webhooks_output_log_flag_overrides_the_config(home: Path) -> None:
    (home / "config.yaml").write_text(
        CONFIG + "webhook_server:\n  output_log: false\n", encoding="utf-8"
    )
    seen: list[bool] = []
    with _capture_webhook_output_log(home, seen):
        for flags in ([], ["--output-log"], ["--no-output-log"]):
            result = runner.invoke(cli.app, ["webhooks", *flags])
            assert result.exit_code == 0, result.output
    assert seen == [False, True, False]


def test_webhooks_no_output_log_flag_overrides_a_true_config(home: Path) -> None:
    (home / "config.yaml").write_text(
        CONFIG + "webhook_server:\n  output_log: true\n", encoding="utf-8"
    )
    seen: list[bool] = []
    with _capture_webhook_output_log(home, seen):
        result = runner.invoke(cli.app, ["webhooks", "--no-output-log"])
        assert result.exit_code == 0, result.output
    assert seen == [False]
