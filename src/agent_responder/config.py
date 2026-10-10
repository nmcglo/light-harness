"""Configuration: field defaults < config.yaml < environment variables.

The config file lives at ``$AGENT_RESPONDER_HOME/config.yaml`` (default home
``$HOME/.agent-responder``). It is optional. Precedence for every value is:
environment variable > config file > built-in default.

Config file values may reference environment variables with ``$NAME`` or
``${NAME}``; set variables are substituted, unset ones are left as-is.

Structure:

- ``providers:`` maps a name to an endpoint (``base_url``, ``api_key``,
  ``timeout``) plus the ``models`` it offers -- the set you can switch among.
- ``agent:`` names the ``provider`` and ``model`` the loop uses, plus the
  ``tools`` it may call (opt-in).

``Config.endpoint()`` resolves those into the effective :class:`Settings` the
agent loop talks to. New features add their own section + settings model to
``Config``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_FILENAME = "config.yaml"
ENV_PREFIX = "AGENT"

# Single source of truth for built-in defaults, shared by Provider, AgentConfig,
# and Settings so they can't drift apart.
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = ""
DEFAULT_TIMEOUT: float = 60.0
DEFAULT_MODEL = "qwen3.8:27b"
DEFAULT_PROVIDER_NAME = "local"


def config_home() -> Path:
    """The ``AGENT_RESPONDER_HOME`` directory, defaulting to ``$HOME/.agent-responder``."""
    env_home = os.environ.get("AGENT_RESPONDER_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".agent-responder"


def config_file_path() -> Path:
    """Path to the config file inside :func:`config_home`."""
    return config_home() / CONFIG_FILENAME


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse the YAML config file. Returns ``{}`` if missing, empty, or invalid."""
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


_ENV_VAR_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def resolve_env(value: str) -> str:
    """Resolve ``$NAME`` / ``${NAME}`` references in ``value`` from the environment.

    A reference whose name is not set in the environment is left as-is (the
    literal ``$NAME`` / ``${NAME}`` text is preserved).
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return os.environ[name] if name in os.environ else match.group(0)

    return _ENV_VAR_RE.sub(_sub, value)


def _resolve_env_in_value(value: Any) -> Any:
    """Recursively resolve env-var references in all string values (keys untouched)."""
    if isinstance(value, str):
        return resolve_env(value)
    if isinstance(value, list):
        return [_resolve_env_in_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_env_in_value(item) for key, item in value.items()}
    return value


def _env_value(field: str) -> str | None:
    """Read the env var for ``field`` (unset or empty -> ``None``). ``api_key``
    additionally falls back to ``OPENAI_API_KEY``."""
    value = os.environ.get(f"{ENV_PREFIX}_{field.upper()}")
    if field == "api_key":
        value = value or os.environ.get("OPENAI_API_KEY")
    return value if value else None


def _parse_providers(raw: Any) -> dict[str, Provider]:
    """Parse the ``providers:`` section. Malformed entries are skipped."""
    if not isinstance(raw, dict):
        return {}
    providers: dict[str, Provider] = {}
    for name, spec in raw.items():
        if isinstance(spec, dict):
            try:
                providers[str(name)] = Provider.model_validate(spec)
            except ValidationError:
                continue
    return providers


def _parse_agent(raw: Any) -> AgentConfig:
    """Parse the ``agent:`` section, falling back to defaults."""
    if not isinstance(raw, dict):
        return AgentConfig()
    try:
        return AgentConfig.model_validate(raw)
    except ValidationError:
        return AgentConfig()


class Provider(BaseModel):
    """A model endpoint: connection settings plus the models it offers."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    timeout: float = Field(default=DEFAULT_TIMEOUT, gt=0)
    models: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Which provider + model the agent loop uses, and which tools it may call."""

    model_config = ConfigDict(extra="ignore")

    provider: str = DEFAULT_PROVIDER_NAME
    model: str = DEFAULT_MODEL

    # Optional opt-in list of built-in tool names. None (absent) -> all built-in
    # tools; [] -> no tools; [...] -> only the named tools.
    tools: list[str] | None = None


class Settings(BaseModel):
    """The effective endpoint the agent loop talks to."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    timeout: float = Field(default=DEFAULT_TIMEOUT, gt=0)

    @classmethod
    def from_sources(cls, file_values: dict[str, Any]) -> Settings:
        """Apply precedence: env var > file value > field default."""
        values: dict[str, Any] = {}
        for field in cls.model_fields:
            env_value = _env_value(field)
            if env_value is not None:
                values[field] = env_value
            elif file_values.get(field) not in (None, ""):
                values[field] = file_values[field]
        return cls(**values)


class Config:
    """Top-level configuration: ``providers`` + ``agent`` selection.

    ``endpoint()`` resolves the effective :class:`Settings` for the loop. Add
    further sections here as features grow (e.g. memory, retrieval).
    """

    def __init__(self, providers: dict[str, Provider], agent: AgentConfig) -> None:
        self.providers = providers
        self.agent = agent

    @classmethod
    def load(cls) -> Config:
        raw = _resolve_env_in_value(load_yaml(config_file_path()))
        return cls(
            providers=_parse_providers(raw.get("providers")),
            agent=_parse_agent(raw.get("agent")),
        )

    def endpoint(self) -> Settings:
        """Resolve the effective endpoint: env > selected provider/agent > default."""
        provider_name = _env_value("provider") or self.agent.provider
        provider = self.providers.get(provider_name) or Provider()
        file_values = {
            "model": self.agent.model,
            "base_url": provider.base_url,
            "api_key": provider.api_key,
            "timeout": provider.timeout,
        }
        return Settings.from_sources(file_values)
