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
DEFAULT_AGENT_NAME = "default"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def config_home() -> Path:
    """The ``AGENT_RESPONDER_HOME`` directory, defaulting to ``$HOME/.agent-responder``."""
    env_home = os.environ.get("AGENT_RESPONDER_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".agent-responder"


def config_file_path() -> Path:
    """Path to the config file inside :func:`config_home`."""
    return config_home() / CONFIG_FILENAME

SOUL_FILENAME = "SOUL.md"


def personas_home() -> Path:
    """The ``personas/`` directory under :func:`config_home`."""
    return config_home() / "personas"


def persona_dir(name: str) -> Path:
    """The directory for a single named persona.

    Raises :class:`ValueError` unless ``name`` is a safe single path segment, so
    every persona path helper (and :func:`ensure_persona`) is traversal-safe.
    """
    if not _valid_persona_name(name):
        raise ValueError(f"invalid persona name {name!r}: must be a single path segment")
    return personas_home() / name


def persona_soul_path(name: str) -> Path:
    """Path to a persona's stored ``SOUL.md``."""
    return persona_dir(name) / SOUL_FILENAME


def persona_sessions_dir(name: str) -> Path:
    """Path to a persona's ``sessions/`` directory (conversation logs)."""
    return persona_dir(name) / "sessions"


def _valid_persona_name(name: str) -> bool:
    """True if ``name`` is a safe single path segment (no path traversal)."""
    return (
        bool(name)
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
    )


def ensure_persona(name: str, seed: str = "") -> Path:
    """Ensure ``personas/<name>/`` exists with ``SOUL.md`` and ``sessions/``.

    Creates the directory tree and, on first creation only, writes ``SOUL.md``
    from ``seed`` (the persona's inline ``SOUL.md`` from the config file). An
    existing ``SOUL.md`` is never overwritten. Returns the persona directory.
    """
    directory = persona_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    soul = persona_soul_path(name)
    if not soul.exists():
        soul.write_text(seed, encoding="utf-8")
    persona_sessions_dir(name).mkdir(parents=True, exist_ok=True)
    return directory


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load ``KEY=VALUE`` lines from a ``.env`` file (default ``./.env``) into the
    environment. Variables already set in the environment win. Supports ``#``
    comments, blank lines, an optional ``export`` prefix, and single/double-quoted
    values. Returns the names that were set. A missing file is not an error."""
    env_path = path if path is not None else Path(".env")
    try:
        text = env_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{env_path}: cannot read env file: {exc}") from exc
    applied: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied


class ConfigError(ValueError):
    """The config file exists but cannot be used (unreadable, invalid YAML, wrong shape)."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse the YAML config file.

    Returns ``{}`` if the file is missing or empty. Raises :class:`ConfigError`
    if it is unreadable, is not valid YAML, or its top level is not a mapping --
    silently treating a broken file as empty would drop agents and webhook
    secrets without telling anyone.
    """
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: cannot parse config file: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


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


def _parse_providers(raw: Any, warnings: list[str]) -> dict[str, Provider]:
    """Parse the ``providers:`` section. Malformed entries are skipped (and reported)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        warnings.append("providers: expected a mapping; section ignored")
        return {}
    providers: dict[str, Provider] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            warnings.append(f"providers.{name}: expected a mapping; entry ignored")
            continue
        try:
            providers[str(name)] = Provider.model_validate(spec)
        except ValidationError as exc:
            warnings.append(f"providers.{name}: {_first_error(exc)}; entry ignored")
    return providers


def _first_error(exc: ValidationError) -> str:
    """A short ``field: message`` summary of the first validation error."""
    err = exc.errors()[0]
    loc = ".".join(str(part) for part in err["loc"])
    return f"{loc}: {err['msg']}" if loc else str(err["msg"])


def _parse_agents(raw: Any, warnings: list[str]) -> list[AgentConfig]:
    """Parse the ``agents:`` section. Entries without a usable ``name`` are
    skipped; duplicates and malformed entries are dropped (and reported)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append("agents: expected a list; section ignored")
        return []
    agents: list[AgentConfig] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        where = f"agents[{index}]"
        if not isinstance(entry, dict):
            warnings.append(f"{where}: expected a mapping; entry ignored")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            warnings.append(f"{where}: missing or empty 'name'; entry ignored")
            continue
        try:
            agent = AgentConfig.model_validate(entry)
        except ValidationError as exc:
            warnings.append(f"{where} ({name}): {_first_error(exc)}; entry ignored")
            continue
        if agent.name in seen:
            warnings.append(f"{where}: duplicate agent name '{agent.name}'; entry ignored")
            continue
        seen.add(agent.name)
        agents.append(agent)
    return agents


def _parse_personas(raw: Any, warnings: list[str]) -> dict[str, dict[str, str]]:
    """Parse the ``personas:`` section: name -> {file: content} (strings only)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        warnings.append("personas: expected a mapping; section ignored")
        return {}
    personas: dict[str, dict[str, str]] = {}
    for name, content in raw.items():
        if not isinstance(content, dict):
            warnings.append(f"personas.{name}: expected a mapping; entry ignored")
            continue
        personas[str(name)] = {
            str(key): str(value)
            for key, value in content.items()
            if isinstance(value, (str, int, float))
        }
    return personas


def _parse_interaction(raw: Any, warnings: list[str]) -> Interaction:
    """Parse the ``interaction:`` section, falling back to defaults."""
    if raw is None:
        return Interaction()
    if not isinstance(raw, dict):
        warnings.append("interaction: expected a mapping; section ignored")
        return Interaction()
    try:
        return Interaction.model_validate(raw)
    except ValidationError as exc:
        warnings.append(f"interaction: {_first_error(exc)}; section ignored")
        return Interaction()


def _parse_webhooks(raw: Any, warnings: list[str]) -> list[WebhookConfig]:
    """Parse the ``webhooks:`` section. Each entry needs a ``path`` (a string
    starting with ``/``); ``agent`` names the handling agent, ``secret``
    (optional) requires ``Authorization: Bearer <secret>``, and ``system_prompt``
    (optional) overrides the agent's default system prompt. Malformed or
    duplicate-path entries are dropped (and reported)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append("webhooks: expected a list; section ignored")
        return []
    webhooks: list[WebhookConfig] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        where = f"webhooks[{index}]"
        if not isinstance(entry, dict):
            warnings.append(f"{where}: expected a mapping; entry ignored")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            warnings.append(f"{where}: 'path' must be a string starting with '/'; entry ignored")
            continue
        normalized = dict(entry)
        secret = normalized.get("secret")
        if secret is not None and not isinstance(secret, str):
            if isinstance(secret, (int, float)):
                normalized["secret"] = str(secret)
            else:
                warnings.append(f"{where} ({path}): 'secret' must be a string; entry ignored")
                continue
        system_prompt = normalized.get("system_prompt")
        if system_prompt is not None:
            if not isinstance(system_prompt, str):
                warnings.append(
                    f"{where} ({path}): 'system_prompt' must be a string; entry ignored"
                )
                continue
            if not system_prompt.strip():
                normalized["system_prompt"] = None
        try:
            webhook = WebhookConfig.model_validate(normalized)
        except ValidationError as exc:
            warnings.append(f"{where} ({path}): {_first_error(exc)}; entry ignored")
            continue
        if webhook.path in seen:
            warnings.append(f"{where}: duplicate path '{webhook.path}'; entry ignored")
            continue
        seen.add(webhook.path)
        webhooks.append(webhook)
    return webhooks


class Provider(BaseModel):
    """A model endpoint: connection settings plus the models it offers."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    timeout: float = Field(default=DEFAULT_TIMEOUT, gt=0)
    models: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """A named agent: which provider + model it uses, its tools, and its persona."""

    model_config = ConfigDict(extra="ignore")

    name: str = DEFAULT_AGENT_NAME
    provider: str = DEFAULT_PROVIDER_NAME
    model: str = DEFAULT_MODEL

    # Optional opt-in list of built-in tool names. None (absent) -> all built-in
    # tools; [] -> no tools; [...] -> only the named tools.
    tools: list[str] | None = None

    # Optional persona (a name in ``personas:``) that customizes this agent,
    # e.g. via its ``SOUL.md`` (used as the system prompt).
    persona: str | None = None


class InteractionChat(BaseModel):
    """The agent named for the ``chat`` interaction."""

    model_config = ConfigDict(extra="ignore")

    agent: str = DEFAULT_AGENT_NAME


class Interaction(BaseModel):
    """Which named agent each interaction uses. Only ``chat`` exists for now."""

    model_config = ConfigDict(extra="ignore")

    chat: InteractionChat = Field(default_factory=InteractionChat)


class WebhookConfig(BaseModel):
    """A configured webhook: an HTTP ``path`` that triggers a specific ``agent``.

    When ``secret`` is set, requests must carry ``Authorization: Bearer <secret>``.
    When ``system_prompt`` is set, the agent runs with it instead of the agent's
    default (persona-based) system prompt.
    """

    model_config = ConfigDict(extra="ignore")

    path: str
    agent: str = DEFAULT_AGENT_NAME
    secret: str | None = None
    system_prompt: str | None = None


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
    """Top-level configuration: ``providers``, named ``agents``, ``personas``,
    ``interaction`` (which agent each interaction uses), and ``webhooks``.

    ``endpoint()`` resolves the effective :class:`Settings` for the loop;
    ``active_agent()`` picks the agent the ``chat`` loop uses; and
    ``system_prompt_for()`` resolves an agent's persona (e.g. ``SOUL.md``).
    """

    def __init__(
        self,
        providers: dict[str, Provider],
        agents: list[AgentConfig],
        personas: dict[str, dict[str, str]],
        interaction: Interaction,
        webhooks: list[WebhookConfig],
        warnings: list[str] | None = None,
    ) -> None:
        self.providers = providers
        self.agents = agents
        self.personas = personas
        self.interaction = interaction
        self.webhooks = webhooks
        # Human-readable notes about config entries that were ignored on load.
        self.warnings = warnings if warnings is not None else []

    @classmethod
    def load(cls) -> Config:
        """Load the config file. Raises :class:`ConfigError` if it cannot be parsed;
        entries that are individually malformed are dropped and listed in ``warnings``."""
        raw = _resolve_env_in_value(load_yaml(config_file_path()))
        warnings: list[str] = []
        providers = _parse_providers(raw.get("providers"), warnings)
        agents = _parse_agents(raw.get("agents"), warnings)
        personas = _parse_personas(raw.get("personas"), warnings)
        interaction = _parse_interaction(raw.get("interaction"), warnings)
        webhooks = _parse_webhooks(raw.get("webhooks"), warnings)
        for name, files in personas.items():
            if _valid_persona_name(name):
                ensure_persona(name, files.get(SOUL_FILENAME, ""))
            else:
                warnings.append(f"personas.{name}: invalid persona name; entry ignored")
        return cls(
            providers=providers,
            agents=agents,
            personas=personas,
            interaction=interaction,
            webhooks=webhooks,
            warnings=warnings,
        )

    def active_agent(self) -> AgentConfig:
        """The agent the ``chat`` loop uses: the one named by
        ``interaction.chat.agent``, else the first agent, else a default."""
        wanted = self.interaction.chat.agent
        for agent in self.agents:
            if agent.name == wanted:
                return agent
        if self.agents:
            return self.agents[0]
        return AgentConfig()

    def get_agent(self, name: str) -> AgentConfig | None:
        """The named agent, or ``None`` if no agent has that name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def endpoint(self) -> Settings:
        """Resolve the effective endpoint for the active (chat) agent."""
        return self.endpoint_for(self.active_agent())

    def endpoint_for(self, agent: AgentConfig) -> Settings:
        """Resolve the effective endpoint for ``agent``: env > agent/provider > default."""
        provider_name = _env_value("provider") or agent.provider
        provider = self.providers.get(provider_name) or Provider()
        file_values = {
            "model": agent.model,
            "base_url": provider.base_url,
            "api_key": provider.api_key,
            "timeout": provider.timeout,
        }
        return Settings.from_sources(file_values)

    def webhook_by_path(self, path: str) -> WebhookConfig | None:
        """The webhook whose ``path`` matches, or ``None``."""
        for webhook in self.webhooks:
            if webhook.path == path:
                return webhook
        return None

    def system_prompt_for(self, agent: AgentConfig) -> str:
        """The system prompt for ``agent``: its persona's stored ``SOUL.md``
        (under ``AGENT_RESPONDER_HOME``) if set, else the built-in default."""
        if agent.persona and _valid_persona_name(agent.persona):
            persona = self.personas.get(agent.persona)
            if persona:
                ensure_persona(agent.persona, persona.get(SOUL_FILENAME, ""))
                soul = persona_soul_path(agent.persona).read_text(encoding="utf-8")
                if soul.strip():
                    return soul
        return DEFAULT_SYSTEM_PROMPT
