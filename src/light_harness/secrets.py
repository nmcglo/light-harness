"""Secret registry: the model handles names, the harness handles values.

A secret is declared in the config's ``secrets:`` section by *reference*
(``env:NAME`` or ``file:/path``) plus a usage policy; the literal value never
appears in ``config.yaml``. Tools ask the :class:`SecretRegistry` for a secret
at the last moment (with the tool name, target host and the calling agent's
:class:`SecretContext`); the registry checks the policy and hands back a
:class:`SecretValue`, whose real value only :meth:`SecretValue.reveal` exposes.

Everything flowing back toward the model or the logs goes through a
:class:`Redactor`, built from every value the registry knows (including provider
API keys and webhook secrets), which replaces each value -- raw, base64,
URL-encoded or JSON-escaped -- with ``[REDACTED:name]``.

Policy vocabulary (all conditions must hold):

- ``allow.tools`` and ``allow.hosts``: required; a secret without them (or
  without an ``allow`` block at all) is usable by no tool.
- ``allow.agents`` / ``allow.interactions``: optional restrictions.
- ``confirm``: ``never`` | ``once-per-session`` | ``always``; anything but
  ``never`` needs an interactive chat to ask in, so it denies webhooks.
- ``expires``: a date after which the secret is denied.
- ``deny_after_untrusted``: deny once untrusted content (a webhook payload, a
  fetched issue/PR) is in the conversation.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn
from urllib.parse import quote, quote_plus, urlsplit

from pydantic import BaseModel, ConfigDict, Field

REFERENCE_PREFIX = "secret:"
SOURCE_SCHEMES = ("env", "file")
SECRET_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
# Values shorter than this are not redacted: replacing every occurrence of a
# two-letter string would wreck the transcript. Such values get a load warning.
MIN_REDACT_LENGTH = 6
INTERACTIONS = ("chat", "webhook")

ConfirmMode = Literal["never", "once-per-session", "always"]
# confirm(secret_name, tool, host) -> True to allow this use.
ConfirmFn = Callable[[str, str, str], bool]
AuditFn = Callable[[str], None]


class SecretDenied(Exception):  # noqa: N818 - reads as "secret denied" at call sites
    """A secret was requested but may not be used here. The message names the
    secret only; which rule failed goes to the audit log, never to the model."""

    def __init__(self, name: str) -> None:
        super().__init__(f"use of secret '{name}' was denied")
        self.name = name


class SecretLeakError(RuntimeError):
    """A registered secret value was about to leave the harness (pre-send guard)."""


class SecretValue:
    """A secret's value that does not print itself.

    ``str``/``repr``/``format`` give ``[REDACTED:name]``; :meth:`reveal` returns
    the value and is meant only for the spot that builds a header or env var.
    Not JSON-serializable and not picklable."""

    __slots__ = ("_value", "name")

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"[REDACTED:{self.name}]"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __reduce__(self) -> NoReturn:
        raise TypeError("SecretValue cannot be pickled")


class SecretPolicy(BaseModel):
    """The ``allow:`` block of a secret."""

    model_config = ConfigDict(extra="forbid")

    tools: list[str] = Field(default_factory=list)
    hosts: list[str] = Field(default_factory=list)
    agents: list[str] | None = None
    interactions: list[Literal["chat", "webhook"]] | None = None


class SecretSpec(BaseModel):
    """One entry of the ``secrets:`` section."""

    model_config = ConfigDict(extra="forbid")

    source: str
    allow: SecretPolicy | None = None
    confirm: ConfirmMode = "never"
    expires: date | None = None
    deny_after_untrusted: bool = False


@dataclass
class SecretContext:
    """Who is asking: one agent session. ``untrusted`` is set once untrusted
    content entered the conversation; ``confirmed`` remembers
    ``once-per-session`` approvals. ``bindings`` maps a purpose (``forgejo``) to
    a secret name, from ``agents[].secrets``."""

    agent: str
    interaction: Literal["chat", "webhook"]
    bindings: dict[str, str] = field(default_factory=dict)
    confirm: ConfirmFn | None = None
    untrusted: bool = False
    confirmed: set[str] = field(default_factory=set)


def parse_source(source: str) -> tuple[str, str] | None:
    """``(scheme, rest)`` of an ``env:NAME`` / ``file:PATH`` source, else ``None``."""
    scheme, sep, rest = source.partition(":")
    if not sep or scheme not in SOURCE_SCHEMES or not rest.strip():
        return None
    if scheme == "env" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rest):
        return None
    return scheme, rest.strip()


def file_source_path(rest: str, home: Path) -> Path:
    """The path a ``file:`` source names; relative paths are relative to ``home``."""
    path = Path(rest).expanduser()
    return path if path.is_absolute() else home / path


_KNOWN_TOKEN_RE = re.compile(
    r"^(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{16,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|[0-9a-f]{40})$"
)


def looks_like_secret(value: str) -> bool:
    """Heuristic for a literal credential pasted into the config: a known token
    shape, or a long unbroken string mixing letters and digits."""
    if _KNOWN_TOKEN_RE.match(value):
        return True
    return (
        len(value) >= 20
        and not re.search(r"\s", value)
        and bool(re.search(r"[A-Za-z]", value))
        and bool(re.search(r"\d", value))
    )


def host_matches(patterns: list[str], host: str) -> bool:
    """True if ``host`` (``name`` or ``name:port``) is listed. An entry without a
    port matches any port; matching is case-insensitive."""
    host = host.lower()
    hostname = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return any(p.lower() in (host, hostname) for p in patterns)


def host_of(url: str) -> str:
    """The ``host[:port]`` part of ``url`` (lowercased), or ``""``."""
    try:
        parts = urlsplit(url.strip())
        hostname = parts.hostname or ""
        port = parts.port
    except ValueError:
        return ""
    return f"{hostname}:{port}" if port else hostname


@dataclass
class _Entry:
    name: str
    source: str
    spec: SecretSpec | None  # None: implicit (provider/webhook field), no tool use


def _file_audit(path: Path) -> AuditFn:
    def write(line: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass  # auditing must not take the tool call down with it

    return write


class SecretRegistry:
    """Named secrets: lazy, cached resolution plus policy checks.

    ``home`` anchors relative ``file:`` paths; ``audit`` receives one line per
    decision (default: ``home/secrets-audit.log``)."""

    def __init__(self, home: Path, audit: AuditFn | None = None) -> None:
        self.home = home
        self._audit = audit if audit is not None else _file_audit(home / "secrets-audit.log")
        self._entries: dict[str, _Entry] = {}
        self._cache: dict[str, str | None] = {}

    # -- declaration -------------------------------------------------------

    def define(self, name: str, spec: SecretSpec) -> None:
        self._entries[name] = _Entry(name, spec.source, spec)

    def define_implicit(self, name: str, source: str | None = None, value: str = "") -> None:
        """Register a value used by a single config field (a provider ``api_key``,
        a webhook ``secret``): no tool may use it, but it is redacted everywhere."""
        if name in self._entries:
            return
        self._entries[name] = _Entry(name, source or "", None)
        if source is None:
            self._cache[name] = value or None

    def names(self) -> list[str]:
        """Names declared in ``secrets:`` (not the implicit field values)."""
        return [n for n, e in self._entries.items() if e.spec is not None]

    def spec(self, name: str) -> SecretSpec | None:
        entry = self._entries.get(name)
        return entry.spec if entry else None

    def file_paths(self) -> list[Path]:
        """Every path a ``file:`` source points at (for workspace overlap checks)."""
        paths = []
        for entry in self._entries.values():
            parsed = parse_source(entry.source)
            if parsed and parsed[0] == "file":
                paths.append(file_source_path(parsed[1], self.home))
        return paths

    # -- resolution --------------------------------------------------------

    def _resolve(self, name: str) -> str | None:
        if name in self._cache:
            return self._cache[name]
        entry = self._entries.get(name)
        value: str | None = None
        parsed = parse_source(entry.source) if entry else None
        if parsed is not None:
            scheme, rest = parsed
            if scheme == "env":
                value = os.environ.get(rest) or None
            else:
                try:
                    text = file_source_path(rest, self.home).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    text = ""
                value = text.rstrip("\r\n") or None
        self._cache[name] = value
        return value

    def is_resolved(self, name: str) -> bool:
        return self._resolve(name) is not None

    def resolve_field(self, name: str, field_name: str) -> str | None:
        """The value of ``name`` for a config field that references it (implicitly
        allowed); ``None`` if it is unknown or unresolved."""
        if name not in self._entries:
            return None
        value = self._resolve(name)
        self._log(name, field_name, "", None, "allowed" if value else "denied: unresolved")
        return value

    def values(self) -> dict[str, str]:
        """Every resolvable value, by name. For the redactor only."""
        out = {}
        for name in self._entries:
            value = self._resolve(name)
            if value:
                out[name] = value
        return out

    def invalidate(self, name: str) -> None:
        """Forget the cached value of ``name`` (after its backend was rewritten)."""
        self._cache.pop(name, None)

    # -- policy ------------------------------------------------------------

    def get(self, name: str, *, tool: str, host: str, context: SecretContext) -> SecretValue:
        """The secret ``name`` for ``tool`` talking to ``host`` on behalf of
        ``context``; raises :class:`SecretDenied` if any rule fails."""
        reason = self._deny_reason(name, tool=tool, host=host, context=context)
        if reason is None:
            value = self._resolve(name)
            if value is None:
                reason = "unresolved"
        self._log(name, tool, host, context, "allowed" if reason is None else f"denied: {reason}")
        if reason is not None:
            raise SecretDenied(name)
        assert value is not None
        return SecretValue(name, value)

    def find_for_host(self, host: str, tool: str) -> str | None:
        """The declared secret whose ``allow.hosts`` lists ``host``, preferring one
        whose ``allow.tools`` also lists ``tool``; ``None`` if none lists the host."""
        candidates = [
            e for e in self._entries.values()
            if e.spec is not None and e.spec.allow is not None
            and host_matches(e.spec.allow.hosts, host)
        ]  # fmt: skip
        for entry in candidates:
            assert entry.spec is not None and entry.spec.allow is not None
            if tool in entry.spec.allow.tools:
                return entry.name
        return candidates[0].name if candidates else None

    def _deny_reason(
        self, name: str, *, tool: str, host: str, context: SecretContext
    ) -> str | None:
        entry = self._entries.get(name)
        if entry is None:
            return "unknown secret"
        spec = entry.spec
        if spec is None or spec.allow is None:
            return "no allow policy"
        allow = spec.allow
        if tool not in allow.tools:
            return "tool not allowed"
        if not host or not host_matches(allow.hosts, host):
            return "host not allowed"
        if allow.agents is not None and context.agent not in allow.agents:
            return "agent not allowed"
        if allow.interactions is not None and context.interaction not in allow.interactions:
            return "interaction not allowed"
        if spec.expires is not None and datetime.now(UTC).date() > spec.expires:
            return "expired"
        if spec.deny_after_untrusted and context.untrusted:
            return "untrusted content in conversation"
        if spec.confirm != "never":
            if context.interaction != "chat" or context.confirm is None:
                return "confirmation required but not interactive"
            if spec.confirm == "once-per-session" and name in context.confirmed:
                return None
            if not context.confirm(name, tool, host):
                return "not confirmed by user"
            if spec.confirm == "once-per-session":
                context.confirmed.add(name)
        return None

    def _log(
        self, name: str, tool: str, host: str, context: SecretContext | None, decision: str
    ) -> None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        who = f" agent={context.agent} interaction={context.interaction}" if context else ""
        self._audit(f"{stamp} secret={name} tool={tool} host={host or '-'}{who} {decision}")


def _variants(value: str) -> set[str]:
    """Encodings of ``value`` that commonly show up in tool output or payloads."""
    raw = value.encode("utf-8")
    forms = {
        value,
        quote(value, safe=""),
        quote_plus(value, safe=""),
        json.dumps(value)[1:-1],
        json.dumps(value, ensure_ascii=False)[1:-1],
        base64.b64encode(raw).decode("ascii"),
        base64.urlsafe_b64encode(raw).decode("ascii"),
    }
    # base64 of the value at each byte alignment inside a larger blob: drop the
    # output characters that depend on neighbouring bytes.
    for pad in range(3):
        for encode in (base64.b64encode, base64.urlsafe_b64encode):
            text = encode(b"\0" * pad + raw).decode("ascii").rstrip("=")
            start = (pad * 4 + 2) // 3  # chars touched by the padding bytes
            end_bits = (pad + len(raw)) * 8 % 6
            core = text[start : len(text) - (1 if end_bits else 0)]
            if len(core) >= MIN_REDACT_LENGTH:
                forms.add(core)
    return {f for f in forms if len(f) >= MIN_REDACT_LENGTH}


class Redactor:
    """Replaces every known secret value (and its common encodings) with
    ``[REDACTED:name]``."""

    def __init__(self, values: dict[str, str]) -> None:
        pairs = sorted(
            ((form, name) for name, value in values.items() for form in _variants(value)),
            key=lambda p: -len(p[0]),
        )
        self._names = dict(pairs)
        self._re = (
            re.compile("|".join(re.escape(form) for form, _ in pairs)) if pairs else None
        )

    @classmethod
    def empty(cls) -> Redactor:
        return cls({})

    def __bool__(self) -> bool:
        return self._re is not None

    def find(self, text: str) -> str | None:
        """The name of the first secret found in ``text``, or ``None``."""
        if self._re is None:
            return None
        match = self._re.search(text)
        return self._names[match.group(0)] if match else None

    def redact(self, text: str) -> str:
        if self._re is None or not text:
            return text
        return self._re.sub(lambda m: f"[REDACTED:{self._names[m.group(0)]}]", text)

    def redact_obj(self, value: Any) -> Any:
        """``value`` with every string inside it redacted (dicts, lists)."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, list):
            return [self.redact_obj(v) for v in value]
        if isinstance(value, dict):
            return {k: self.redact_obj(v) for k, v in value.items()}
        return value

    def guard(self, payload: str) -> None:
        """Raise :class:`SecretLeakError` if ``payload`` contains a secret."""
        name = self.find(payload)
        if name is not None:
            raise SecretLeakError(
                f"refusing to send: the request contains secret '{name}'"
            )


class ToolSecrets:
    """What a tool is handed at construction: the registry plus the asking
    session's context. Tools ask by purpose and target URL; the model never
    names a secret."""

    def __init__(self, registry: SecretRegistry, context: SecretContext) -> None:
        self.registry = registry
        self.context = context

    def for_url(self, purpose: str, tool: str, url: str) -> SecretValue | None:
        """The secret for ``tool`` calling ``url``: the agent's binding for
        ``purpose`` if any, else the secret whose ``allow.hosts`` lists the URL's
        host. ``None`` when no secret is configured for that host; raises
        :class:`SecretDenied` when one is but its policy refuses this use."""
        host = host_of(url)
        name = self.context.bindings.get(purpose) or self.registry.find_for_host(host, tool)
        if name is None:
            return None
        return self.registry.get(name, tool=tool, host=host, context=self.context)

    def describe(self, tool_names: list[str]) -> str | None:
        """A system-prompt note on where this session's tools are authenticated:
        for each secret whose policy admits this agent and interaction, its
        allowed hosts and the allowed tools among ``tool_names``, plus whether it
        is set. Never names or values. ``None`` when there is nothing to say."""
        lines = []
        today = datetime.now(UTC).date()
        for name in self.registry.names():
            spec = self.registry.spec(name)
            assert spec is not None
            allow = spec.allow
            if allow is None or not allow.hosts:
                continue
            if allow.agents is not None and self.context.agent not in allow.agents:
                continue
            interactions = allow.interactions
            if interactions is not None and self.context.interaction not in interactions:
                continue
            if spec.expires is not None and today > spec.expires:
                continue
            if spec.confirm != "never" and self.context.interaction != "chat":
                continue
            tools = [t for t in tool_names if t in allow.tools]
            if not tools:
                continue
            line = f"- hosts {', '.join(allow.hosts)}: {', '.join(tools)}"
            if not self.registry.is_resolved(name):
                line += " (credential not set; calls will be refused)"
            elif spec.confirm != "never":
                line += " (the user is asked to approve each use)"
            if spec.deny_after_untrusted:
                line += " (unavailable once untrusted content is in the conversation)"
            lines.append(line)
        if not lines:
            return None
        return (
            "Authenticated access: the harness attaches credentials itself; never ask "
            "for, pass or print tokens. Calls to other hosts are unauthenticated.\n"
            + "\n".join(lines)
        )


def with_secret_note(system_prompt: str, secrets: ToolSecrets, tool_names: list[str]) -> str:
    """``system_prompt`` plus :meth:`ToolSecrets.describe`, if it has anything."""
    note = secrets.describe(tool_names)
    return f"{system_prompt}\n\n{note}" if note else system_prompt
