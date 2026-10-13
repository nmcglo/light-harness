"""Small helpers shared by the stdlib HTTP servers (``web.py``, ``webhooks.py``)."""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

MAX_BODY_BYTES = 1024 * 1024

_CHUNK_END = (b"\r\n", b"\n")


def secret_matches(supplied: str, expected: str) -> bool:
    """Constant-time comparison of a header value against a configured secret.

    ``hmac.compare_digest`` raises ``TypeError`` when both operands are ``str``
    and either contains non-ASCII, which crashes the handler mid-request with no
    response. Compare bytes instead: ``http.server`` decodes headers as
    iso-8859-1, so re-encoding ``supplied`` as iso-8859-1 recovers the exact
    bytes the client put on the wire, and the config ``expected`` is compared in
    its canonical UTF-8 form. A client that sends the secret as UTF-8 bytes
    (curl, HTTPie) therefore authenticates; any other value -- non-ASCII or
    not -- simply fails the comparison.
    """
    try:
        supplied_bytes = supplied.encode("iso-8859-1")
        expected_bytes = expected.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        # A parsed header value is always iso-8859-1-encodable; anything else
        # (or an unencodable secret) can never match a wire value.
        return False
    return hmac.compare_digest(supplied_bytes, expected_bytes)


_WILDCARD_BINDS = frozenset({"", "0.0.0.0", "::"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_LOOPBACK_SUFFIX = ".localhost"


def host_allow_list(bind_host: str, extra: Iterable[str] = ()) -> frozenset[str]:
    """The hostnames a request's ``Host`` header may carry, on top of loopback
    names, IP literals and the bind host itself (see :func:`host_is_allowed`).

    ``extra`` is what the operator names explicitly: a mDNS name, the name a
    reverse proxy forwards. A wildcard bind contributes no name of its own, but
    it does not switch the check off -- that is where the check matters most,
    because a browser on this machine can be made to reach the server under a
    hostname an attacker controls.
    """
    names = set(_LOOPBACK_HOSTS) | {name.strip().lower() for name in extra if name.strip()}
    if bind_host.strip() not in _WILDCARD_BINDS:
        names.add(bind_host.strip().lower())
    return frozenset(names)


def host_is_allowed(handler: BaseHTTPRequestHandler, allowed_hosts: frozenset[str]) -> bool:
    """True when the request's ``Host`` header names this server, rather than a
    hostname that merely resolves to it.

    Accepted: the names from :func:`host_allow_list` -- loopback names, the bind host,
    and whatever the operator named -- plus ``*.localhost`` (which browsers resolve to
    this machine) and any IP literal (LAN access by address). Everything else,
    including a missing or unparseable ``Host``, is refused.

    The refused shape is DNS rebinding: an attacker's page at ``evil.example``
    whose DNS now answers ``127.0.0.1``, so the browser speaks to the local server
    while still sending ``Host: evil.example``. Without this check it is served
    the page that embeds the API token, and with that token it reads (and rewrites)
    a config holding API keys and webhook secrets. The hostname is never used to
    route anything here, so it is checked and nothing else.
    """
    raw = handler.headers.get("Host")
    if raw is None:
        return False
    try:
        hostname = urlsplit("//" + raw.strip()).hostname
    except (ValueError, UnicodeError):
        return False  # an invalid port or a host name that cannot be encoded
    if hostname is None:
        return False
    hostname = hostname.rstrip(".")  # "localhost." is "localhost"
    if hostname in allowed_hosts or hostname.endswith(_LOOPBACK_SUFFIX):
        return True
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


class BodyError(Exception):
    """A request body that must be rejected; carries the HTTP status to send."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def declared_body(handler: BaseHTTPRequestHandler) -> bool:
    """True when the request declares a body.

    Bytes left on a connection that stays open would be read as the next request,
    so a handler that answers without consuming the body has to close the
    connection. :func:`read_body` does that for every body it refuses; the
    handlers that answer before reading one (auth failure, unknown path) use this.
    """
    if (handler.headers.get("Transfer-Encoding") or "").strip():
        return True
    raw = handler.headers.get("Content-Length")
    if not raw or not raw.strip():
        return False
    try:
        return int(raw) > 0
    except ValueError:
        return True  # an unreadable length can't be skipped, so the stream is lost


def read_body(handler: BaseHTTPRequestHandler, max_bytes: int = MAX_BODY_BYTES) -> bytes:
    """Read the request body in full, or raise BodyError without consuming it.

    ``Transfer-Encoding: chunked`` is decoded (chunk-size extensions and trailer
    headers tolerated) and outranks ``Content-Length``, as RFC 9112 section 3.3.3
    requires; any other transfer coding is refused. A malformed or oversized body
    is rejected on a connection marked for close, because its bytes stay on the
    socket.
    """
    encoding = (handler.headers.get("Transfer-Encoding") or "").strip().lower()
    if encoding:
        if encoding != "chunked":
            raise _refuse(handler, 400, "unsupported Transfer-Encoding")
        return _read_chunked(handler, max_bytes)
    raw = handler.headers.get("Content-Length")
    if raw is None or not raw.strip():
        return b""
    try:
        length = int(raw)
    except ValueError:
        raise _refuse(handler, 400, "invalid Content-Length") from None
    if length < 0:
        raise _refuse(handler, 400, "invalid Content-Length")
    if length > max_bytes:
        raise _refuse(handler, 413, f"request body exceeds {max_bytes} bytes")
    return handler.rfile.read(length) if length else b""


def _refuse(handler: BaseHTTPRequestHandler, status: int, message: str) -> BodyError:
    """The BodyError to raise for a body that is left unread, closing the
    connection so those bytes are not read as the next request."""
    handler.close_connection = True
    return BodyError(status, message)


def _read_chunked(handler: BaseHTTPRequestHandler, max_bytes: int) -> bytes:
    """Decode a chunked body, keeping the accumulated size under ``max_bytes``.

    The last chunk (size 0) is followed by optional trailer headers; they are
    consumed without being treated as body content, so the stream stays aligned.
    """
    parts: list[bytes] = []
    total = 0
    while True:
        line = handler.rfile.readline(65537)
        if not line:
            raise _refuse(handler, 400, "truncated chunked body") from None
        try:
            size = int(line.strip().split(b";", 1)[0], 16)
        except ValueError:
            raise _refuse(handler, 400, "invalid chunk size") from None
        if size < 0:
            raise _refuse(handler, 400, "invalid chunk size") from None
        if size == 0:
            while True:
                trailer = handler.rfile.readline(65537)
                if trailer in _CHUNK_END:
                    break
                if not trailer:
                    raise _refuse(handler, 400, "truncated chunked body") from None
            break
        if total + size > max_bytes:
            raise _refuse(handler, 413, f"request body exceeds {max_bytes} bytes") from None
        chunk = handler.rfile.read(size)
        if len(chunk) != size:
            raise _refuse(handler, 400, "truncated chunked body") from None
        parts.append(chunk)
        total += size
        if handler.rfile.readline(65537) not in _CHUNK_END:
            raise _refuse(handler, 400, "chunk not terminated") from None
    return b"".join(parts)
