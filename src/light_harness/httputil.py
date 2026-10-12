"""Small helpers shared by the stdlib HTTP servers (``web.py``, ``webhooks.py``)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler

MAX_BODY_BYTES = 1024 * 1024

_CHUNK_END = (b"\r\n", b"\n")


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
