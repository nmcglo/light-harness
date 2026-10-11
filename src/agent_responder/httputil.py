"""Small helpers shared by the stdlib HTTP servers (``web.py``, ``webhooks.py``)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler

MAX_BODY_BYTES = 1024 * 1024


class BodyError(Exception):
    """A request body that must be rejected; carries the HTTP status to send."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def read_body(handler: BaseHTTPRequestHandler, max_bytes: int = MAX_BODY_BYTES) -> bytes:
    """Read the request body, validating ``Content-Length`` before reading.

    Raises :class:`BodyError` (400 for a malformed/negative length, 413 when it
    exceeds ``max_bytes``) without consuming the oversized body.
    """
    header = handler.headers.get("Content-Length")
    if header is None or not header.strip():
        return b""
    try:
        length = int(header)
    except ValueError:
        raise BodyError(400, "invalid Content-Length") from None
    if length < 0:
        raise BodyError(400, "invalid Content-Length")
    if length > max_bytes:
        raise BodyError(413, f"request body exceeds {max_bytes} bytes")
    return handler.rfile.read(length) if length else b""
