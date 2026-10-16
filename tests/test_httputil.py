"""Body-reading rules shared by both HTTP servers (httputil.py), tested without sockets
so what the server consumed from the stream can be observed directly."""

from __future__ import annotations

import io
from email.message import Message
from types import SimpleNamespace

import pytest

from light_harness.httputil import BodyError, read_body


def _handler(headers: dict[str, str], stream: bytes) -> SimpleNamespace:
    message = Message()
    for name, value in headers.items():
        message[name] = value
    return SimpleNamespace(headers=message, rfile=io.BytesIO(stream), close_connection=False)


def test_an_oversized_content_length_is_refused_without_reading_any_of_the_body() -> None:
    handler = _handler({"Content-Length": "5000"}, b"x" * 5000)

    with pytest.raises(BodyError) as refused:
        read_body(handler, max_bytes=1000)  # type: ignore[arg-type]

    assert refused.value.status == 413
    assert handler.rfile.tell() == 0  # not a byte of it was consumed
    assert handler.close_connection is True  # ... so the connection cannot be reused


def test_an_oversized_chunk_is_refused_after_its_size_line_only() -> None:
    stream = b"fa0\r\n" + b"x" * 4000 + b"\r\n0\r\n\r\n"  # a 4000-byte chunk
    handler = _handler({"Transfer-Encoding": "chunked"}, stream)

    with pytest.raises(BodyError) as refused:
        read_body(handler, max_bytes=1000)  # type: ignore[arg-type]

    assert refused.value.status == 413
    assert handler.rfile.tell() == len(b"fa0\r\n")  # the size line, never the chunk
    assert handler.close_connection is True


@pytest.mark.parametrize("length", ["abc", "-5", "1.5"])
def test_a_malformed_content_length_is_refused_unread(length: str) -> None:
    handler = _handler({"Content-Length": length}, b"hello")

    with pytest.raises(BodyError) as refused:
        read_body(handler)  # type: ignore[arg-type]

    assert refused.value.status == 400
    assert handler.rfile.tell() == 0
    assert handler.close_connection is True


def test_a_body_within_the_limit_is_read_exactly_and_the_connection_stays_open() -> None:
    handler = _handler({"Content-Length": "5"}, b"helloGET /next")

    assert read_body(handler) == b"hello"  # type: ignore[arg-type]
    assert handler.rfile.read() == b"GET /next"  # the next request is untouched
    assert handler.close_connection is False
