"""Small helpers shared by the HTTP server tests."""

from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer


def join_thread(thread: threading.Thread, timeout: float = 5.0) -> None:
    """Join a server thread and fail if it is still running: ``join`` alone returns
    silently on timeout, so a wedged handler thread would otherwise pass."""
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "server thread did not stop within the timeout"


def stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    join_thread(thread)
