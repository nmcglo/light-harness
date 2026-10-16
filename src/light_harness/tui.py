"""Terminal rendering of one agent turn: streamed Markdown, tool-call lines, spinner.

Streaming Markdown through a single ``rich.live.Live`` breaks once the answer is
taller than the terminal (rewritten lines have scrolled out of reach). So the text
is split into blocks at blank lines (never inside a code fence): finished blocks are
printed permanently as Markdown, and only the short in-progress tail lives in a
``Live`` region.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.status import Status

_FENCES = ("```", "~~~")


@dataclass(frozen=True)
class Glyphs:
    """The symbols the TUI draws with."""

    prompt: str
    agent: str
    tool: str
    arrow: str
    stop: str
    error: str
    ellipsis: str
    sep: str
    bye: str
    spinner: str


UNICODE = Glyphs("❯", "●", "⚙", "→", "⏹", "✗", "…", "·", " 👋", "dots")
ASCII = Glyphs(">", "*", "*", "->", "!", "x", "...", "|", "", "line")


def glyphs_for(console: Console) -> Glyphs:
    """ASCII symbols when the terminal cannot show Unicode (non-UTF-8 encoding or
    ``TERM=dumb``), otherwise the Unicode set."""
    if console.options.ascii_only or os.environ.get("TERM") == "dumb":
        return ASCII
    return UNICODE


def is_interactive(console: Console) -> bool:
    """True when ``console`` is a real, capable terminal (so spinners, live
    redraws and decorative panels make sense); false for pipes and dumb terminals."""
    return console.is_terminal and not console.is_dumb_terminal


def split_blocks(text: str) -> tuple[list[str], str]:
    """Split ``text`` into finished Markdown blocks and the unfinished tail.

    A block ends at a blank line outside a code fence. The last, unterminated line
    always belongs to the tail, as does an open code fence."""
    blocks: list[str] = []
    current: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):  # still being written
            current.append(line)
            break
        stripped = line.strip()
        if fence is None and stripped.startswith(_FENCES):
            fence = stripped[:3]
        elif fence is not None and stripped.startswith(fence):
            fence = None
        if not stripped and fence is None and current:
            blocks.append("".join(current))
            current = []
        elif stripped or current:
            current.append(line)
    return blocks, "".join(current)


def tool_line(
    name: str, arguments: dict[str, Any], result: str, glyphs: Glyphs = UNICODE
) -> str:
    """One dim, single-line summary of a finished tool call (Rich markup)."""
    args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    if len(args) > 60:
        args = args[:59] + glyphs.ellipsis
    first = result.strip().splitlines()[0] if result.strip() else ""
    if len(first) > 80:
        first = first[:79] + glyphs.ellipsis
    failed = first.startswith("Error")
    color = "red" if failed else "dim"
    return (
        f"[dim]{glyphs.tool} {escape(name)}({escape(args)}) {glyphs.arrow}[/dim] "
        f"[{color}]{escape(first)}[/{color}]"
    )


class TurnView:
    """Renders one turn. Call :meth:`token` / :meth:`tool` as events arrive and
    :meth:`finish` (or :meth:`abort`) at the end."""

    def __init__(self, console: Console, *, label: bool = False) -> None:
        self._console = console
        self._glyphs = glyphs_for(console)
        self._label = label
        self._buffer = ""
        self._headed = False
        self._status: Status | None = None
        self._live: Live | None = None
        self._spin()

    @property
    def started(self) -> bool:
        return self._headed

    # -- events -------------------------------------------------------------

    def token(self, text: str) -> None:
        self._stop_spin()
        self._head()
        self._buffer += text
        blocks, tail = split_blocks(self._buffer)
        if blocks:
            self._stop_live()
            for block in blocks:
                self._console.print(Markdown(block))
            self._buffer = tail
        self._show_tail()

    def tool(self, name: str, arguments: dict[str, Any], result: str) -> None:
        self._flush()
        self._console.print(tool_line(name, arguments, result, self._glyphs))
        self._spin()

    def finish(self, answer: str) -> None:
        """End of the turn. If nothing streamed, show ``answer`` as the reply."""
        self._stop_spin()
        if not self._headed:
            self._head()
            self._buffer = answer
        self._flush()

    def abort(self) -> None:
        """The turn ended early (interrupt or error): keep what was shown."""
        self._stop_spin()
        self._flush()

    # -- internals ----------------------------------------------------------

    def _head(self) -> None:
        if not self._headed:
            self._headed = True
            if self._label:
                agent = self._glyphs.agent
                self._console.print(f"[bold green]{agent}[/bold green] [dim]agent[/dim]")

    def _show_tail(self) -> None:
        if not is_interactive(self._console) or not self._buffer.strip():
            return
        renderable = Markdown(self._buffer)
        if self._live is None:
            self._live = Live(
                renderable,
                console=self._console,
                refresh_per_second=10,
                transient=True,
                vertical_overflow="ellipsis",
            )
            self._live.start()
        else:
            self._live.update(renderable)

    def _flush(self) -> None:
        self._stop_live()
        if self._buffer.strip():
            self._console.print(Markdown(self._buffer))
        self._buffer = ""

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _spin(self) -> None:
        if self._status is None:
            glyphs = self._glyphs
            self._status = self._console.status(
                f"[dim]thinking{glyphs.ellipsis}[/dim]", spinner=glyphs.spinner
            )
            self._status.start()

    def _stop_spin(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
