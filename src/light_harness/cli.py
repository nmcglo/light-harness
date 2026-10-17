"""Command-line interface.

``agent chat "prompt"`` runs a single turn; ``agent chat`` (no prompt) drops into
an interactive REPL. ``agent tools`` lists the registered tools.
``agent web`` serves a small web UI for viewing and editing the config file.
``agent webhooks`` serves the configured webhook listeners.

The commands are synchronous Typer commands that drive the async agent via
``asyncio.run`` -- this keeps the entrypoint portable across Typer versions.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Annotated

import httpx
import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from .agent import Agent
from .config import (
    Config,
    ConfigError,
    Settings,
    config_home,
    ensure_persona,
    load_default_dotenv,
    persona_sessions_dir,
)
from .llm import LLMError, OpenAICompatibleClient
from .sessions import SessionLog
from .tools import ToolRegistry, build_tool_registry
from .tui import TurnView, glyphs_for, is_interactive
from .web import DEFAULT_HOST as WEBUI_DEFAULT_HOST
from .web import DEFAULT_PORT as WEBUI_DEFAULT_PORT
from .web import ConfigServer, start_config_server
from .webhooks import DEFAULT_HOST, DEFAULT_PORT, serve_webhooks

app = typer.Typer(help="Run LLM agents in response to chat prompts and webhooks.")
console = Console()
err_console = Console(stderr=True)


def _tolerate_unencodable_output() -> None:
    """Replace characters the terminal's encoding cannot represent instead of
    crashing on them (model replies and Markdown bullets are not always ASCII)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


@app.callback()
def _main() -> None:
    """Load ``./.env`` and ``$LIGHT_HARNESS_HOME/.env`` (without overriding variables
    already set) before any command."""
    _tolerate_unencodable_output()
    try:
        load_default_dotenv()
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None


def _load_config() -> Config:
    """Load the config; exit with a clear error if the file is broken, and print a
    warning for each entry that was ignored."""
    try:
        config = Config.load()
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    for warning in config.warnings:
        err_console.print(f"[yellow]Config warning:[/yellow] {escape(warning)}", highlight=False)
    return config


def _build_agent(
    settings: Settings,
    tools: ToolRegistry,
    system_prompt: str,
    session: SessionLog | None = None,
) -> Agent:
    return Agent(
        client=OpenAICompatibleClient(settings),
        tools=tools,
        system_prompt=system_prompt,
        session=session,
    )


@app.command()
def chat(
    prompt: Annotated[
        str,
        typer.Argument(help="Prompt to send. Omit to start an interactive REPL."),
    ] = "",
    model: Annotated[str | None, typer.Option("--model", help="Model name.")] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="OpenAI-compatible base URL.")
    ] = None,
    webui: WebuiOpt = False,
    webui_host: WebuiHostOpt = WEBUI_DEFAULT_HOST,
    webui_port: WebuiPortOpt = WEBUI_DEFAULT_PORT,
) -> None:
    """Run a single prompt, or drop into an interactive REPL."""
    config = _load_config()
    active = config.active_agent()
    settings = config.endpoint()
    if model:
        settings.model = model
    if base_url:
        settings.base_url = base_url
    if not settings.api_key:
        console.print(
            "[red]No API key found.[/red] Set `api_key` on the agent's provider in "
            "config.yaml (e.g. `api_key: $MY_KEY`)."
        )
        raise typer.Exit(code=1)

    registry = build_tool_registry(active.tools, active.workspace_path())
    persona_name = active.persona or active.name
    try:
        system_prompt = config.system_prompt_for(active)
        ensure_persona(persona_name)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    session = SessionLog(
        persona_sessions_dir(persona_name),
        meta={"persona": persona_name, "agent": active.name, "model": settings.model},
    )
    agent = _build_agent(settings, registry, system_prompt, session=session)
    ui = _start_webui(webui, webui_host, webui_port)
    try:
        asyncio.run(_run_agent(agent, prompt))
    finally:
        _stop_webui(ui)
        session.close()


@app.command()
def tools() -> None:
    """List the tools the agent will use (per the current config)."""
    config = _load_config()
    active = config.active_agent()
    registry = build_tool_registry(active.tools, active.workspace_path())
    for spec in registry.tool_schemas():
        fn = spec["function"]
        console.print(f"[bold]{fn['name']}[/bold] -- {fn['description']}")


WebuiOpt = Annotated[
    bool, typer.Option("--webui", help="Also serve the config web UI in the background.")
]
WebuiHostOpt = Annotated[
    str, typer.Option("--webui-host", help="Web UI bind host (with --webui).")
]
WebuiPortOpt = Annotated[
    int, typer.Option("--webui-port", help="Web UI bind port (with --webui).")
]


def _start_webui(enabled: bool, host: str, port: int) -> ConfigServer | None:
    if not enabled:
        return None
    try:
        return start_config_server(host=host, port=port)
    except OSError as exc:  # port busy / bad host
        err_console.print(f"[red]Error:[/red] web UI: {exc}")
        raise typer.Exit(code=1) from None


def _stop_webui(server: ConfigServer | None) -> None:
    if server is not None:
        server.shutdown()
        server.server_close()


@app.command()
def webhooks(
    host: Annotated[
        str | None,
        typer.Option(
            "--host", help=f"Bind host (default: webhook_server.host or {DEFAULT_HOST})."
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port", help=f"Bind port (default: webhook_server.port or {DEFAULT_PORT})."
        ),
    ] = None,
    allowed_host: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-host",
            help=(
                "Host name this server answers for, besides loopback names, IP "
                "addresses and the bind host. Repeat for every name it is reached by. "
                "Replaces webhook_server.allowed_hosts when given."
            ),
        ),
    ] = None,
    output_log: Annotated[
        bool | None,
        typer.Option(
            "--output-log/--no-output-log",
            help=(
                "Print a log line for each accepted webhook call, naming the "
                "session log file of its run. Overrides webhook_server.output_log "
                "when given."
            ),
        ),
    ] = None,
    webui: WebuiOpt = False,
    webui_host: WebuiHostOpt = WEBUI_DEFAULT_HOST,
    webui_port: WebuiPortOpt = WEBUI_DEFAULT_PORT,
) -> None:
    """Serve the configured webhook listeners (the ``webhooks:`` section)."""
    config = _load_config()
    ui = _start_webui(webui, webui_host, webui_port)
    try:
        server = config.webhook_server
        if output_log is not None:
            # The handler reads the flag off the config, so the CLI override
            # lands there.
            server.output_log = output_log
        serve_webhooks(
            config,
            host=host or server.host or DEFAULT_HOST,
            port=port or server.port or DEFAULT_PORT,
            allowed_hosts=allowed_host or server.allowed_hosts,
        )
    finally:
        _stop_webui(ui)



async def _run_agent(agent: Agent, prompt: str) -> None:
    try:
        if prompt:
            await _run_turn(agent, prompt)
        else:
            await _run_repl(agent)
    finally:
        await agent.client.close()


async def _run_turn(agent: Agent, prompt: str) -> None:
    try:
        interrupted = await _stream_turn(agent, prompt)
    except Exception as exc:  # noqa: BLE001 - report any failed run, not a traceback
        err_console.print(f"[red]Error:[/red] {_describe_failure(exc)}")
        raise typer.Exit(code=1) from None
    if interrupted:
        raise typer.Exit(code=130)


def _describe_failure(exc: Exception) -> str:
    """A one-line, actionable description of why a turn failed (Rich markup-safe)."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return f"authentication failed (HTTP {status}) -- check the provider's api_key"
        if status == 404:
            return "not found (HTTP 404) -- check the provider's base_url and the model name"
        if status == 429:
            return "rate limited (HTTP 429) -- wait a moment and try again"
        if status >= 500:
            return f"the model endpoint had a server error (HTTP {status}) -- try again shortly"
        return f"the model endpoint returned HTTP {status}"
    if isinstance(exc, httpx.TransportError):
        try:
            host = exc.request.url.host
        except RuntimeError:  # httpx raises this when the exception has no request
            host = ""
        where = f" {host}" if host else " the model endpoint"
        if isinstance(exc, httpx.TimeoutException):
            return f"the request to{where} timed out -- it may be slow or unreachable"
        return f"could not reach{where} -- check the provider's base_url and your network"
    if isinstance(exc, LLMError):
        return escape(str(exc))
    return escape(f"{type(exc).__name__}: {exc}")


async def _stream_turn(agent: Agent, prompt: str, *, label: bool = False) -> bool:
    """Run one turn, rendering the reply as Markdown while it streams in and
    showing each tool call as a dim line. A client that does not stream still gets
    its final answer printed.

    Ctrl-C while the turn runs cancels just this turn (the agent keeps what had
    streamed so far in its history); returns True if the turn was interrupted."""
    view = TurnView(console, label=label)
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(agent.run(prompt, on_token=view.token, on_tool=view.tool))
    interrupted = False

    def interrupt() -> None:
        nonlocal interrupted
        interrupted = True
        task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, interrupt)
        handler_installed = True
    except (NotImplementedError, ValueError, RuntimeError):  # e.g. Windows, non-main thread
        handler_installed = False
    try:
        answer = await task
    except asyncio.CancelledError:
        view.abort()
        if not interrupted:
            raise
        console.print(f"[yellow]{glyphs_for(console).stop} interrupted[/yellow]")
        return True
    except BaseException:
        view.abort()
        raise
    finally:
        if handler_installed:
            loop.remove_signal_handler(signal.SIGINT)
    view.finish(answer)
    return False


_COMMANDS = {
    "/help": "show the available commands",
    "/clear": "forget the conversation so far",
    "/tools": "list the tools the agent can use",
    "/exit": "leave the chat",
    "/quit": "leave the chat",
}

ReadLine = Callable[[], Awaitable[str]]


class _CommandCompleter(Completer):
    """Complete ``/commands`` only while the whole input is a single ``/word``, so
    nothing pops up after a space or in the middle of a prompt."""

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or any(c.isspace() for c in text):
            return
        for command, description in _COMMANDS.items():
            if command.startswith(text.lower()) and command != text:
                yield Completion(command, start_position=-len(text), display_meta=description)


def _make_reader() -> ReadLine:
    """Return an async line reader. On a terminal it is a prompt_toolkit session with
    persistent history, slash-command completion and Alt-Enter for a newline;
    otherwise (piped input) it falls back to plain ``console.input``."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("TERM") == "dumb":

        async def plain() -> str:
            return console.input(f"\n[bold cyan]{glyphs_for(console).prompt}[/bold cyan] ")

        return plain

    glyphs = glyphs_for(console)
    sep = glyphs.sep

    history: History
    try:
        path = config_home() / "history"
        path.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(path))
    except OSError:
        history = InMemoryHistory()
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def _clear_or_quit(event: KeyPressEvent) -> None:
        """Ctrl-C clears what has been typed; on an empty prompt it leaves the chat."""
        buffer = event.current_buffer
        if buffer.text:
            buffer.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    session: PromptSession[str] = PromptSession(
        history=history,
        completer=_CommandCompleter(),
        complete_while_typing=True,
        key_bindings=bindings,
        bottom_toolbar=(
            f" Enter send {sep} Alt-Enter newline {sep} / commands {sep} Ctrl-D quit"
        ),
        style=Style.from_dict(
            {
                "": "bold",  # what the user types stands out from the output
                # The default is a reverse-video bar; use dim grey text, no fill.
                "bottom-toolbar": "noreverse nobold bg:default fg:#6c6c6c",
                "bottom-toolbar.text": "noreverse nobold bg:default fg:#6c6c6c",
            }
        ),
    )

    async def interactive() -> str:
        console.print()
        return await session.prompt_async(
            FormattedText([("bold ansicyan", f"{glyphs.prompt} ")])
        )

    return interactive


def _run_command(agent: Agent, line: str) -> bool:
    """Handle a ``/command``; returns True when the REPL should exit."""
    name = line.split()[0].lower()
    if name in {"/exit", "/quit"}:
        return True
    if name == "/help":
        for command, text in _COMMANDS.items():
            console.print(f"  [cyan]{command:<7}[/cyan] [dim]{text}[/dim]")
    elif name == "/clear":
        agent.reset()
        console.print("[dim]conversation cleared[/dim]")
    elif name == "/tools":
        specs = agent.tools.tool_schemas()
        if not specs:
            console.print("[dim]no tools enabled[/dim]")
        for spec in specs:
            fn = spec["function"]
            console.print(f"  [bold]{fn['name']}[/bold] [dim]{escape(fn['description'])}[/dim]")
    else:
        console.print(f"[red]unknown command[/red] {escape(name)} [dim](try /help)[/dim]")
    return False


async def _run_repl(agent: Agent, read_line: ReadLine | None = None) -> None:
    read_line = read_line or _make_reader()
    if is_interactive(console):
        console.print(
            Panel(
                "Type a prompt and press [bold]Enter[/bold].  "
                "[cyan]/help[/cyan] lists commands, [cyan]/exit[/cyan] leaves.  "
                "[bold]Ctrl-C[/bold] stops a reply.",
                title="[bold magenta]light-harness[/bold magenta]",
                title_align="left",
                border_style="magenta",
                padding=(0, 2),
            )
        )
    else:
        console.print("light-harness -- type a prompt, /help for commands, /exit to leave.")
    while True:
        try:
            line = (await read_line()).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            if _run_command(agent, line):
                break
            continue
        try:
            started = time.monotonic()
            await _stream_turn(agent, line, label=True)
            if is_interactive(console):
                console.rule(
                    f"[dim]{time.monotonic() - started:.1f}s[/dim]", style="dim", align="right"
                )
        except Exception as exc:  # noqa: BLE001 - one failed turn must not end the session
            err_console.print(
                f"[red]{glyphs_for(err_console).error} Error:[/red] {_describe_failure(exc)}"
            )
    console.print(f"\n[dim]bye{glyphs_for(console).bye}[/dim]")


if __name__ == "__main__":
    app()
