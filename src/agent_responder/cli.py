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
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from .agent import Agent
from .config import (
    Config,
    ConfigError,
    Settings,
    ensure_persona,
    load_dotenv,
    persona_sessions_dir,
)
from .llm import OpenAICompatibleClient
from .sessions import SessionLog
from .tools import ToolRegistry, build_tool_registry
from .web import serve_config
from .webhooks import DEFAULT_HOST, DEFAULT_PORT, serve_webhooks

app = typer.Typer(help="Agentic responder scaffold.")
console = Console()
err_console = Console(stderr=True)


@app.callback()
def _main() -> None:
    """Load ``./.env`` (without overriding variables already set) before any command."""
    try:
        load_dotenv()
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
    system_prompt = config.system_prompt_for(active)
    persona_name = active.persona or active.name
    try:
        ensure_persona(persona_name)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    session = SessionLog(
        persona_sessions_dir(persona_name),
        meta={"persona": persona_name, "agent": active.name, "model": settings.model},
    )
    agent = _build_agent(settings, registry, system_prompt, session=session)
    try:
        asyncio.run(_run_agent(agent, prompt))
    finally:
        session.close()


@app.command()
def tools() -> None:
    """List the tools the agent will use (per the current config)."""
    active = _load_config().active_agent()
    registry = build_tool_registry(active.tools, active.workspace_path())
    for spec in registry.tool_schemas():
        fn = spec["function"]
        console.print(f"[bold]{fn['name']}[/bold] -- {fn['description']}")


@app.command()
def web(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8765,
    cors_origin: Annotated[
        str,
        typer.Option(
            "--cors-origin",
            help="Access-Control-Allow-Origin for /api/* (empty disables CORS).",
        ),
    ] = "*",
) -> None:
    """Serve a web UI for viewing and editing the config file."""
    serve_config(host=host, port=port, cors_origin=cors_origin)


@app.command()
def webhooks(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = DEFAULT_PORT,
) -> None:
    """Serve the configured webhook listeners (the ``webhooks:`` section)."""
    serve_webhooks(_load_config(), host=host, port=port)


async def _run_agent(agent: Agent, prompt: str) -> None:
    try:
        if prompt:
            await _run_turn(agent, prompt)
        else:
            await _run_repl(agent)
    finally:
        await agent.client.close()


async def _run_turn(agent: Agent, prompt: str) -> None:
    console.print(await agent.run(prompt))


async def _run_repl(agent: Agent) -> None:
    console.print(
        "[bold]Agentic responder[/bold] -- type a prompt, or [italic]exit[/italic] to quit."
    )
    while True:
        try:
            line = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        console.print(await agent.run(line))
    console.print("bye")


if __name__ == "__main__":
    app()
