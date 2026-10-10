"""Command-line interface.

``agent chat "prompt"`` runs a single turn; ``agent chat`` (no prompt) drops into
an interactive REPL. ``agent tools`` lists the registered tools.

The commands are synchronous Typer commands that drive the async agent via
``asyncio.run`` -- this keeps the entrypoint portable across Typer versions.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console

from .agent import Agent
from .config import Config, Settings
from .llm import OpenAICompatibleClient
from .tools import ToolRegistry, build_tool_registry

app = typer.Typer(help="Agentic responder scaffold.")
console = Console()


def _build_agent(settings: Settings, tools: ToolRegistry) -> Agent:
    return Agent(client=OpenAICompatibleClient(settings), tools=tools)


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
    config = Config.load()
    settings = config.endpoint()
    if model:
        settings.model = model
    if base_url:
        settings.base_url = base_url
    if not settings.api_key:
        console.print(
            "[red]No API key found.[/red] Set AGENT_API_KEY (or OPENAI_API_KEY) and try again."
        )
        raise typer.Exit(code=1)

    registry = build_tool_registry(config.agent.tools)
    asyncio.run(_run_agent(_build_agent(settings, registry), prompt))


@app.command()
def tools() -> None:
    """List the tools the agent will use (per the current config)."""
    registry = build_tool_registry(Config.load().agent.tools)
    for spec in registry.tool_schemas():
        fn = spec["function"]
        console.print(f"[bold]{fn['name']}[/bold] -- {fn['description']}")


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
