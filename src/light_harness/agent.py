"""The agentic loop: user input -> LLM -> tool calls -> tool results -> LLM ->
final answer, repeated until the model stops calling tools or the step budget
is exhausted.

Every message is passed through the :class:`Redactor` before it enters the
conversation (and so the session log and the next request): a secret value a
tool printed, or the user pasted, is replaced with ``[REDACTED:name]``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .llm import ChatMessage, LLMClient, ToolCall
from .secrets import Redactor, SecretContext
from .sessions import SessionLog
from .state import Conversation
from .tools import ToolRegistry

_STOPPED_MESSAGE = "Stopped: reached the maximum number of steps without a final answer."
_INTERRUPTED_MARK = "\n[interrupted by user]"

ToolCallback = Callable[[str, dict[str, Any], str], None]


class Agent:
    """Drives the tool-calling loop over a persistent :class:`Conversation`."""

    def __init__(
        self,
        client: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int = 8,
        session: SessionLog | None = None,
        redactor: Redactor | None = None,
        secret_context: SecretContext | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max(1, max_steps)
        self.conversation = Conversation()
        self.session = session
        self.redactor = redactor if redactor is not None else Redactor.empty()
        # Shared with the tools' ToolSecrets: tells the secret policy whether
        # untrusted content has entered the conversation.
        self.secret_context = secret_context

    def reset(self) -> None:
        """Clear the conversation history (start a fresh session)."""
        self.conversation.reset()
        if self.secret_context is not None and self.secret_context.interaction == "chat":
            self.secret_context.untrusted = False

    def secret_in(self, text: str) -> str | None:
        """The name of a registered secret found in ``text`` (e.g. a pasted
        prompt), or ``None``. The text is redacted when it is remembered anyway;
        this lets callers warn the user."""
        return self.redactor.find(text)

    async def run(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None = None,
        on_tool: ToolCallback | None = None,
    ) -> str:
        """Run one turn of the loop and return the final assistant answer.

        ``on_token`` receives streamed text deltas; ``on_tool(name, arguments,
        result)`` is called after each tool call completes.

        If the turn is cancelled, the conversation is rolled back to a state the
        model can continue from: the user message, then an assistant message with
        whatever text had streamed so far plus an interruption marker (no dangling
        tool calls). The cancellation is then re-raised."""
        start = len(self.conversation.messages)
        streamed: list[str] = []

        def tap(text: str) -> None:
            streamed.append(text)
            if on_token is not None:
                on_token(text)

        try:
            return await self._loop(
                user_input, tap if on_token is not None else None, on_tool
            )
        except asyncio.CancelledError:
            del self.conversation.messages[start + 1 :]
            self._remember(
                ChatMessage(role="assistant", content="".join(streamed) + _INTERRUPTED_MARK)
            )
            raise

    async def _loop(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None,
        on_tool: ToolCallback | None,
    ) -> str:
        self._remember(ChatMessage(role="user", content=user_input))

        for _ in range(self.max_steps):
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                *self.conversation.messages,
            ]
            extra = {"on_token": on_token} if on_token is not None else {}
            response = await self.client.chat(
                messages, tools=self.tools.tool_schemas() or None, **extra
            )

            if response.tool_calls:
                self._remember(
                    ChatMessage(
                        role="assistant",
                        content=response.message.content,
                        tool_calls=response.tool_calls,
                    )
                )
                for call in response.tool_calls:
                    if call.error is not None:
                        result = f"Error: invalid arguments for '{call.name}': {call.error}."
                    else:
                        result = await self.tools.run(call.name, call.arguments)
                        self._note_untrusted(call.name, result)
                    result = self.redactor.redact(result)
                    self._remember(
                        ChatMessage(role="tool", tool_call_id=call.id, content=result)
                    )
                    if on_tool is not None:
                        on_tool(call.name, self.redactor.redact_obj(call.arguments), result)
            else:
                final = self.redactor.redact(response.message.content)
                self._remember(ChatMessage(role="assistant", content=final))
                return final

        return _STOPPED_MESSAGE

    def _note_untrusted(self, name: str, result: str) -> None:
        """Mark the conversation untrusted once a tool returned third-party text."""
        tool = self.tools.get(name)
        if (
            self.secret_context is not None
            and getattr(tool, "untrusted_output", False)
            and not result.startswith("Error:")
        ):
            self.secret_context.untrusted = True

    def _remember(self, message: ChatMessage) -> None:
        """Append ``message`` (redacted) to the conversation and to the session
        log, if any."""
        if self.redactor:
            message = message.model_copy(
                update={
                    "content": self.redactor.redact(message.content),
                    "tool_calls": [
                        ToolCall(
                            id=tc.id, name=tc.name, error=tc.error,
                            arguments=self.redactor.redact_obj(tc.arguments),
                        )  # fmt: skip
                        for tc in message.tool_calls
                    ]
                    if message.tool_calls
                    else message.tool_calls,
                }
            )
        self.conversation.add(message)
        if self.session is not None:
            self.session.record(message)
