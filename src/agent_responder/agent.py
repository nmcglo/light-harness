"""The agentic loop: user input -> LLM -> tool calls -> tool results -> LLM ->
final answer, repeated until the model stops calling tools or the step budget
is exhausted.
"""

from __future__ import annotations

from .llm import ChatMessage, LLMClient
from .sessions import SessionLog
from .state import Conversation
from .tools import ToolRegistry

_STOPPED_MESSAGE = "Stopped: reached the maximum number of steps without a final answer."


class Agent:
    """Drives the tool-calling loop over a persistent :class:`Conversation`."""

    def __init__(
        self,
        client: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int = 8,
        session: SessionLog | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max(1, max_steps)
        self.conversation = Conversation()
        self.session = session

    def reset(self) -> None:
        """Clear the conversation history (start a fresh session)."""
        self.conversation.reset()

    async def run(self, user_input: str) -> str:
        """Run one turn of the loop and return the final assistant answer."""
        self._remember(ChatMessage(role="user", content=user_input))

        for _ in range(self.max_steps):
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                *self.conversation.messages,
            ]
            response = await self.client.chat(
                messages, tools=self.tools.tool_schemas() or None
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
                    self._remember(
                        ChatMessage(role="tool", tool_call_id=call.id, content=result)
                    )
            else:
                final = response.message.content
                self._remember(ChatMessage(role="assistant", content=final))
                return final

        return _STOPPED_MESSAGE

    def _remember(self, message: ChatMessage) -> None:
        """Append ``message`` to the conversation and to the session log, if any."""
        self.conversation.add(message)
        if self.session is not None:
            self.session.record(message)
