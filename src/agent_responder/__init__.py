"""agent_responder -- a uv-friendly Python scaffold for an agentic harness.

The core pieces:

- :class:`Agent` drives a tool-calling loop: user input -> LLM -> tool calls ->
  tool results -> LLM -> final answer, repeated until the model stops calling
  tools or the step budget is exhausted.
- :class:`LLMClient` (protocol) is the pluggable model boundary.
  :class:`OpenAICompatibleClient` is a default implementation that talks to any
  OpenAI-compatible ``/chat/completions`` endpoint.
- :class:`ToolRegistry` holds callable tools the model can invoke.
- :class:`Conversation` keeps the multi-turn message history.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .agent import Agent
from .config import AgentConfig, Config, Interaction, InteractionChat, Provider, Settings
from .llm import (
    ChatMessage,
    LLMClient,
    LLMResponse,
    OpenAICompatibleClient,
    ToolCall,
)
from .state import Conversation
from .tools import Tool, ToolRegistry, default_registry

__all__ = [
    "__version__",
    "Agent",
    "AgentConfig",
    "Config",
    "Interaction",
    "InteractionChat",
    "Conversation",
    "ChatMessage",
    "LLMClient",
    "LLMResponse",
    "OpenAICompatibleClient",
    "Settings",
    "Provider",
    "Tool",
    "ToolRegistry",
    "ToolCall",
    "default_registry",
]
