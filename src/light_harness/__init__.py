"""light_harness -- a small agentic harness that runs LLM agents in response to
triggers (a chat prompt, or an HTTP webhook).

The core pieces:

- :class:`Agent` drives a tool-calling loop: user input -> LLM -> tool calls ->
  tool results -> LLM -> final answer, repeated until the model stops calling
  tools or the step budget is exhausted.
- :class:`LLMClient` (protocol) is the pluggable model boundary.
  :class:`OpenAICompatibleClient` is a default implementation that talks to any
  OpenAI-compatible ``/chat/completions`` endpoint, retrying transient failures.
- :class:`ToolRegistry` holds callable tools the model can invoke, validating
  their arguments and bounding each call's run time.
- :class:`Conversation` keeps the multi-turn message history.
- :class:`Config` loads ``config.yaml`` (providers, agents, personas, interaction,
  webhooks); :class:`SessionLog` records each session as JSONL under the agent's
  persona directory; :func:`serve_webhooks` serves the configured webhooks.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .agent import Agent
from .config import (
    AgentConfig,
    Config,
    Interaction,
    InteractionChat,
    Provider,
    Settings,
    WebhookConfig,
    ensure_persona,
    persona_dir,
    persona_sessions_dir,
    persona_soul_path,
    persona_soul_text,
    personas_home,
    write_persona_soul,
)
from .llm import (
    ChatMessage,
    LLMClient,
    LLMResponse,
    OpenAICompatibleClient,
    ToolCall,
)
from .sessions import SessionLog
from .state import Conversation
from .tools import Tool, ToolRegistry, default_registry
from .webhooks import create_webhook_server, parse_webhook_body, serve_webhooks

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
    "SessionLog",
    "ensure_persona",
    "persona_dir",
    "persona_sessions_dir",
    "persona_soul_path",
    "persona_soul_text",
    "personas_home",
    "write_persona_soul",
    "WebhookConfig",
    "create_webhook_server",
    "parse_webhook_body",
    "serve_webhooks",
]
