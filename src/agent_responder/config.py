"""Runtime configuration for the agent.

Settings are read from ``AGENT_*`` environment variables (falling back to
``OPENAI_API_KEY`` for the key). The values mirror an OpenAI-compatible API so
the default client works out of the box, but they are all overridable to point
at any compatible endpoint.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Connection + model settings for the LLM client."""

    model: str = "qwen3.8:27b"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    timeout: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls, prefix: str = "AGENT") -> Settings:
        """Build settings from the environment.

        Env vars: ``{prefix}_MODEL``, ``{prefix}_BASE_URL``, ``{prefix}_API_KEY``
        (or ``OPENAI_API_KEY``), ``{prefix}_TIMEOUT``. Any unset variable falls
        back to the corresponding field default.
        """
        values: dict[str, Any] = {}
        for attr, var in (
            ("model", f"{prefix}_MODEL"),
            ("base_url", f"{prefix}_BASE_URL"),
        ):
            if value := os.environ.get(var):
                values[attr] = value
        api_key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            values["api_key"] = api_key
        if timeout := os.environ.get(f"{prefix}_TIMEOUT"):
            values["timeout"] = float(timeout)
        return cls(**values)
