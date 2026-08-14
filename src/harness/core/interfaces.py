from typing import Protocol

from harness.core.config import LlmConfig


class LlmError(Exception):
    """Base llm error. Raised when an exception occurred,
    but no differentiation is required."""


class LlmConfigurationError(LlmError):
    """Raised when the internal configuration is not set properly."""


class LlmUnavailableError(LlmError):
    """Raised when the language model cannot be reached or does not answer."""


class LlmClient(Protocol):
    """Port for text completion. Implemented by adapters in infrastructure."""

    def complete(self, system_prompt: str, user_message: str, config: LlmConfig) -> str:
        """Return the model's answer or raise LlmError."""
        ...
