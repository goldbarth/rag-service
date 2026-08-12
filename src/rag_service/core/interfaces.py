from typing import Protocol


class LlmError(Exception):
    """Raised when the language model cannot be reached or does not answer."""


class LlmClient(Protocol):
    """Port for text completion. Implemented by adapters in infrastructure."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Return the model's answer, or raise LlmError."""
        ...
