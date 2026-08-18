from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from harness.core.config import LlmConfig


class LlmError(Exception):
    """Base llm error. Raised when an exception occurred,
    but no differentiation is required."""


class LlmConfigurationError(LlmError):
    """Raised when the internal configuration is not set properly."""


class LlmUnavailableError(LlmError):
    """Raised when the language model cannot be reached or does not answer."""


class LlmResponseFormatError(LlmError):
    """Raised when the model's response cannot be parsed or is invalid."""


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class LlmStructuredCompletion[T: BaseModel]:
    parsed: T
    usage: TokenUsage | None


@dataclass(frozen=True)
class LlmCompletion:
    text: str
    usage: TokenUsage | None = None
    incomplete_reason: Literal["max_output_tokens", "content_filter"] | None = None
    """Set when the provider stopped early. The text is then a partial answer."""


class LlmClient(Protocol):
    """Port for text completion. Implemented by adapters in infrastructure."""

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        """Return the model's answer or raise LlmError."""
        ...

    def complete_structured[T: BaseModel](
        self, system_prompt: str, user_message: str, config: LlmConfig, schema: type[T]
    ) -> LlmStructuredCompletion[T]:
        """Return the model's answer parsed into the schema or raise LlmError.

        Args:
            system_prompt: Instructions that define the model's behavior and role.
            user_message: The input text to be processed by the model.
            config: Configuration settings for the language model.
            schema: Pydantic model type that defines the expected response structure.

        Returns:
            LlmStructuredCompletion containing the parsed response and token usage.

        Raises:
            LlmError: When an error occurs during completion or response parsing.
        """
        ...
