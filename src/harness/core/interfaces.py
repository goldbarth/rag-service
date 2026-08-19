from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, get_args

from pydantic import BaseModel

from harness.core.config import LlmConfig

LlmIncompleteReason = Literal["max_output_tokens", "content_filter"]
LLM_INCOMPLETE_REASONS = cast(
    tuple[LlmIncompleteReason, ...],
    get_args(LlmIncompleteReason),
)

LlmToolStopReason = Literal["completed", "incomplete_details", "max_rounds"]
LLM_TOOL_STOP_REASONS = cast(
    tuple[LlmToolStopReason, ...],
    get_args(LlmToolStopReason),
)


class LlmError(Exception):
    """Base llm error. Raised when an exception occurred,
    but no differentiation is required."""


class LlmConfigurationError(LlmError):
    """Raised when the internal configuration is not set properly."""


class LlmUnavailableError(LlmError):
    """Raised when the language model cannot be reached or does not answer."""


class LlmResponseFormatError(LlmError):
    """Raised when the model's response cannot be parsed or is invalid."""


class LlmToolError(LlmError):
    """Raised when a tool handler fails during a tool-augmented run.

    The fault is ours, not the provider's: the model asked for a tool we
    offered and our handler could not deliver it. It stays inside the LlmError
    family so one except still covers a whole run, and it names the tool so the
    log points at the handler rather than at the model.
    """


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
    incomplete_reason: LlmIncompleteReason | None = None
    """Set when the provider stopped early. The text is then a partial answer."""


class TextCompleter(Protocol):
    """Port for plain text completion. Implemented by adapters in infrastructure."""

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        """Return the model's answer or raise LlmError."""
        ...


class StructuredCompleter(Protocol):
    """Port for schema-bound completion. What the judge needs, and no more."""

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


class ToolCompleter(Protocol):
    """Port for tool-augmented completion. The model may call the given tools."""

    def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        max_rounds: int = 5,
    ) -> LlmToolCompletion:
        """Run the model until it answers without calling a tool.

        Raises:
            LlmError: On transport failure, an unknown tool name, an empty
                answer, or a tool handler that raised.
            ValueError: If max_rounds is below 1, or if two tools share a name.
                Both are mistakes in the calling code, checked before the first
                request. The call therefore always reaches the provider at
                least once, so "max_rounds" never means "no attempt".
        """
        ...


class LlmClient(TextCompleter, StructuredCompleter, ToolCompleter, Protocol):
    """The full adapter surface, for wiring only.

    Consumers depend on the single role they use, so a new method on one role
    cannot break a test double that never touches it. Protocol has to stay in
    the bases here: without it this becomes a plain ABC and adapters would have
    to inherit from it to satisfy it.
    """


@dataclass(frozen=True)
class ToolSpec[T: BaseModel]:
    name: str
    params: type[T]
    handler: Callable[[T], str]
    description: str | None = None


@dataclass(frozen=True)
class ToolRound:
    tool_names: tuple[str, ...]
    usage: TokenUsage | None


@dataclass(frozen=True)
class LlmToolCompletion:
    text: str
    rounds: tuple[ToolRound, ...]
    stop_reason: LlmToolStopReason


@dataclass(frozen=True)
class SectionHit:
    doc_id: str
    section: str
    text: str


class SectionSearch(Protocol):
    def find(self, query: str, top_k: int) -> Sequence[SectionHit]:
        """Search for document sections matching the query.

        Args:
            query: Search terms to match against document sections.
            top_k: Maximum number of results to return.

        Returns:
            Sequence of matching section hits, ranked by relevance.
        """
        ...
