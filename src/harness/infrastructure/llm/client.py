import logging

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    omit,
)

from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmCompletion,
    LlmConfigurationError,
    LlmUnavailableError,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class OpenAiLlmClient:
    """LlmClient adapter for the OpenAI Responses API."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        try:
            response = self._client.responses.create(
                model=config.model_name,
                instructions=system_prompt,
                input=user_message,
                temperature=config.temperature
                if config.temperature is not None
                else omit,
                max_output_tokens=config.max_output_tokens
                if config.max_output_tokens is not None
                else omit,
            )
        except (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
        ) as exc:
            raise LlmConfigurationError(
                f"Request for model {config.model_name} was rejected."
            ) from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} is currently unavailable."
            ) from exc
        except OpenAIError as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} did not answer."
            ) from exc

        details = response.incomplete_details
        incomplete_reason = details.reason if details is not None else None

        text = response.output_text
        if incomplete_reason is not None:
            # The provider stopped early, so an empty text is explained and not
            # a sign that the model is unavailable.
            logger.warning(
                "LLM response is incomplete",
                extra={
                    "model_name": config.model_name,
                    "incomplete_reason": incomplete_reason,
                },
            )
        elif not text.strip():
            raise LlmUnavailableError(
                f"The response for the {config.model_name} model contains no content."
            )

        usage = response.usage
        if usage is None:
            token_usage = None
            logger.warning(
                "LLM call completed without usage data",
                extra={"model_name": config.model_name},
            )
        else:
            token_usage = TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cached_tokens=usage.input_tokens_details.cached_tokens,
                cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
                reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
            )
            logger.info(
                "LLM call completed",
                extra={
                    "model_name": config.model_name,
                    "input_tokens": token_usage.input_tokens,
                    "output_tokens": token_usage.output_tokens,
                    "total_tokens": token_usage.total_tokens,
                    "cached_tokens": token_usage.cached_tokens,
                    "cache_write_tokens": token_usage.cache_write_tokens,
                    "reasoning_tokens": token_usage.reasoning_tokens,
                },
            )

        return LlmCompletion(text, token_usage, incomplete_reason)
