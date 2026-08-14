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
from harness.core.interfaces import LlmConfigurationError, LlmUnavailableError


class OpenAiLlmClient:
    """LlmClient adapter for the OpenAI Responses API."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def complete(self, system_prompt: str, user_message: str, config: LlmConfig) -> str:
        try:
            response = self._client.responses.create(
                model=config.model_name,
                instructions=system_prompt,
                input=user_message,
                temperature=config.temperature
                if config.temperature is not None
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

        text = response.output_text

        if not text.strip():
            raise LlmUnavailableError(
                f"The response for the {config.model_name} model contains no content."
            )

        return text
