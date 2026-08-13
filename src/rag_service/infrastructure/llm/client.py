from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from rag_service.core.interfaces import LlmConfigurationError, LlmUnavailableError


class OpenAiLlmClient:
    """LlmClient adapter for the OpenAI Responses API."""

    def __init__(self, client: OpenAI, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def complete(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self._client.responses.create(
                model=self._model_name,
                instructions=system_prompt,
                input=user_message,
            )
        except (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
        ) as exc:
            raise LlmConfigurationError(
                f"Request for model {self._model_name} was rejected."
            ) from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LlmUnavailableError(
                f"Model {self._model_name} is currently unavailable."
            ) from exc
        except OpenAIError as exc:
            raise LlmUnavailableError(
                f"Model {self._model_name} did not answer."
            ) from exc

        return response.output_text
