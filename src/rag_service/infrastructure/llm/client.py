from openai import OpenAI, OpenAIError

from rag_service.core.interfaces import LlmError


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
        except OpenAIError as exc:
            raise LlmError(f"Model {self._model_name} did not answer.") from exc

        return response.output_text
