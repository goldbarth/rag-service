from functools import lru_cache

from openai import OpenAI, OpenAIError

from rag_service.core.config import get_settings
from rag_service.core.interfaces import LlmClient, LlmError
from rag_service.infrastructure.llm.client import OpenAiLlmClient


@lru_cache
def get_llm_client() -> LlmClient:
    settings = get_settings()

    try:
        client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.timeout,
        )
    except OpenAIError as exc:
        raise LlmError("Could not create the OpenAI client.") from exc

    return OpenAiLlmClient(client=client, model_name=settings.model_name)
