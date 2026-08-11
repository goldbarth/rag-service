from functools import lru_cache

from openai import OpenAI, OpenAIError

from rag_service.llm import LlmConfig, LlmError, complete
from rag_service.settings import get_settings


@lru_cache
def get_config() -> LlmConfig:
    settings = get_settings()
    return LlmConfig(model_name=settings.model_name)
    

@lru_cache
def get_client() -> OpenAI:
    settings = get_settings()

    try:
        return OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.timeout,
        )
    except OpenAIError as exc:
        raise LlmError("Could not create the OpenAI client.") from exc


if __name__ == "__main__":
    settings = get_settings()
    answer = complete(
        client=get_client(),
        config=get_config(),
        system_prompt="You're a helpful assistant. Keep your answers brief.",
        user_message="Tell me in one sentence what a retrieval-regression harness is.",
    )
    print(answer)
