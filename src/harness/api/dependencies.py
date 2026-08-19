from functools import lru_cache
from typing import Any

from openai import OpenAI, OpenAIError

from harness.core.config import DEFAULT_MODEL_NAME, LlmConfig, get_settings
from harness.core.interfaces import (
    LlmClient,
    LlmConfigurationError,
    SectionSearch,
    ToolSpec,
)
from harness.core.tools import build_section_search_tool
from harness.infrastructure.llm.client import OpenAiLlmClient
from harness.infrastructure.retrieval.stub import build_stub_corpus


@lru_cache
def get_llm_client() -> LlmClient:
    settings = get_settings()

    try:
        client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.timeout,
        )
    except OpenAIError as exc:
        raise LlmConfigurationError("Could not create the OpenAI client.") from exc

    return OpenAiLlmClient(client=client)


def get_llm_config() -> LlmConfig:
    return LlmConfig(model_name=DEFAULT_MODEL_NAME)


@lru_cache
def get_section_search() -> SectionSearch:
    return build_stub_corpus()


def get_tools() -> list[ToolSpec[Any]]:
    return [build_section_search_tool(get_section_search())]
