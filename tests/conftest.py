from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from harness.api.dependencies import get_llm_client, get_llm_config
from harness.core.config import LlmConfig
from harness.core.interfaces import LlmCompletion
from harness.main import app


class FakeLlmClient:
    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        return LlmCompletion(f"Analyzed: {user_message}")


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Run API tests without touching the real OpenAI client.

    FastAPI resolves dependency_overrides per request, not when TestClient is
    constructed. Therefore this fixture must yield so the overrides stay active
    for the whole test and are cleaned up afterwards.

    The overrides live on the global app instance. Clearing them in finally
    prevents fake dependencies from leaking into later tests.
    """
    app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient()
    app.dependency_overrides[get_llm_config] = lambda: LlmConfig(
        model_name="test-model",
        temperature=0.2,
    )

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
