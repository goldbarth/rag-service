from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from harness.api.dependencies import get_llm_client
from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmConfigurationError,
    LlmError,
    LlmResponseFormatError,
    LlmToolError,
    LlmUnavailableError,
)
from harness.main import app


class RaisingLlmClient:
    def __init__(self, error_factory: Callable[[], LlmError]) -> None:
        self._error_factory = error_factory

    def complete(self, system_prompt: str, user_message: str, config: LlmConfig) -> str:
        raise self._error_factory()


@pytest.fixture
def error_client(
    request: pytest.FixtureRequest,
) -> Iterator[TestClient]:
    """Reach the LLM exception handlers through the real /analyze route.

    /analyze now calls the LLM port, so the tests no longer need throwaway
    routes. The override replaces only the LLM client dependency with a fake
    adapter that raises the requested domain error.

    FastAPI resolves dependency_overrides per request. Therefore this fixture
    must yield so the override stays active while the test sends its request.
    The overrides live on the global app instance and must be cleared afterwards
    to avoid leaking fake dependencies into later tests.
    """
    error_factory = request.param
    app.dependency_overrides[get_llm_client] = lambda: RaisingLlmClient(error_factory)

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "error_client",
    [lambda: LlmConfigurationError("wrong model name")],
    indirect=True,
)
def test_configuration_error_answers_500(error_client: TestClient) -> None:
    response = error_client.post("/analyze", json={"text": "I love Donuts :)"})

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}


@pytest.mark.parametrize(
    "error_client",
    [lambda: LlmUnavailableError("provider did not answer")],
    indirect=True,
)
def test_unavailable_error_answers_502(error_client: TestClient) -> None:
    response = error_client.post("/analyze", json={"text": "I love Donuts :)"})

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream model unavailable"}


@pytest.mark.parametrize(
    "error_client",
    [lambda: LlmResponseFormatError("verdict did not satisfy the schema")],
    indirect=True,
)
def test_response_format_error_answers_502(error_client: TestClient) -> None:
    # The provider answered, it just answered with something unusable. That is
    # still upstream's fault, so it sits with the other 502 rather than with
    # the 500s we raise about ourselves.
    response = error_client.post("/analyze", json={"text": "I love Donuts :)"})

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream model answer was unusable"}


@pytest.mark.parametrize(
    "error_client",
    [lambda: LlmToolError("tool 'search' failed in round 0")],
    indirect=True,
)
def test_tool_error_answers_500(error_client: TestClient) -> None:
    # Our handler broke, not the model, so this must not read as an upstream
    # problem no matter which route it came through.
    response = error_client.post("/analyze", json={"text": "I love Donuts :)"})

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
