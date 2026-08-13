from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_service.core.interfaces import LlmConfigurationError, LlmUnavailableError
from rag_service.main import app


@pytest.fixture
def error_client() -> Iterator[TestClient]:
    """Reach the exception handlers in main.py through throwaway routes.

    No endpoint calls the LLM yet, so the handlers cannot be triggered by any
    real request. The routes are removed again after the test.
    """

    def raise_configuration_error() -> None:
        raise LlmConfigurationError("wrong model name")

    def raise_unavailable_error() -> None:
        raise LlmUnavailableError("provider did not answer")

    original_routes = list(app.router.routes)
    app.add_api_route("/raises-configuration-error", raise_configuration_error)
    app.add_api_route("/raises-unavailable-error", raise_unavailable_error)

    try:
        yield TestClient(app)
    finally:
        app.router.routes[:] = original_routes
        # The cached schema would still list the throwaway routes.
        app.openapi_schema = None


def test_configuration_error_answers_500(error_client: TestClient) -> None:
    response = error_client.get("/raises-configuration-error")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}


def test_unavailable_error_answers_502(error_client: TestClient) -> None:
    response = error_client.get("/raises-unavailable-error")

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream model unavailable"}
