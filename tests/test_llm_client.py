from dataclasses import dataclass, field
from typing import cast

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from rag_service.core.interfaces import LlmConfigurationError, LlmUnavailableError
from rag_service.infrastructure.llm.client import OpenAiLlmClient


@dataclass
class FakeResponse:
    output_text: str


@dataclass
class FakeResponses:
    result: FakeResponse | None = None
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@dataclass
class FakeOpenAI:
    responses: FakeResponses


def make_client(responses: FakeResponses) -> OpenAiLlmClient:
    return OpenAiLlmClient(
        client=cast(OpenAI, cast(object, FakeOpenAI(responses=responses))),
        model_name="test-model",
    )


def make_status_error(
    error_type: type[APIStatusError], status_code: int
) -> APIStatusError:
    """Build an openai status error without touching the network."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code=status_code, request=request)
    return error_type("upstream said no", response=response, body=None)


def test_complete_returns_response_text() -> None:
    responses = FakeResponses(result=FakeResponse(output_text="Hello from the model."))
    client = make_client(responses)

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
    )

    assert result == "Hello from the model."
    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
        }
    ]


@pytest.mark.parametrize(
    ("error_type", "status_code"),
    [
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (BadRequestError, 400),
        (NotFoundError, 404),
    ],
)
def test_complete_maps_rejected_requests_to_configuration_error(
    error_type: type[APIStatusError], status_code: int
) -> None:
    upstream_error = make_status_error(error_type, status_code)
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmConfigurationError) as exc_info:
        client.complete(system_prompt="You are helpful.", user_message="Say hello.")

    assert exc_info.value.__cause__ is upstream_error


def test_complete_maps_rate_limit_to_unavailable_error() -> None:
    upstream_error = make_status_error(RateLimitError, 429)
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        client.complete(system_prompt="You are helpful.", user_message="Say hello.")

    assert exc_info.value.__cause__ is upstream_error


@pytest.mark.parametrize("error_type", [APIConnectionError, APITimeoutError])
def test_complete_maps_connection_error_to_unavailable_error(
    error_type: type[APIConnectionError],
) -> None:
    upstream_error = error_type(
        request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        client.complete(system_prompt="You are helpful.", user_message="Say hello.")

    assert exc_info.value.__cause__ is upstream_error


def test_complete_maps_unknown_openai_error_to_unavailable_error() -> None:
    upstream_error = OpenAIError("something else went wrong")
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        client.complete(system_prompt="You are helpful.", user_message="Say hello.")

    assert exc_info.value.__cause__ is upstream_error


@pytest.mark.parametrize("output_text", ["", "   \n\t "])
def test_complete_rejects_empty_response(output_text: str) -> None:
    client = make_client(FakeResponses(result=FakeResponse(output_text=output_text)))

    with pytest.raises(LlmUnavailableError):
        client.complete(system_prompt="You are helpful.", user_message="Say hello.")
