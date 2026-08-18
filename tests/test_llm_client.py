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
    omit,
)

from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmCompletion,
    LlmConfigurationError,
    LlmUnavailableError,
    TokenUsage,
)
from harness.infrastructure.llm.client import OpenAiLlmClient


@dataclass
class FakeInputTokensDetails:
    cached_tokens: int
    cache_write_tokens: int


@dataclass
class FakeOutputTokensDetails:
    reasoning_tokens: int


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: FakeInputTokensDetails
    output_tokens_details: FakeOutputTokensDetails


@dataclass
class FakeIncompleteDetails:
    reason: str | None


@dataclass
class FakeResponse:
    output_text: str
    usage: FakeUsage | None = None
    incomplete_details: FakeIncompleteDetails | None = None


def make_usage() -> FakeUsage:
    """Distinct numbers everywhere, so a swapped field cannot pass."""
    return FakeUsage(
        input_tokens=11,
        output_tokens=22,
        total_tokens=33,
        input_tokens_details=FakeInputTokensDetails(
            cached_tokens=44, cache_write_tokens=55
        ),
        output_tokens_details=FakeOutputTokensDetails(reasoning_tokens=66),
    )


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
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )

    assert result == LlmCompletion(text="Hello from the model.")
    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "temperature": 0.2,
            "max_output_tokens": omit,
        }
    ]


def test_complete_uses_omit_for_temperature_when_not_configured() -> None:
    responses = FakeResponses(result=FakeResponse(output_text="Hello from the model."))
    client = make_client(responses)

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="reasoning-model", temperature=None),
    )

    assert result == LlmCompletion(text="Hello from the model.")
    assert responses.calls == [
        {
            "model": "reasoning-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "temperature": omit,
            "max_output_tokens": omit,
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
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )

    assert exc_info.value.__cause__ is upstream_error


def test_complete_maps_rate_limit_to_unavailable_error() -> None:
    upstream_error = make_status_error(RateLimitError, 429)
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )

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
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )

    assert exc_info.value.__cause__ is upstream_error


def test_complete_maps_unknown_openai_error_to_unavailable_error() -> None:
    upstream_error = OpenAIError("something else went wrong")
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )

    assert exc_info.value.__cause__ is upstream_error


@pytest.mark.parametrize("output_text", ["", "   \n\t "])
def test_complete_rejects_empty_response(output_text: str) -> None:
    client = make_client(FakeResponses(result=FakeResponse(output_text=output_text)))

    with pytest.raises(LlmUnavailableError):
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )


def test_complete_passes_full_usage_through() -> None:
    client = make_client(
        FakeResponses(
            result=FakeResponse(output_text="Hello.", usage=make_usage()),
        )
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )

    assert result.usage == TokenUsage(
        input_tokens=11,
        output_tokens=22,
        total_tokens=33,
        cached_tokens=44,
        cache_write_tokens=55,
        reasoning_tokens=66,
    )


def test_complete_returns_no_usage_when_the_provider_omits_it() -> None:
    client = make_client(
        FakeResponses(result=FakeResponse(output_text="Hello.", usage=None))
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )

    assert result.usage is None


def test_complete_sends_max_output_tokens_when_configured() -> None:
    responses = FakeResponses(result=FakeResponse(output_text="Hello."))
    client = make_client(responses)

    client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(
            model_name="test-model", temperature=0.2, max_output_tokens=64
        ),
    )

    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "temperature": 0.2,
            "max_output_tokens": 64,
        }
    ]


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
def test_complete_reports_a_truncated_answer_instead_of_hiding_it(reason: str) -> None:
    client = make_client(
        FakeResponses(
            result=FakeResponse(
                output_text="Half an ans",
                usage=make_usage(),
                incomplete_details=FakeIncompleteDetails(reason=reason),
            )
        )
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", max_output_tokens=8),
    )

    assert result.text == "Half an ans"
    assert result.incomplete_reason == reason


def test_complete_does_not_blame_the_provider_for_an_empty_truncated_answer() -> None:
    """The budget went into reasoning tokens, so there is no text but also no outage."""
    client = make_client(
        FakeResponses(
            result=FakeResponse(
                output_text="",
                usage=make_usage(),
                incomplete_details=FakeIncompleteDetails(reason="max_output_tokens"),
            )
        )
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", max_output_tokens=8),
    )

    assert result.text == ""
    assert result.incomplete_reason == "max_output_tokens"


def test_complete_leaves_incomplete_reason_unset_for_a_full_answer() -> None:
    client = make_client(
        FakeResponses(result=FakeResponse(output_text="Hello.", usage=make_usage()))
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )

    assert result.incomplete_reason is None
