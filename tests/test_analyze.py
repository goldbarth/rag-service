from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from harness.api.dependencies import get_llm_client, get_llm_config
from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LLM_INCOMPLETE_REASONS,
    LlmCompletion,
    LlmIncompleteReason,
)
from harness.main import app


def test_analyze_is_valid_text_returns_200(client: TestClient) -> None:
    response = client.post("/analyze", json={"text": "I love Donuts :)"})
    assert response.status_code == 200
    assert response.json()["result"] == "Analyzed: I love Donuts :)"
    assert response.json()["num_chars"] == 26


def test_analyze_is_valid_text_trimmed_returns_200(client: TestClient) -> None:
    response = client.post("/analyze", json={"text": "   I love Donuts :)     "})
    assert response.status_code == 200
    assert response.json()["result"] == "Analyzed: I love Donuts :)"
    assert response.json()["num_chars"] == 26


def test_analyze_rejects_whitespace_only_text_returns_422(client: TestClient) -> None:
    response = client.post("/analyze", json={"text": "  "})
    detail = response.json()["detail"][0]
    assert response.status_code == 422
    assert detail["type"] == "string_too_short"
    assert detail["loc"] == ["body", "text"]


def test_analyze_is_not_valid_missing_text_returns_422(client: TestClient) -> None:
    response = client.post("/analyze", json={})
    detail = response.json()["detail"][0]
    assert response.status_code == 422
    assert detail["type"] == "missing"
    assert detail["loc"] == ["body", "text"]


def test_analyze_rejects_misspelled_text_key_returns_422(client: TestClient) -> None:
    response = client.post("/analyze", json={"txet": "I love Donuts :)"})
    errors = [
        {"type": detail["type"], "loc": detail["loc"]}
        for detail in response.json()["detail"]
    ]

    assert response.status_code == 422
    assert {"type": "missing", "loc": ["body", "text"]} in errors
    assert {"type": "extra_forbidden", "loc": ["body", "txet"]} in errors


def test_analyze_rejects_invalid_input_type_422(client: TestClient) -> None:
    response = client.post("/analyze", json={"text": 123})
    detail = response.json()["detail"][0]
    assert response.status_code == 422
    assert detail["type"] == "string_type"
    assert detail["loc"] == ["body", "text"]


@dataclass
class RecordedLlmCall:
    system_prompt: str
    user_message: str
    config: LlmConfig


class RecordingLlmClient:
    def __init__(self, completion: LlmCompletion | None = None) -> None:
        self.calls: list[RecordedLlmCall] = []
        self.completion = completion or LlmCompletion("local fake answer")

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        self.calls.append(
            RecordedLlmCall(
                system_prompt=system_prompt,
                user_message=user_message,
                config=config,
            )
        )
        return self.completion


def test_analyze_passes_request_and_config_to_llm() -> None:
    llm = RecordingLlmClient()
    llm_config = LlmConfig(model_name="local-test-model", temperature=0.7)

    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_llm_config] = lambda: llm_config

    try:
        response = TestClient(app).post(
            "/analyze",
            json={"text": "   I love Donuts :)     "},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "result": "local fake answer",
        "num_chars": len("local fake answer"),
        "incomplete_reason": None,
    }

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.system_prompt
    assert call.user_message == "I love Donuts :)"
    assert call.config == llm_config


@pytest.mark.parametrize("reason", LLM_INCOMPLETE_REASONS)
def test_analyze_reports_that_the_provider_cut_the_answer_short(
    reason: LlmIncompleteReason,
) -> None:
    # A truncated answer is still a 200, so the body is the only place that can
    # tell the caller the text is partial.
    llm = RecordingLlmClient(LlmCompletion("half an ans", None, reason))

    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_llm_config] = lambda: LlmConfig(
        model_name="local-test-model"
    )

    try:
        response = TestClient(app).post("/analyze", json={"text": "tell me more"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["result"] == "half an ans"
    assert response.json()["incomplete_reason"] == reason
