from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harness.api.dependencies import get_llm_client, get_llm_config, get_tools
from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LLM_TOOL_STOP_REASONS,
    LlmToolCompletion,
    LlmToolStopReason,
    SectionHit,
    ToolCompleter,
    ToolRound,
    ToolSpec,
)
from harness.core.tools import build_section_search_tool
from harness.main import app


@dataclass
class RecordedToolCall:
    system_prompt: str
    user_message: str
    config: LlmConfig
    tools: tuple[str, ...]


class RecordingToolCompleter:
    """Only implements ToolCompleter. That it suffices is the point of the role."""

    def __init__(self, completion: LlmToolCompletion) -> None:
        self.completion = completion
        self.calls: list[RecordedToolCall] = []

    def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        max_rounds: int = 5,
    ) -> LlmToolCompletion:
        self.calls.append(
            RecordedToolCall(
                system_prompt=system_prompt,
                user_message=user_message,
                config=config,
                tools=tuple(tool.name for tool in tools),
            )
        )
        return self.completion


def test_the_double_satisfies_the_tool_role() -> None:
    # Static check, not a runtime one: if ToolCompleter grows a method, this
    # assignment stops type-checking and the double gets fixed on purpose
    # rather than by a failing request test.
    completer: ToolCompleter = RecordingToolCompleter(
        LlmToolCompletion("", (), "completed")
    )

    assert completer is not None


class FakeSearch:
    def find(self, query: str, top_k: int) -> list[SectionHit]:
        return [SectionHit("doc", "section", "body")]


RagClientFactory = Callable[[RecordingToolCompleter, LlmConfig], TestClient]


@pytest.fixture
def rag_client() -> Iterator[RagClientFactory]:
    """Build a TestClient with every /rag/analyze dependency overridden.

    Every test in this file must go through here, including the ones that
    expect a 422. FastAPI resolves the dependencies for a request even when the
    body fails validation, so an un-overridden test still runs the real
    get_llm_client, reads Settings and needs an API key. That passes locally
    where a .env sits in the working directory and fails in CI with a
    ValidationError that has nothing to do with the status code under test.
    """

    def build(llm: RecordingToolCompleter, llm_config: LlmConfig) -> TestClient:
        tools: list[ToolSpec[Any]] = [build_section_search_tool(FakeSearch())]
        app.dependency_overrides[get_llm_client] = lambda: llm
        app.dependency_overrides[get_llm_config] = lambda: llm_config
        app.dependency_overrides[get_tools] = lambda: tools
        return TestClient(app)

    try:
        yield build
    finally:
        app.dependency_overrides.clear()


def _completer(
    text: str, stop_reason: LlmToolStopReason = "completed"
) -> RecordingToolCompleter:
    return RecordingToolCompleter(LlmToolCompletion(text, (), stop_reason))


def test_rag_analyze_returns_the_model_answer(rag_client: RagClientFactory) -> None:
    llm = _completer("tool backed answer")
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "  who? "})

    assert response.status_code == 200
    assert response.json() == {
        "result": "tool backed answer",
        "num_chars": len("tool backed answer"),
        "stop_reason": "completed",
    }


def test_rag_analyze_passes_request_config_and_tools_to_the_llm(
    rag_client: RagClientFactory,
) -> None:
    llm_config = LlmConfig(model_name="rag-test-model", temperature=0.3)
    llm = RecordingToolCompleter(
        LlmToolCompletion(
            "answer", (ToolRound(("search_sections",), None),), "completed"
        )
    )
    client = rag_client(llm, llm_config)

    response = client.post("/rag/analyze", json={"text": "  who? "})

    assert response.status_code == 200
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.system_prompt
    assert call.user_message == "who?"
    assert call.config == llm_config
    assert call.tools == ("search_sections",)


def test_rag_analyze_rejects_whitespace_only_text_returns_422(
    rag_client: RagClientFactory,
) -> None:
    llm = _completer("never reached")
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "   "})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "text"]
    assert llm.calls == []


def test_rag_analyze_returns_the_partial_text_when_the_run_hit_max_rounds(
    rag_client: RagClientFactory,
) -> None:
    # max_rounds is not an error: the caller gets what the model produced, and
    # stop_reason is the only thing that says the answer is not a full one.
    llm = _completer("partial", stop_reason="max_rounds")
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "who?"})

    assert response.status_code == 200
    assert response.json()["result"] == "partial"
    assert response.json()["stop_reason"] == "max_rounds"


@pytest.mark.parametrize("stop_reason", LLM_TOOL_STOP_REASONS)
def test_rag_analyze_reports_every_stop_reason(
    stop_reason: LlmToolStopReason, rag_client: RagClientFactory
) -> None:
    # Parametrized over the alias itself: a new stop reason cannot be added to
    # the port without this test demanding that the router passes it on.
    llm = _completer("answer", stop_reason=stop_reason)
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "who?"})

    assert response.status_code == 200
    assert response.json()["stop_reason"] == stop_reason
