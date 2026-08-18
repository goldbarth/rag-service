from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from harness.core.config import LlmConfig
from harness.core.evaluation import JudgeVerdict, judge_answer
from harness.core.interfaces import LlmCompletion, LlmStructuredCompletion, TokenUsage


def test_judge_verdict_allows_correct_without_reasoning() -> None:
    verdict = JudgeVerdict(verdict="correct", reasoning="")

    assert verdict.verdict == "correct"
    assert verdict.reasoning == ""


def test_judge_verdict_rejects_incorrect_without_reasoning() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(verdict="incorrect", reasoning="")


def test_judge_verdict_rejects_incorrect_with_whitespace_reasoning() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(verdict="incorrect", reasoning="   ")


def test_judge_verdict_allows_incorrect_with_reasoning() -> None:
    verdict = JudgeVerdict(verdict="incorrect", reasoning="Missing required facts.")

    assert verdict.verdict == "incorrect"
    assert verdict.reasoning == "Missing required facts."


def test_judge_verdict_rejects_unclear_without_reasoning() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(verdict="unclear", reasoning="")


def test_judge_verdict_rejects_unclear_with_whitespace_reasoning() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(verdict="unclear", reasoning="   ")


def test_judge_verdict_allows_unclear_with_reasoning() -> None:
    """Guards the other direction: unclear must stay reachable, not be
    rejected wholesale."""
    verdict = JudgeVerdict(
        verdict="unclear", reasoning="The expected answer covers a different reading."
    )

    assert verdict.verdict == "unclear"
    assert verdict.reasoning == "The expected answer covers a different reading."


@dataclass
class RecordedStructuredCall:
    system_prompt: str
    user_message: str
    config: LlmConfig
    schema: type[Any]


class RecordingJudgeLlmClient:
    def __init__(self, verdict: JudgeVerdict, usage: TokenUsage | None = None) -> None:
        self.verdict = verdict
        self.usage = usage
        self.calls: list[RecordedStructuredCall] = []

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        raise NotImplementedError(
            "This test double only supports structured completion."
        )

    def complete_structured[T: BaseModel](
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        schema: type[T],
    ) -> LlmStructuredCompletion[T]:
        self.calls.append(
            RecordedStructuredCall(
                system_prompt=system_prompt,
                user_message=user_message,
                config=config,
                schema=schema,
            )
        )
        return cast(
            LlmStructuredCompletion[T],
            LlmStructuredCompletion(parsed=self.verdict, usage=self.usage),
        )


def test_judge_answer_passes_loose_parameters_to_structured_llm() -> None:
    llm_config = LlmConfig(model_name="judge-test-model", temperature=0.0)
    llm = RecordingJudgeLlmClient(
        JudgeVerdict(
            verdict="incorrect", reasoning="The answer contradicts the gold answer."
        )
    )

    result = judge_answer(
        question="What is the capital of France?",
        expected_answer="Paris is the capital of France.",
        actual_answer="Lyon is the capital of France.",
        llm=llm,
        llm_config=llm_config,
    )

    assert result.verdict == JudgeVerdict(
        verdict="incorrect",
        reasoning="The answer contradicts the gold answer.",
    )

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.system_prompt
    assert "Question:\nWhat is the capital of France?" in call.user_message
    assert "Expected answer:\nParis is the capital of France." in call.user_message
    assert "Actual answer:\nLyon is the capital of France." in call.user_message
    assert call.config == llm_config
    assert call.schema is JudgeVerdict


def test_judge_answer_reports_the_token_usage_of_the_call() -> None:
    """A run's cost is the sum over judge calls, so usage must not be dropped."""
    usage = TokenUsage(
        input_tokens=11,
        output_tokens=22,
        total_tokens=33,
        cached_tokens=44,
        cache_write_tokens=55,
        reasoning_tokens=66,
    )
    llm = RecordingJudgeLlmClient(
        JudgeVerdict(verdict="correct", reasoning=""), usage=usage
    )

    result = judge_answer(
        question="What is the capital of France?",
        expected_answer="Paris is the capital of France.",
        actual_answer="Paris.",
        llm=llm,
        llm_config=LlmConfig(model_name="judge-test-model"),
    )

    assert result.usage is usage


def test_judge_answer_reports_no_usage_when_the_provider_omits_it() -> None:
    llm = RecordingJudgeLlmClient(JudgeVerdict(verdict="correct", reasoning=""))

    result = judge_answer(
        question="What is the capital of France?",
        expected_answer="Paris is the capital of France.",
        actual_answer="Paris.",
        llm=llm,
        llm_config=LlmConfig(model_name="judge-test-model"),
    )

    assert result.usage is None
