from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from harness.core.config import LlmConfig
from harness.core.interfaces import LlmClient, TokenUsage
from harness.core.prompts import JUDGE_SYSTEM_PROMPT


class JudgeVerdict(BaseModel):
    """Structured verdict the judge returns for one gold question.

    Three states, not two. "unclear" says the comparison could not be decided,
    which is a defect in the test case, not a property of the answer. It is
    therefore not a measurement: the runner must report it separately instead
    of folding it into hits or misses, because counting it either way moves the
    score for a reason that has nothing to do with retrieval.
    """

    # Field descriptions do steer the model, confirmed in
    # scripts/schema_constraint_probe.py, but the judge criteria stay in
    # JUDGE_SYSTEM_PROMPT alone. Prompt version is a configuration dimension of
    # the harness, so a criterion living here would not be covered by it and a
    # diff between two prompt versions could not explain a flipped verdict.
    #
    # No pydantic defaults here either: strict makes every field required, the
    # default never fires, and the model supplies something. Use `str | None` to
    # express "may be absent".
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["correct", "incorrect", "unclear"]
    reasoning: str

    @model_validator(mode="after")
    def require_reasoning_for_a_negative_verdict(self) -> Self:
        if self.verdict in ("incorrect", "unclear") and not self.reasoning.strip():
            raise ValueError(
                "Reasoning must be provided for incorrect and unclear verdicts"
            )
        return self


@dataclass(frozen=True)
class JudgeResult:
    verdict: JudgeVerdict
    usage: TokenUsage | None
    """None when the provider reported no usage. Judge calls are billed per gold
    question, so a run's cost is wrong without them."""


def judge_answer(
    question: str,
    expected_answer: str,
    actual_answer: str,
    llm: LlmClient,
    llm_config: LlmConfig,
) -> JudgeResult:
    """Judge one answer against the expected answer.

    Open: judge determinism. Reasoning models reject temperature, so leaving it
    None (the adapter sends omit) is the only way to reach them at all, and
    there is no other knob. Verdicts can therefore drift between runs, and a
    diff cannot yet separate a retrieval regression from judge noise. Measuring
    the flip rate over repeated runs comes before trying to suppress it.
    """
    user_message = f"""Question:
{question}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}
"""

    result = llm.complete_structured(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_message=user_message,
        config=llm_config,
        schema=JudgeVerdict,
    )

    return JudgeResult(verdict=result.parsed, usage=result.usage)
