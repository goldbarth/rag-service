"""Probe which pydantic Field constraints the provider's strict mode accepts.

Manual experiment against the real API. Needs a valid .env and spends tokens:
one call per accepted schema. Rejected schemas fail before generation.

Two separate questions per probe, and only the first is answered by the
provider's error:
  accepted  - the schema passed strict validation at all
  honored   - the constraint actually shaped the output

A constraint that is accepted but ignored is the dangerous case: the schema
looks like a guarantee and is not one. That is why every probe asks the model
for a value that violates its own constraint.

Usage:
    uv run python scripts/schema_constraint_probe.py [model_name] [probe_name ...]

Results are model and API specific, not pydantic specific. Every run recorded
here therefore carries model, date and SDK version, so the table stays evidence
rather than a claim.

Run 2026-08-18, model gpt-5.6-luna, openai SDK 2.53.0, 1522 tokens
------------------------------------------------------------------
Rejected by strict schema validation:
  unique_items    set[str]         -> "'uniqueItems' is not permitted"
  fixed_tuple     tuple[str, int]  -> "array schema missing items"
  free_form_dict  dict[str, str]   -> "'required' ... Extra required key 'value'"

Accepted and enforced, every probe asked for a violating value and got a
compliant one instead:
  string_length        max_length=5      "long sentence" -> 'Paris'
  string_pattern       ^[A-Z]{3}$        "paris"         -> 'IIS'
  int_range            ge=1 le=5         900             -> 1
  int_exclusive_range  gt=1 lt=5         5               -> 4
  int_multiple_of      multiple_of=10    7               -> 70
  float_range          ge=0.0 le=1.0     42.5            -> 1.0
  list_length          min=2 max=3       ten capitals    -> 3 entries
  literal_choice       Literal[...]      "maybe"         -> 'incorrect'
  field_description    description=...   "Lyon"          -> 'Paris'
  optional_field       str | None        "unknown"       -> None
  nested_model, union_of_models, datetime_format, extra_allowed: all accepted

Accepted but NOT what pydantic means locally:
  field_default   value: str = "unset", asked to omit the field -> {'value': ''}
                  strict makes every field required, so the default never fires
                  and the model supplies something. Use `str | None` to express
                  "may be absent", never a default.

The central finding: an enforced constraint makes the model fabricate a
compliant value rather than fail. 'IIS' is not an answer, it is schema shaped
noise, and it arrives as a valid response with no error. strict guarantees
structure, never meaning. A Literal without an escape value therefore has no
abstain path: the model guesses inside the allowed set.

Validators (field_validator, model_validator) never appear in the schema at
all, so they remain the only semantic gate on this side.

extra="forbid" is redundant: extra_allowed was accepted without it, the SDK
sets additionalProperties: false itself.

Consequence for the product code: set, tuple and open-ended dict cannot be used
in a text_format model. Relevant for the gold question model, where a list of
expected sources must be a list, not a set.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.api.dependencies import get_llm_client
from harness.core.config import DEFAULT_MODEL_NAME, LlmConfig
from harness.core.interfaces import (
    LlmClient,
    LlmConfigurationError,
    LlmError,
    LlmResponseFormatError,
    LlmUnavailableError,
    TokenUsage,
)

SYSTEM_PROMPT = "Fill the schema. Follow the user's request as literally as possible."


class Baseline(BaseModel):
    """Control probe. If this one fails, nothing below means anything."""

    model_config = ConfigDict(extra="forbid")

    value: str


class StringLength(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1, max_length=5)


class StringPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(pattern=r"^[A-Z]{3}$")


class IntRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int = Field(ge=1, le=5)


class IntExclusiveRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int = Field(gt=1, lt=5)


class IntMultipleOf(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int = Field(multiple_of=10)


class FloatRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)


class ListLength(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[str] = Field(min_length=2, max_length=3)


class UniqueItems(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: set[str]


class FieldDefault(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = "unset"


class OptionalField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | None


class LiteralChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Literal["correct", "incorrect"]


class FieldDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(description="Always the single word Paris, nothing else.")


class Nested(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")

        city: str

    value: Inner


class FreeFormDict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: dict[str, str]


class FixedTuple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: tuple[str, int]


class DateTimeFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: datetime


class ExtraAllowed(BaseModel):
    """No extra="forbid". Strict mode requires additionalProperties: false."""

    value: str


class UnionOfModels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class Hit(BaseModel):
        model_config = ConfigDict(extra="forbid")

        city: str

    class Miss(BaseModel):
        model_config = ConfigDict(extra="forbid")

        reason: str

    value: Hit | Miss


@dataclass(frozen=True)
class Probe:
    name: str
    schema: type[BaseModel]
    user_message: str
    honored_if: str
    """What to look for in the printed value to judge whether it was enforced."""


PROBES = [
    Probe("baseline", Baseline, "Value is Paris.", "any string comes back"),
    Probe(
        "string_length",
        StringLength,
        "Value is a long sentence about the capital of France.",
        "value is at most 5 characters",
    ),
    Probe(
        "string_pattern",
        StringPattern,
        "Value is the lowercase word paris.",
        "value is three uppercase letters",
    ),
    Probe("int_range", IntRange, "Value is 900.", "value is between 1 and 5"),
    Probe(
        "int_exclusive_range",
        IntExclusiveRange,
        "Value is 5.",
        "value is 2, 3 or 4, never 1 or 5",
    ),
    Probe("int_multiple_of", IntMultipleOf, "Value is 7.", "value is divisible by 10"),
    Probe("float_range", FloatRange, "Value is 42.5.", "value is between 0.0 and 1.0"),
    Probe(
        "list_length",
        ListLength,
        "Values are ten European capitals.",
        "2 or 3 entries come back",
    ),
    Probe(
        "unique_items",
        UniqueItems,
        "Values are Paris, Paris and Paris.",
        "duplicates collapsed rather than rejected",
    ),
    Probe(
        "field_default",
        FieldDefault,
        "Leave the value out entirely.",
        "schema accepted at all, since strict wants every field required",
    ),
    Probe("optional_field", OptionalField, "Value is unknown.", "null comes back"),
    Probe(
        "literal_choice",
        LiteralChoice,
        "Value is maybe.",
        "one of the two literals",
    ),
    Probe(
        "field_description",
        FieldDescription,
        "Value is Lyon.",
        "description steered the value to Paris",
    ),
    Probe("nested_model", Nested, "The city is Paris.", "nested object comes back"),
    Probe(
        "free_form_dict",
        FreeFormDict,
        "Value maps country names to capitals.",
        "schema accepted despite open-ended keys",
    ),
    Probe(
        "fixed_tuple",
        FixedTuple,
        "Value is Paris and 2.",
        "positional types accepted",
    ),
    Probe(
        "datetime_format",
        DateTimeFormat,
        "Value is the 14th of July 1789.",
        "parsed into a datetime",
    ),
    Probe(
        "extra_allowed",
        ExtraAllowed,
        "Value is Paris.",
        "the SDK closed the object for us",
    ),
    Probe(
        "union_of_models",
        UnionOfModels,
        "The city is Paris.",
        "one of the two branches",
    ),
]


def run_probe(probe: Probe, llm: LlmClient, config: LlmConfig) -> TokenUsage | None:
    try:
        result = llm.complete_structured(
            system_prompt=SYSTEM_PROMPT,
            user_message=probe.user_message,
            config=config,
            schema=probe.schema,
        )
    except LlmConfigurationError as exc:
        # BadRequestError lands here, and a rejected schema is a BadRequest.
        # The client's own message is generic, so the reason is in the cause.
        print(f"  REJECTED  {_describe_cause(exc)}")
        return None
    except LlmResponseFormatError as exc:
        print(f"  ACCEPTED, no usable answer: {exc}")
        return None
    except LlmUnavailableError as exc:
        print(f"  INCONCLUSIVE (provider unavailable): {exc}")
        return None
    except LlmError as exc:
        print(f"  INCONCLUSIVE ({type(exc).__name__}): {exc}")
        return None

    print(f"  ACCEPTED  value={result.parsed.model_dump()!r}")
    print(f"  honored if {probe.honored_if}")
    return result.usage


def _describe_cause(exc: LlmError) -> str:
    cause = exc.__cause__
    if cause is None:
        return str(exc)
    return f"{type(cause).__name__}: {cause}"


def main(argv: list[str]) -> int:
    model_name = argv[0] if argv else DEFAULT_MODEL_NAME
    wanted = set(argv[1:])
    probes = [p for p in PROBES if not wanted or p.name in wanted]
    if not probes:
        print(f"No probe matched {sorted(wanted)}.")
        print(f"Known probes: {', '.join(p.name for p in PROBES)}")
        return 1

    # temperature unset on purpose, see LlmConfig. The 2026-08-18 run sent 0.0
    # and lost 16 probes to it; three still came back with a schema error, so a
    # parameter error does not reliably shadow the schema error either.
    config = LlmConfig(model_name=model_name, max_output_tokens=256)
    llm = get_llm_client()

    total_tokens = 0
    for probe in probes:
        print(f"\n{probe.name}: {probe.user_message}")
        usage = run_probe(probe, llm, config)
        if usage is not None:
            total_tokens += usage.total_tokens

    print(f"\n{len(probes)} probes against {model_name}, {total_tokens} tokens total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
