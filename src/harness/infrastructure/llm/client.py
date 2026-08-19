import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    omit,
)
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_usage import ResponseUsage as OpenAiResponseUsage
from pydantic import BaseModel, ValidationError

from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmCompletion,
    LlmConfigurationError,
    LlmError,
    LlmResponseFormatError,
    LlmStructuredCompletion,
    LlmToolCompletion,
    LlmToolError,
    LlmToolStopReason,
    LlmUnavailableError,
    TokenUsage,
    ToolRound,
    ToolSpec,
)

logger = logging.getLogger(__name__)


def _to_token_usage(
    usage: OpenAiResponseUsage | None, model_name: str
) -> TokenUsage | None:
    if usage is None:
        logger.warning(
            "LLM call completed without usage data",
            extra={"model_name": model_name},
        )
        return None

    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
    )


LlmToolOutcome = LlmToolStopReason | Literal["failed"]
"""How a tool run ended, for the log only.

A stop reason describes a run that returned something. "failed" covers the runs
that raised instead, and those are billed exactly the same, so the log needs a
word for them that the port does not.
"""


def _log_tool_run(
    model_name: str, rounds: Sequence[ToolRound], outcome: LlmToolOutcome
) -> None:
    """Log one line per finished tool run.

    complete() logs every call it makes; without this the tool path only ever
    logged its warnings, so a successful multi-round run left no trace and its
    cost could not be reconstructed from the logs.
    """
    billed = [r.usage for r in rounds if r.usage is not None]
    logger.info(
        "LLM tool run completed",
        extra={
            "model_name": model_name,
            "outcome": outcome,
            "rounds": len(rounds),
            "tool_calls": sum(len(r.tool_names) for r in rounds),
            "input_tokens": sum(u.input_tokens for u in billed),
            "output_tokens": sum(u.output_tokens for u in billed),
            "total_tokens": sum(u.total_tokens for u in billed),
            "cached_tokens": sum(u.cached_tokens for u in billed),
            "cache_write_tokens": sum(u.cache_write_tokens for u in billed),
            "reasoning_tokens": sum(u.reasoning_tokens for u in billed),
        },
    )


def _to_tool_param(spec: ToolSpec[Any]) -> FunctionToolParam:
    """Describe one tool the way the Responses API wants it.

    Precondition on spec.params, unchecked here: strict mode wants
    additionalProperties: false and every property in required, so the model
    needs model_config = ConfigDict(extra="forbid") and no field defaults. Use
    `str | None` where a value may be absent, since strict makes every field
    required and a default therefore never fires. A model that breaks this
    still builds a schema, and the provider rejects it as a BadRequestError,
    which arrives as LlmConfigurationError far from the tool that caused it.

    The SDK has to_strict_json_schema for exactly this, but it lives in
    openai.lib._tools and is private, and model_json_schema() on a model that
    holds the precondition produces the same document.
    """
    param: FunctionToolParam = {
        "type": "function",
        "name": spec.name,
        "parameters": spec.params.model_json_schema(),
        "strict": True,
    }
    description = spec.description or spec.params.__doc__
    if description is not None:
        # The key stays out when there is nothing to say. An empty string would
        # tell the model the tool has an (empty) description.
        param["description"] = description
    return param


class OpenAiLlmClient:
    """LlmClient adapter for the OpenAI Responses API."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        try:
            response = self._client.responses.create(
                model=config.model_name,
                instructions=system_prompt,
                input=user_message,
                temperature=config.temperature
                if config.temperature is not None
                else omit,
                max_output_tokens=config.max_output_tokens
                if config.max_output_tokens is not None
                else omit,
            )
        except (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
        ) as exc:
            raise LlmConfigurationError(
                f"Request for model {config.model_name} was rejected."
            ) from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} is currently unavailable."
            ) from exc
        except OpenAIError as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} did not answer."
            ) from exc

        details = response.incomplete_details
        incomplete_reason = details.reason if details is not None else None

        text = response.output_text
        if incomplete_reason is not None:
            # The provider stopped early, so an empty text is explained and not
            # a sign that the model is unavailable.
            logger.warning(
                "LLM response is incomplete",
                extra={
                    "model_name": config.model_name,
                    "incomplete_reason": incomplete_reason,
                },
            )
        elif not text.strip():
            raise LlmUnavailableError(
                f"The response for the {config.model_name} model contains no content."
            )

        usage = response.usage
        token_usage = _to_token_usage(usage, config.model_name)
        if token_usage is not None:
            logger.info(
                "LLM call completed",
                extra={
                    "model_name": config.model_name,
                    "input_tokens": token_usage.input_tokens,
                    "output_tokens": token_usage.output_tokens,
                    "total_tokens": token_usage.total_tokens,
                    "cached_tokens": token_usage.cached_tokens,
                    "cache_write_tokens": token_usage.cache_write_tokens,
                    "reasoning_tokens": token_usage.reasoning_tokens,
                },
            )

        return LlmCompletion(text, token_usage, incomplete_reason)

    def complete_structured[T: BaseModel](
        self, system_prompt: str, user_message: str, config: LlmConfig, schema: type[T]
    ) -> LlmStructuredCompletion[T]:
        try:
            response = self._client.responses.parse(
                model=config.model_name,
                instructions=system_prompt,
                input=user_message,
                text_format=schema,
                temperature=config.temperature
                if config.temperature is not None
                else omit,
                max_output_tokens=config.max_output_tokens
                if config.max_output_tokens is not None
                else omit,
            )
        except (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
        ) as exc:
            raise LlmConfigurationError(
                f"Request for model {config.model_name} was rejected."
            ) from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} is currently unavailable."
            ) from exc
        except ValidationError as exc:
            # Known gap: the SDK validates inside parse(), so the response object
            # and its usage are lost here. The call was billed but stays
            # unrecorded. Closing this would mean create() plus our own
            # model_validate_json, and with it the schema generation we
            # deliberately left to the SDK.
            raise LlmResponseFormatError(
                f"Response from {config.model_name} did not satisfy {schema.__name__}."
            ) from exc
        except OpenAIError as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} did not answer."
            ) from exc

        parsed = response.output_parsed
        if parsed is None:
            details = response.incomplete_details
            reason = (
                details.reason
                if details is not None and details.reason is not None
                else "refusal"
            )
            raise LlmResponseFormatError(
                f"Response from {config.model_name} could not be parsed ({reason})."
            )

        token_usage = _to_token_usage(response.usage, config.model_name)

        return LlmStructuredCompletion(parsed=parsed, usage=token_usage)

    def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        max_rounds: int = 5,
    ) -> LlmToolCompletion:
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be at least 1, got {max_rounds}.")

        by_name = {s.name: s for s in tools}
        if len(by_name) != len(tools):
            counts = Counter(s.name for s in tools)
            duplicates = sorted(name for name, count in counts.items() if count > 1)
            raise ValueError(f"Duplicate tool names: {', '.join(duplicates)}.")
        tool_params = [_to_tool_param(s) for s in tools]
        items: list[Any] = [{"role": "user", "content": user_message}]
        rounds: list[ToolRound] = []
        last_response_text = ""

        # Any raise below leaves a run that already spent rounds. Those calls
        # were billed, so the cost line has to go out before the exception
        # does, or a failed run is the one run whose price nobody can see.
        try:
            for round_index in range(max_rounds):
                try:
                    response = self._client.responses.create(
                        model=config.model_name,
                        instructions=system_prompt,
                        input=items,
                        tools=tool_params,
                        temperature=config.temperature
                        if config.temperature is not None
                        else omit,
                        max_output_tokens=config.max_output_tokens
                        if config.max_output_tokens is not None
                        else omit,
                    )
                except (
                    AuthenticationError,
                    PermissionDeniedError,
                    BadRequestError,
                    NotFoundError,
                ) as exc:
                    raise LlmConfigurationError(
                        f"Request for model {config.model_name} was rejected."
                    ) from exc
                except (RateLimitError, APIConnectionError) as exc:
                    raise LlmUnavailableError(
                        f"Model {config.model_name} is currently unavailable."
                    ) from exc
                except OpenAIError as exc:
                    raise LlmUnavailableError(
                        f"Model {config.model_name} did not answer."
                    ) from exc

                usage = _to_token_usage(response.usage, config.model_name)
                calls = [i for i in response.output if i.type == "function_call"]
                rounds.append(ToolRound(tuple(c.name for c in calls), usage))
                last_response_text = response.output_text

                details = response.incomplete_details
                if details is not None:
                    # The provider cut this round off, so any call it contains may be
                    # truncated. We stop instead of running a handler on half an
                    # argument object.
                    logger.warning(
                        "LLM tool response is incomplete",
                        extra={
                            "model_name": config.model_name,
                            "incomplete_reason": details.reason,
                            "round": round_index,
                        },
                    )
                    _log_tool_run(config.model_name, rounds, "incomplete_details")
                    return LlmToolCompletion(
                        response.output_text, tuple(rounds), "incomplete_details"
                    )

                if not calls:
                    text = response.output_text
                    if not text.strip():
                        # Nothing stopped the model early and it asked for no tool,
                        # so an empty answer has no explanation left.
                        raise LlmUnavailableError(
                            f"The response for the {config.model_name} model "
                            f"contains no content."
                        )

                    _log_tool_run(config.model_name, rounds, "completed")
                    return LlmToolCompletion(text, tuple(rounds), "completed")

                if round_index == max_rounds - 1:
                    # No round left to feed the results into, so we stop before doing
                    # work whose output nobody would read.
                    break

                # The calls must go back, otherwise the output is rejected with a 400.
                items += response.output
                for call in calls:
                    spec = by_name.get(call.name)
                    if spec is None:
                        raise LlmResponseFormatError(
                            f"Model {config.model_name} called the unknown tool "
                            f"{call.name!r}."
                        )
                    try:
                        args = spec.params.model_validate_json(call.arguments)
                    except ValidationError as exc:
                        raise LlmResponseFormatError(
                            f"Arguments for tool {call.name!r} did not satisfy "
                            f"{spec.params.__name__}."
                        ) from exc
                    try:
                        output = spec.handler(args)
                    except Exception as exc:
                        # A handler is our code, so this is not a provider failure.
                        # Wrapping it keeps the caller's single except intact and
                        # names the tool, which a raw exception from three frames
                        # down does not.
                        raise LlmToolError(
                            f"Tool {call.name!r} failed in round {round_index}."
                        ) from exc

                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": output,
                        }
                    )

        except LlmError:
            _log_tool_run(config.model_name, rounds, "failed")
            raise

        _log_tool_run(config.model_name, rounds, "max_rounds")
        return LlmToolCompletion(last_response_text, tuple(rounds), "max_rounds")
