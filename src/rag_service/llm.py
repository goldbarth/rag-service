from dataclasses import dataclass

from openai import OpenAI, OpenAIError


@dataclass(frozen=True)
class LlmConfig:
    model_name: str
    
    
class LlmError(Exception):
    pass


def complete(
        client: OpenAI,
        config: LlmConfig,
        system_prompt: str, 
        user_message: str
) -> str:
    try:
        response = client.responses.create(
            model=config.model_name,
            instructions=system_prompt,
            input=user_message,
        )
    except OpenAIError as exc:
        raise LlmError(f"Model {config.model_name} did not answer.") from exc

    return response.output_text
