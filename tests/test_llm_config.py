import pytest
from pydantic import ValidationError

from rag_service.core.config import LlmConfig


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_llm_config_rejects_temperature_outside_openai_range(
    temperature: float,
) -> None:
    with pytest.raises(ValidationError):
        LlmConfig(model_name="test-model", temperature=temperature)


@pytest.mark.parametrize("temperature", [0.0, 2.0, None])
def test_llm_config_accepts_valid_temperature_values(
    temperature: float | None,
) -> None:
    config = LlmConfig(model_name="test-model", temperature=temperature)

    assert config.temperature == temperature


@pytest.mark.parametrize("model_name", ["", "   ", "\n\t"])
def test_llm_config_rejects_blank_model_name(model_name: str) -> None:
    with pytest.raises(ValidationError):
        LlmConfig(model_name=model_name, temperature=0.2)


def test_llm_config_strips_model_name_whitespace() -> None:
    config = LlmConfig(model_name="  test-model  ", temperature=0.2)

    assert config.model_name == "test-model"
