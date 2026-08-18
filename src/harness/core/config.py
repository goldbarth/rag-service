from functools import lru_cache
from typing import Annotated

from pydantic import BaseModel, Field, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: SecretStr
    timeout: float = 30.0


# The one place the project's model is named. Every call site draws from here,
# so a model change is one edit and cannot drift between API and scripts.
#
# Open: this is a moving alias, not a pinned snapshot. A provider side model
# change would look like a retrieval regression in a run diff. Pin a snapshot
# here once the sweep runner records configurations per run.
DEFAULT_MODEL_NAME = "gpt-5.6-luna"


class LlmConfig(BaseModel):
    model_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    # Reasoning models reject this parameter outright. Leave it None for them,
    # the adapter then sends omit. Confirmed against gpt-5.6-luna, 2026-08-18.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, gt=0)


@lru_cache
def get_settings() -> Settings:
    # openai_api_key is read from the environment, not passed in.
    return Settings()  # pyright: ignore[reportCallIssue]
