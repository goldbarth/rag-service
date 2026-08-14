from functools import lru_cache
from typing import Annotated

from pydantic import BaseModel, Field, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: SecretStr
    timeout: float = 30.0


class LlmConfig(BaseModel):
    model_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


@lru_cache
def get_settings() -> Settings:
    # openai_api_key is read from the environment, not passed in.
    return Settings()  # pyright: ignore[reportCallIssue]
