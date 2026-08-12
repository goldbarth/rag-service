from typing import Literal

from pydantic import BaseModel


class TextResponse(BaseModel):
    result: str
    num_chars: int


class HealthResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str
