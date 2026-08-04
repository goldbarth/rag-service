from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TextRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    
    text: str = Field(min_length=1)


class TextResponse(BaseModel):
    
    result: str
    num_chars: int


class HealthResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str
