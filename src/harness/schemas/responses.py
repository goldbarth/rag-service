from typing import Literal

from pydantic import BaseModel

from harness.core.interfaces import LlmIncompleteReason, LlmToolStopReason


class TextResponse(BaseModel):
    result: str
    num_chars: int
    incomplete_reason: LlmIncompleteReason | None = None
    """Set when the provider stopped early. The result is then a partial answer."""


class HealthResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str


class RagResponse(BaseModel):
    result: str
    num_chars: int
    stop_reason: LlmToolStopReason
