from typing import Annotated

from fastapi import APIRouter, Depends

from rag_service.api.dependencies import get_llm_client, get_llm_config
from rag_service.core.config import LlmConfig
from rag_service.core.interfaces import LlmClient
from rag_service.schemas.requests import TextRequest
from rag_service.schemas.responses import TextResponse

router = APIRouter(tags=["analyze"])

SYSTEM_PROMPT = "Analyze the given text and return a concise result."


@router.post("/analyze")
def analyze(
    request: TextRequest,
    llm: Annotated[LlmClient, Depends(get_llm_client)],
    llm_config: Annotated[LlmConfig, Depends(get_llm_config)],
) -> TextResponse:
    result = llm.complete(
        system_prompt=SYSTEM_PROMPT,
        user_message=request.text,
        config=llm_config,
    )

    return TextResponse(result=result, num_chars=len(result))
