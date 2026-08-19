from typing import Annotated, Any

from fastapi import APIRouter, Depends

from harness.api.dependencies import get_llm_client, get_llm_config, get_tools
from harness.core.config import LlmConfig
from harness.core.interfaces import ToolCompleter, ToolSpec
from harness.schemas.requests import TextRequest
from harness.schemas.responses import RagResponse

router = APIRouter(tags=["rag"])

SYSTEM_PROMPT = (
    "Answer the question from the indexed corpus, not from memory.\n"
    "Call search_sections first, then ground every claim in what it returned.\n"
    "Cite each claim as doc_id#section.\n"
    "If nothing relevant comes back, say so instead of guessing."
)


@router.post("/rag/analyze")
def rag_analyze(
    request: TextRequest,
    llm: Annotated[ToolCompleter, Depends(get_llm_client)],
    llm_config: Annotated[LlmConfig, Depends(get_llm_config)],
    tools: Annotated[list[ToolSpec[Any]], Depends(get_tools)],
) -> RagResponse:
    result = llm.complete_with_tools(
        system_prompt=SYSTEM_PROMPT,
        user_message=request.text,
        config=llm_config,
        tools=tools,
    )

    # stop_reason travels to the caller: "max_rounds" and "incomplete_details"
    # both carry a partial answer, and a 200 alone cannot say which one it is.
    return RagResponse(
        result=result.text,
        num_chars=len(result.text),
        stop_reason=result.stop_reason,
    )
