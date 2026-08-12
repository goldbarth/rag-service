from fastapi import APIRouter

from rag_service.schemas.requests import TextRequest
from rag_service.schemas.responses import TextResponse

router = APIRouter(tags=["analyze"])


@router.post("/analyze")
def analyze(request: TextRequest) -> TextResponse:
    return TextResponse(result=request.text, num_chars=len(request.text))
