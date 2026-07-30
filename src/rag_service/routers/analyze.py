from fastapi import APIRouter

from rag_service.schemas import TextRequest, TextResponse

router = APIRouter(tags=["analyze"])


@router.post("/analyze")
def analyze(request: TextRequest) -> TextResponse:
    return TextResponse(result=request.text, num_chars=len(request.text))
