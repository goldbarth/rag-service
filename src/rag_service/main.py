import logging

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from rag_service.llm import LlmError
from rag_service.routers import analyze_router, health_router

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(analyze_router)

app.include_router(health_router)


@app.exception_handler(LlmError)
def handle_llm_error(request: Request, exc: LlmError) -> JSONResponse:
    logger.exception("LLM call failed")
    return JSONResponse(
        status_code=502, content={"detail": "upstream model unavailable"}
    )
