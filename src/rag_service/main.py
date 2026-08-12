import logging

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from rag_service.api.routers import analyze_router, health_router
from rag_service.core.interfaces import LlmError

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
