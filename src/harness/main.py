import logging

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from harness.api.routers import analyze_router, health_router
from harness.core.interfaces import LlmConfigurationError, LlmUnavailableError

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(analyze_router)

app.include_router(health_router)


@app.exception_handler(LlmConfigurationError)
def handle_llm_configuration_error(
    request: Request, exc: LlmConfigurationError
) -> JSONResponse:
    logger.exception("LLM call failed because of our own configuration")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.exception_handler(LlmUnavailableError)
def handle_llm_unavailable_error(
    request: Request, exc: LlmUnavailableError
) -> JSONResponse:
    logger.exception("LLM call failed because the provider did not answer")
    return JSONResponse(
        status_code=502, content={"detail": "upstream model unavailable"}
    )
