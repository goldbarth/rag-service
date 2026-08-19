import logging

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from harness.api.routers import analyze_router, health_router, rag_router
from harness.core.interfaces import (
    LlmConfigurationError,
    LlmResponseFormatError,
    LlmToolError,
    LlmUnavailableError,
)

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(analyze_router)

app.include_router(rag_router)

app.include_router(health_router)


@app.exception_handler(LlmConfigurationError)
def handle_llm_configuration_error(
    request: Request, exc: LlmConfigurationError
) -> JSONResponse:
    logger.exception("LLM call failed because of our own configuration")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.exception_handler(LlmToolError)
def handle_llm_tool_error(request: Request, exc: LlmToolError) -> JSONResponse:
    # Our handler failed, not the provider, so this is a 500 like any other
    # bug of ours and must not be reported as an upstream problem.
    logger.exception("LLM tool run failed inside one of our own handlers")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.exception_handler(LlmResponseFormatError)
def handle_llm_response_format_error(
    request: Request, exc: LlmResponseFormatError
) -> JSONResponse:
    # The provider answered, but with something we cannot use: a schema the
    # response missed, a tool we never offered, arguments that do not parse.
    # Same 502 as an absent answer, because the fault sits upstream either way
    # and the caller can do nothing differently.
    logger.exception("LLM call failed because the response could not be used")
    return JSONResponse(
        status_code=502, content={"detail": "upstream model answer was unusable"}
    )


@app.exception_handler(LlmUnavailableError)
def handle_llm_unavailable_error(
    request: Request, exc: LlmUnavailableError
) -> JSONResponse:
    logger.exception("LLM call failed because the provider did not answer")
    return JSONResponse(
        status_code=502, content={"detail": "upstream model unavailable"}
    )
