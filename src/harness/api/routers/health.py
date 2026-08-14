from importlib.metadata import version as package_version

from fastapi import APIRouter

from harness.schemas.responses import HealthResponse, VersionResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/version")
def version() -> VersionResponse:
    return VersionResponse(version=package_version("retrieval-regression-harness"))
