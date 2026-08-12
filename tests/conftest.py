import pytest
from fastapi.testclient import TestClient

from rag_service.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
