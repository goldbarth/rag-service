from fastapi.testclient import TestClient
from rag_service.main import app

client = TestClient(app)

def test_analyze_is_valid_text():
    response = client.post("/analyze", json={"text": "I love Donuts :)"})
    assert response.status_code == 200
    assert response.json()["result"] == "I love Donuts :)"
    assert response.json()["num_chars"] == 16
    
def test_analyze_is_not_valid_text_too_short():
    response = client.post("/analyze", json={"text": "PI"})
    detail = response.json()["detail"][0]
    assert response.status_code == 422
    assert detail["type"] == "string_too_short"
    assert detail["loc"] == ["body", "text"]

def test_analyze_is_not_valid_missing_text():
    response = client.post("/analyze", json={})
    detail = response.json()["detail"][0]
    assert response.status_code == 422
    assert detail["type"] == "missing"
    assert detail["loc"] == ["body", "text"]