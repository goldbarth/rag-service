def test_analyze_is_valid_text_returns_200(client):
    response = client.post("/analyze", json={"text": "I love Donuts :)"})
    assert response.status_code == 200
    assert response.json()["result"] == "I love Donuts :)"
    assert response.json()["num_chars"] == 16


def test_analyze_is_valid_text_trimmed_returns_200(client):
    response = client.post("/analyze", json={"text": "   I love Donuts :)     "})
    assert response.status_code == 200
    assert response.json()["result"] == "I love Donuts :)"
    assert response.json()["num_chars"] == 16


def test_analyze_rejects_whitespace_only_text_returns_422(client):
    response = client.post("/analyze", json={"text": "  "})
    detail = response.json()["detail"][0]
    assert response.status_code == 422
    assert detail["type"] == "string_too_short"
    assert detail["loc"] == ["body", "text"]


def test_analyze_is_not_valid_missing_text_returns_422(client):
    response = client.post("/analyze", json={})
    detail = response.json()["detail"][0]
    assert response.status_code == 422
    assert detail["type"] == "missing"
    assert detail["loc"] == ["body", "text"]


def test_analyze_rejects_misspelled_text_key_returns_422(client):
    response = client.post("/analyze", json={"txet": "I love Donuts :)"})
    errors = [
        {"type": detail["type"], "loc": detail["loc"]}
        for detail in response.json()["detail"]
    ]

    assert response.status_code == 422
    assert {"type": "missing", "loc": ["body", "text"]} in errors
    assert {"type": "extra_forbidden", "loc": ["body", "txet"]} in errors


def test_analyze_rejects_invalid_input_type_422(client):
    response = client.post("/analyze", json={"text": 123})
    detail = response.json()["detail"][0]
    assert response.status_code == 422
    assert detail["type"] == "string_type"
    assert detail["loc"] == ["body", "text"]
