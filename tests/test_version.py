import tomllib
from pathlib import Path

pyproject_path = Path(__file__).parents[1] / "pyproject.toml"

with pyproject_path.open("rb") as file:
    pyproject = tomllib.load(file)


def test_version_returns_200(client):
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": pyproject["project"]["version"]}
