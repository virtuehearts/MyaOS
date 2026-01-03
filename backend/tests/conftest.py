import uuid

import pytest
from fastapi.testclient import TestClient

from backend.main import (
    OAUTH_STATES,
    RATE_LIMITER,
    SESSIONS,
    USER_LOOKUP_LOGS,
    USER_MEMORY_STORE,
    USER_PASSWORDS,
    USERS_BY_EMAIL,
    USERS_BY_ID,
    VECTOR_INDEX,
    app,
)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    USERS_BY_EMAIL.clear()
    USERS_BY_ID.clear()
    USER_PASSWORDS.clear()
    SESSIONS.clear()
    OAUTH_STATES.clear()
    RATE_LIMITER.hits.clear()
    USER_MEMORY_STORE.clear()
    USER_LOOKUP_LOGS.clear()
    VECTOR_INDEX.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def auth_context(client: TestClient) -> tuple[dict[str, str], str]:
    email = f"user-{uuid.uuid4().hex}@example.com"
    password = "supersecret"
    response = client.post(
        "/auth/register",
        json={"email": email, "password": password, "name": "Test User"},
    )
    assert response.status_code == 200
    token = response.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    profile = client.get("/auth/me", headers=headers).json()["user"]
    return headers, profile["user_id"]
