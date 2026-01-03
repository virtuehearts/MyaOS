import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.mark.load
def test_chat_responsiveness_under_low_resources() -> None:
    if os.getenv("RUN_LOAD_TESTS") != "1":
        pytest.skip("Set RUN_LOAD_TESTS=1 to execute load tests.")

    max_seconds = float(os.getenv("CHAT_LOAD_MAX_SECONDS", "10"))
    request_count = int(os.getenv("CHAT_LOAD_REQUESTS", "40"))
    workers = int(os.getenv("CHAT_LOAD_WORKERS", "4"))

    client = TestClient(app)

    def send_request(index: int) -> None:
        response = client.post("/apps/chat", json={"message": f"ping {index}"})
        assert response.status_code == 200

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(send_request, range(request_count)))
    elapsed = time.perf_counter() - start

    assert elapsed <= max_seconds
