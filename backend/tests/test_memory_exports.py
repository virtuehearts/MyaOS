from fastapi.testclient import TestClient

from backend.main import reset_memory_store


def test_json_export_import_roundtrip(
    client: TestClient, auth_context: tuple[dict[str, str], str]
) -> None:
    headers, user_id = auth_context
    reset_memory_store(user_id)
    create_response = client.post(
        "/memory",
        headers=headers,
        json={
            "content": "User appreciated a calm response.",
            "tags": ["feedback", "positive"],
            "source_tags": ["chat", "user-42"],
            "virtue_markers": {"temperance": 0.7, "kindness": 0.9},
            "salience": 0.82,
        },
    )
    assert create_response.status_code == 200

    export_response = client.get("/memory/export/json", headers=headers)
    assert export_response.status_code == 200
    payload = export_response.json()
    assert payload["memories"]

    reset_memory_store(user_id)
    import_response = client.post(
        "/memory/import/json?replace=true", headers=headers, json=payload
    )
    assert import_response.status_code == 200
    assert import_response.json()["imported"] == 1

    list_response = client.get("/memory", headers=headers)
    assert list_response.status_code == 200
    assert len(list_response.json()["memories"]) == 1


def test_xml_export_import_roundtrip(
    client: TestClient, auth_context: tuple[dict[str, str], str]
) -> None:
    headers, user_id = auth_context
    reset_memory_store(user_id)
    create_response = client.post(
        "/memory",
        headers=headers,
        json={
            "content": "System logged a reflective insight.",
            "tags": ["reflection"],
            "source_tags": ["system", "log"],
            "virtue_markers": {"wisdom": 0.88},
            "salience": 0.75,
        },
    )
    assert create_response.status_code == 200

    export_response = client.get("/memory/export/xml", headers=headers)
    assert export_response.status_code == 200
    xml_payload = export_response.text
    assert "<memory_module>" in xml_payload

    reset_memory_store(user_id)
    import_response = client.post(
        "/memory/import/xml?replace=true",
        data=xml_payload,
        headers={**headers, "Content-Type": "application/xml"},
    )
    assert import_response.status_code == 200
    assert import_response.json()["imported"] == 1

    list_response = client.get("/memory", headers=headers)
    assert list_response.status_code == 200
    assert len(list_response.json()["memories"]) == 1
