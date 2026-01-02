from fastapi.testclient import TestClient

from backend.main import app, reset_memory_store


client = TestClient(app)


def test_json_export_import_roundtrip() -> None:
    reset_memory_store()
    create_response = client.post(
        "/memory",
        json={
            "content": "User appreciated a calm response.",
            "tags": ["feedback", "positive"],
            "source_tags": ["chat", "user-42"],
            "virtue_markers": {"temperance": 0.7, "kindness": 0.9},
            "salience": 0.82,
        },
    )
    assert create_response.status_code == 200

    export_response = client.get("/memory/export/json")
    assert export_response.status_code == 200
    payload = export_response.json()
    assert payload["memories"]

    reset_memory_store()
    import_response = client.post("/memory/import/json?replace=true", json=payload)
    assert import_response.status_code == 200
    assert import_response.json()["imported"] == 1

    list_response = client.get("/memory")
    assert list_response.status_code == 200
    assert len(list_response.json()["memories"]) == 1


def test_xml_export_import_roundtrip() -> None:
    reset_memory_store()
    create_response = client.post(
        "/memory",
        json={
            "content": "System logged a reflective insight.",
            "tags": ["reflection"],
            "source_tags": ["system", "log"],
            "virtue_markers": {"wisdom": 0.88},
            "salience": 0.75,
        },
    )
    assert create_response.status_code == 200

    export_response = client.get("/memory/export/xml")
    assert export_response.status_code == 200
    xml_payload = export_response.text
    assert "<memory_module>" in xml_payload

    reset_memory_store()
    import_response = client.post(
        "/memory/import/xml?replace=true",
        data=xml_payload,
        headers={"Content-Type": "application/xml"},
    )
    assert import_response.status_code == 200
    assert import_response.json()["imported"] == 1

    list_response = client.get("/memory")
    assert list_response.status_code == 200
    assert len(list_response.json()["memories"]) == 1
