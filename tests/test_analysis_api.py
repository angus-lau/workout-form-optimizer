from fastapi.testclient import TestClient
from src.api import main

client = TestClient(main.app)


def test_form_analysis_quality():
    # simple pose where knee is fully extended (180°)
    # classifier thresholds default to min=90 max=90, so 180 should be "bad form"
    pose = {"hip": (0, 0, 0), "knee": (0, 1, 0), "ankle": (0, 2, 0)}
    payload = {"pose": pose, "exercise_type": "squat"}

    resp = client.post("/api/form/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # quality field should be present and indicate bad form for extreme angle
    assert data.get("quality") == "bad form"
