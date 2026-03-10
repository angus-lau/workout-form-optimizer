from pathlib import Path
from fastapi.testclient import TestClient

from src.api import main
from src.api import config

client = TestClient(main.app)


def test_videos_list_and_file(tmp_path, monkeypatch):
    # point configuration at temp directory
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    # create a dummy video file under data directory
    video = tmp_path / "foo.mp4"
    video.write_bytes(b"not a real video")

    resp = client.get("/api/videos")
    assert resp.status_code == 200
    data = resp.json()
    assert "videos" in data
    assert any(v["path"] == "foo.mp4" for v in data["videos"])

    # try fetching the file itself
    resp2 = client.get("/api/videos/file/foo.mp4")
    assert resp2.status_code == 200
    assert resp2.content == b"not a real video"


def test_invalid_video_path(monkeypatch):
    # prepare project root so resolve path will reject attempts to escape
    monkeypatch.setattr(config, "PROJECT_ROOT", Path("/tmp"))

    resp = client.get("/api/videos/file/../etc/passwd")
    assert resp.status_code == 400


def test_process_video_endpoint_monkeypatched(monkeypatch):
    # monkeypatch analyze_video to avoid needing real media
    dummy_result = {"frames": [], "video_info": {"width": 0, "height": 0, "fps": 0, "total_frames": 0}}
    def fake_analyze(path):
        return dummy_result

    monkeypatch.setattr("src.ui.opencv_demo.analyze_video", fake_analyze)
    
    # upload a file (contents don't matter)
    files = {"file": ("test.mp4", b"fake data", "video/mp4")}
    resp = client.post("/api/process-video", files=files)
    assert resp.status_code == 200
    assert resp.json() == dummy_result

    # request by path: patch DATA_DIR and put a file there
    tmp = Path("/tmp/video_test")
    tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATA_DIR", tmp)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp)
    f = tmp / "bar.mp4"
    f.write_bytes(b"x")
    resp2 = client.post("/api/process-video", params={"path": "bar.mp4"})
    assert resp2.status_code == 200
    assert resp2.json() == dummy_result
