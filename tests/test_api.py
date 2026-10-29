import io

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from test_inspector import CONFIG, no_open3d  # noqa: F401  (fixture)

from factorylens import api
from factorylens.data import cache, mvtec3d
from factorylens.inspector import Inspector


@pytest.fixture
def client(root):
    splits = {
        s: cache.prepare(root, "bagel", s, CONFIG.sweep.settings) for s in ("train", "validation")
    }
    inspector = Inspector(CONFIG, device="cpu", pretrained=False).fit(
        splits["train"], splits["validation"]
    )
    return TestClient(api.create_app(inspector))


def files(root, i=0):
    s = mvtec3d.list_samples(root, "bagel", "test")[i]
    return {
        "rgb": ("rgb.png", s.rgb_path.read_bytes()),
        "xyz": ("xyz.tiff", s.xyz_path.read_bytes()),
    }


def test_health_names_the_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["category"] == "bagel" and body["window"] == 3


def test_inspect_returns_a_decision(client, root):
    r = client.post("/inspect", files=files(root))
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"score", "p_value", "alarm", "weight_rgb", "weight_3d", "sensor_alert"} <= set(body)
    assert "anomaly_map" not in body
    assert body["timings_ms"]["total"] >= body["timings_ms"]["backbone"] >= 0
    with_map = client.post("/inspect?with_map=true", files=files(root)).json()
    assert np.array(with_map["anomaly_map"]).shape == (32, 32)


def test_bad_uploads_are_rejected(client, root):
    good = files(root)
    assert (
        client.post("/inspect", files={**good, "rgb": ("x.png", b"not an image")}).status_code
        == 422
    )
    flat = io.BytesIO()
    tifffile.imwrite(flat, np.zeros((8, 8), np.float32))
    assert (
        client.post("/inspect", files={**good, "xyz": ("x.tiff", flat.getvalue())}).status_code
        == 422
    )
    small = io.BytesIO()
    tifffile.imwrite(small, np.zeros((8, 8, 3), np.float32))
    assert (
        client.post("/inspect", files={**good, "xyz": ("x.tiff", small.getvalue())}).status_code
        == 422
    )


def test_status_and_reset(client, root):
    assert client.get("/status").json() == {"parts_in_window": 0}
    client.post("/inspect", files=files(root))
    status = client.get("/status").json()
    assert status["parts_in_window"] == 1 and status["ms_per_part"] > 0
    assert client.post("/reset").json() == {"status": "reset"}
    assert client.get("/status").json() == {"parts_in_window": 0}
