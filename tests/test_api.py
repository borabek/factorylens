import io

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image
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
    assert client.get("/status").json()["parts_in_window"] == 0
    client.post("/inspect", files=files(root))
    status = client.get("/status").json()
    assert status["parts_in_window"] == 1 and status["ms_per_part"] > 0
    assert client.post("/reset").json() == {"status": "reset"}
    assert client.get("/status").json()["parts_in_window"] == 0


def test_confirming_a_good_part(client, root):
    n = client.get("/status").json()["calibration_parts"]
    part = client.post("/inspect", files=files(root)).json()["part"]
    r = client.post(f"/confirm?part={part}")
    assert r.status_code == 200 and r.json() == {"calibration_parts": n + 1, "confirmed": 1}
    assert client.post(f"/confirm?part={part}").status_code == 404  # only once
    assert client.post("/confirm?part=999").status_code == 404


def test_a_dead_3d_sensor_gets_an_answer_not_a_500(client):
    empty = io.BytesIO()
    tifffile.imwrite(empty, np.zeros((64, 64, 3), np.float32), photometric="rgb")
    image = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(image, "PNG")
    r = client.post(
        "/inspect", files={"rgb": ("a.png", image.getvalue()), "xyz": ("a.tiff", empty.getvalue())}
    )
    assert r.status_code == 200, r.text
    assert np.isfinite(r.json()["score"])


def test_integer_depth_is_refused(client):
    image = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(image, "PNG")
    cloud = io.BytesIO()
    tifffile.imwrite(cloud, np.zeros((64, 64, 3), np.int16), photometric="rgb")
    r = client.post(
        "/inspect", files={"rgb": ("a.png", image.getvalue()), "xyz": ("a.tiff", cloud.getvalue())}
    )
    assert r.status_code == 422 and "floats in metres" in r.json()["detail"]


def test_health_says_which_version_fitted_the_model(client):
    body = client.get("/health").json()
    assert "fitted_with" in body and "factorylens" not in body
