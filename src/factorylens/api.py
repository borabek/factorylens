"""HTTP service around a saved Inspector.

    factorylens serve models/bagel.pt --port 8000

POST /inspect with two files, `rgb` (any image PIL reads) and `xyz` (the
organised point cloud as a 3-channel float TIFF, as MVTec 3D-AD stores it),
returns the decision for that part as JSON, including `sensor_alert` when a
sensor counted for less than half and the milliseconds each stage took.
GET /status reports sensor health and the median time per part over the
current window, and the recent alarm rate against the range expected from
the calibration set. Parts must be sent in the order they come off the line,
because the reliability window carries over from one request to the next.
POST /reset clears it. POST /confirm?part=N adds part N, checked by an
operator and found good, to the calibration set.

Needs the `serve` extra: uv sync --extra serve
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import tifffile
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from factorylens import __version__
from factorylens.inspector import Inspector


def read_rgb(data: bytes) -> np.ndarray:
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception as e:  # noqa: BLE001 - anything PIL cannot read is the client's problem
        raise HTTPException(422, f"rgb is not a readable image: {e}") from e


def read_xyz(data: bytes) -> np.ndarray:
    try:
        xyz = tifffile.imread(io.BytesIO(data)).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"xyz is not a readable TIFF: {e}") from e
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise HTTPException(422, f"xyz must be (H, W, 3), got {xyz.shape}")
    return xyz


def create_app(inspector: Inspector) -> FastAPI:
    app = FastAPI(title="FactoryLens", version=__version__)
    lock = threading.Lock()  # the window is shared state: one part at a time

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "alpha": inspector.config.alpha,
            "window": inspector.config.reliability.window,
            **inspector.meta,
        }

    @app.post("/inspect")
    async def inspect(
        rgb: Annotated[UploadFile, File()],
        xyz: Annotated[UploadFile, File()],
        with_map: bool = False,
    ) -> dict[str, Any]:
        image, cloud = read_rgb(await rgb.read()), read_xyz(await xyz.read())
        if image.shape[:2] != cloud.shape[:2]:
            raise HTTPException(
                422, f"rgb {image.shape[:2]} and xyz {cloud.shape[:2]} differ in size"
            )
        with lock:
            result = inspector.inspect_raw(image, cloud)
            timings = {k: round(1000 * v, 1) for k, v in inspector.timings.items()}
        return result.as_dict(with_map=with_map) | {"timings_ms": timings}

    @app.get("/status")
    def status() -> dict[str, Any]:
        with lock:
            return inspector.status()

    @app.post("/confirm")
    def confirm(part: int) -> dict[str, Any]:
        with lock:
            try:
                inspector.confirm_good(part)
            except KeyError as e:
                raise HTTPException(404, str(e.args[0])) from e
            except ValueError as e:
                raise HTTPException(409, str(e)) from e
            return {
                "calibration_parts": len(inspector.calibration),
                "confirmed": inspector.meta["confirmed"],
            }

    @app.post("/reset")
    def reset() -> dict[str, str]:
        with lock:
            inspector.reset()
        return {"status": "reset"}

    return app


def serve(model: str | Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn  # type: ignore[import-not-found, unused-ignore]  # only in the serve extra

    uvicorn.run(create_app(Inspector.load(model)), host=host, port=port)
