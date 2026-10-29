"""Interactive demo of the fitted inspectors.

    factorylens demo --models models

Two views:

  one part       pick a category and a test part, optionally break one sensor
                 with a FactoryLens-C corruption, and see both sensors' anomaly
                 maps, the fused map, the decision and the sensor weights. The
                 window is filled with normal parts in the same sensor state
                 first, as on a line that has been running for a while;
  a sensor fails stream a category's test parts in random order, break one sensor
                 half way through, and follow the weights and the alarms part by
                 part, next to plain late fusion with the same threshold.

The logic lives in plain functions so it can be tested without Gradio, which
is only imported by `build_app` (needs the `demo` extra).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402

from factorylens import corruptions  # noqa: E402
from factorylens.data import cache  # noqa: E402
from factorylens.data.cache import Prepared  # noqa: E402
from factorylens.data.mvtec3d import valid_points  # noqa: E402
from factorylens.inspector import Inspection, Inspector  # noqa: E402

NONE = "none"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d6336c"


def one(p: Prepared, i: int) -> Prepared:
    s = slice(i, i + 1)
    return replace(
        p,
        rgb=p.rgb[s],
        xyz=p.xyz[s],
        foreground=p.foreground[s],
        mask=p.mask[s],
        defect=p.defect[s],
        stem=p.stem[s],
    )


def corrupt(p: Prepared, name: str, severity: int, seed: int = 0) -> Prepared:
    if name == NONE:
        return p
    c = corruptions.CORRUPTIONS[name]
    return corruptions.corrupt(p, name, min(severity, c.severities), seed)


def shade(xyz: np.ndarray) -> np.ndarray:
    """Hillshade of the depth map as an RGB image, white where there is no point."""
    valid = valid_points(xyz)
    z = np.where(valid, xyz[..., 2].astype(float), np.nan)
    gy, gx = np.gradient(z)
    n = np.stack([-gx / 2e-4, -gy / 2e-4, np.ones_like(z)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    light = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)
    grey = np.where(valid, 0.15 + 0.8 * np.clip(np.nan_to_num(n @ light), 0, 1), 1.0)
    return np.asarray(np.round(255 * np.repeat(grey[..., None], 3, -1)), np.uint8)


def overlay(rgb: np.ndarray, amap: np.ndarray, vmax: float = 6.0) -> np.ndarray:
    """Anomaly map (in validation SDs) over the image; 0 and below shows the image only."""
    level = np.clip(amap / vmax, 0, 1)
    heat = colormaps["inferno"](level)[..., :3]
    alpha = 0.65 * level[..., None]
    out = (1 - alpha) * (rgb / 255.0) + alpha * heat
    return np.asarray(np.round(255 * out), np.uint8)


def summary(r: Inspection, truth: str | None = None) -> str:
    verdict = "**ALARM**" if r.alarm else "no alarm"
    lines = [
        f"### {verdict}",
        f"score {r.score:.2f} (threshold in the model), p-value {r.p_value:.3f}",
        "",
        "| sensor | weight | health (SDs beyond validation) | own score |",
        "|---|---|---|---|",
        f"| RGB | {r.weight_rgb:.2f} | {r.health_rgb:+.1f} | {r.score_rgb:.3f} |",
        f"| 3D | {r.weight_3d:.2f} | {r.health_3d:+.1f} | {r.score_3d:.3f} |",
    ]
    if r.sensor_alert:
        lines += ["", f"Sensor alert: **{r.sensor_alert}** counted for less than half."]
    if truth is not None:
        lines += ["", f"Ground truth: `{truth}`"]
    return "\n".join(lines)


@dataclass
class OnePart:
    rgb: np.ndarray
    depth: np.ndarray
    fused: np.ndarray
    rgb_map: np.ndarray
    map_3d: np.ndarray
    text: str
    result: Inspection


def inspect_one(
    inspector: Inspector,
    test: Prepared,
    validation: Prepared,
    index: int,
    corruption: str,
    severity: int,
) -> OnePart:
    """One part, as if the line had been running in the same sensor state for a while.

    The reliability window is filled first with normal validation parts under the
    same corruption. Judged alone, with an empty window, a large real defect can
    look like a broken sensor; on a line the parts before it say otherwise.
    """
    warm_up(inspector, validation, corruption, severity)
    part = corrupt(one(test, index), corruption, severity)
    r = inspector.inspect(part.rgb[0], part.xyz[0], part.foreground[0])
    assert r.map_rgb is not None and r.map_3d is not None
    rgb = part.rgb[0]
    return OnePart(
        rgb=rgb,
        depth=shade(part.xyz[0]),
        fused=overlay(rgb, r.anomaly_map),
        rgb_map=overlay(rgb, r.map_rgb),
        map_3d=overlay(shade(part.xyz[0]), r.map_3d),
        text=summary(r, test.defect[index]),
        result=r,
    )


@dataclass
class Stream:
    figure: Any
    text: str
    rows: list[dict[str, Any]]


# per inspector (dropped with it), per (corruption, severity): the window's signals
_WARM: weakref.WeakKeyDictionary[Inspector, dict[tuple[str, int], tuple[list[float], list[float]]]]
_WARM = weakref.WeakKeyDictionary()


def warm_up(inspector: Inspector, validation: Prepared, corruption: str, severity: int) -> None:
    """Fill the reliability window with normal parts in the given sensor state.

    The signals depend only on the model, the parts and the corruption, so they
    are computed once per combination and reused."""
    cached = _WARM.setdefault(inspector, {})
    key = (corruption, severity)
    if key not in cached:
        inspector.reset()
        window = inspector.config.reliability.window or 1
        before = corrupt(validation, corruption, severity)
        for i in range(min(window - 1, len(before))):
            inspector.inspect(before.rgb[i], before.xyz[i], before.foreground[i])
        cached[key] = (list(inspector.history["rgb"]), list(inspector.history["geo"]))
    inspector.reset()
    rgb, geo = cached[key]
    inspector.history["rgb"].extend(rgb)
    inspector.history["geo"].extend(geo)


def run_stream(
    inspector: Inspector,
    test: Prepared,
    corruption: str,
    severity: int,
    onset: float = 0.5,
    seed: int = 0,
    validation: Prepared | None = None,
    max_parts: int | None = None,
) -> Stream:
    """The test parts in random order; from `onset` on, one sensor is corrupted.

    With `validation`, the window starts filled with normal parts, as on a line
    that was already running. `max_parts` shortens the run (for slow machines)."""
    order = np.random.default_rng(seed).permutation(len(test))[:max_parts]
    start = int(round(onset * len(order)))
    bad = corrupt(test, corruption, severity)
    if validation is not None:
        warm_up(inspector, validation, NONE, 1)
    else:
        inspector.reset()
    rows = []
    for t, i in enumerate(order):
        src = bad if t >= start else test
        r = inspector.inspect(src.rgb[i], src.xyz[i], src.foreground[i])
        fs = inspector.fusion
        late = (
            (r.score_rgb - fs.scale_rgb.score_mean) / fs.scale_rgb.score_std
            + (r.score_3d - fs.scale_geo.score_mean) / fs.scale_geo.score_std
        ) / np.sqrt(2)
        rows.append(
            {
                "t": t,
                "defective": test.defect[i] != "good",
                "after": t >= start,
                "score": r.score,
                "late": float(late),
                "alarm": r.alarm,
                "late_alarm": bool(late > inspector.threshold),
                "weight_rgb": r.weight_rgb,
                "weight_3d": r.weight_3d,
                "health_rgb": r.health_rgb,
                "health_3d": r.health_3d,
            }
        )
    return Stream(
        _stream_figure(rows, start, inspector.threshold, corruption), _stream_text(rows), rows
    )


def _rate(rows: list[dict[str, Any]], key: str, defective: bool) -> float:
    sel = [r[key] for r in rows if r["after"] and r["defective"] == defective]
    return float(np.mean(sel)) if sel else float("nan")


def _stream_text(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "After the fault, same threshold for both:",
            "",
            "| | false alarms on good parts | defective parts caught |",
            "|---|---|---|",
            *(
                f"| {name} | {_rate(rows, key, False):.0%} | {_rate(rows, key, True):.0%} |"
                for name, key in (
                    ("plain late fusion", "late_alarm"),
                    ("reliability fusion", "alarm"),
                )
            ),
        ]
    )


def _stream_figure(
    rows: list[dict[str, Any]], start: int, threshold: float, corruption: str
) -> Any:
    t = np.array([r["t"] for r in rows])
    bad = np.array([r["defective"] for r in rows])
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, height_ratios=[1, 1.6])
    fig.patch.set_facecolor(SURFACE)
    a.plot(t, [r["weight_rgb"] for r in rows], color=BLUE, lw=1.5, label="RGB weight")
    a.plot(t, [r["weight_3d"] for r in rows], color=ORANGE, lw=1.5, label="3D weight")
    a.set_ylim(-0.05, 1.05)
    a.set_ylabel("sensor weight", color=INK_2)
    a.legend(frameon=False, fontsize=8, loc="lower left")
    for key, color, label in (
        ("late", MUTED, "plain late fusion"),
        ("score", INK, "reliability fusion"),
    ):
        y = np.array([r[key] for r in rows])
        b.scatter(
            t[~bad], y[~bad], s=10, color=color, marker="o", alpha=0.7, label=f"{label}, good part"
        )
        b.scatter(
            t[bad], y[bad], s=16, color=color, marker="x", alpha=0.9, label=f"{label}, defective"
        )
    b.axhline(threshold, color=RED, lw=1, ls=(0, (4, 3)), label="alarm threshold")
    b.set_yscale("symlog", linthresh=5)
    b.set_ylabel("fused score (validation SDs)", color=INK_2)
    b.set_xlabel("part")
    b.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    for ax in (a, b):
        ax.axvline(start, color=INK_2, lw=1)
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
    a.set_title(
        f"{corruption.replace('_', ' ')} from part {start} on", loc="left", fontsize=9, color=INK
    )
    fig.tight_layout()
    return fig


@lru_cache(maxsize=4)
def load_inspector(path: str) -> Inspector:
    return Inspector.load(path)


@lru_cache(maxsize=8)
def load_split(root: str, cache_dir: str, category: str, split: str) -> Prepared:
    return cache.get(root, cache_dir, category, split)


def build_app(
    models: str | Path, root: str | Path, cache_dir: str | Path, max_parts: int | None = None
) -> Any:
    import gradio as gr  # type: ignore[import-not-found, import-untyped, unused-ignore]  # only in the demo extra

    available = sorted(p.stem for p in Path(models).glob("*.pt"))
    if not available:
        raise FileNotFoundError(f"no models in {models}; fit some with factorylens fit")
    names = [NONE, *corruptions.CORRUPTIONS]

    def model(category: str) -> tuple[Inspector, Prepared, Prepared]:
        return (
            load_inspector(str(Path(models) / f"{category}.pt")),
            load_split(str(root), str(cache_dir), category, "test"),
            load_split(str(root), str(cache_dir), category, "validation"),
        )

    def single(category: str, index: int, corruption: str, severity: int) -> tuple[Any, ...]:
        inspector, test, validation = model(category)
        out = inspect_one(
            inspector, test, validation, min(int(index), len(test) - 1), corruption, int(severity)
        )
        return out.rgb, out.depth, out.rgb_map, out.map_3d, out.fused, out.text

    def stream(category: str, corruption: str, severity: int, seed: int) -> tuple[Any, ...]:
        inspector, test, validation = model(category)
        s = run_stream(
            inspector,
            test,
            corruption,
            int(severity),
            seed=int(seed),
            validation=validation,
            max_parts=max_parts,
        )
        return s.figure, s.text

    with gr.Blocks(title="FactoryLens") as app:
        gr.Markdown(
            "# FactoryLens\nRGB + 3D anomaly detection that notices when one of its sensors "
            "breaks. Models were fitted on normal parts of MVTec 3D-AD only."
        )
        with gr.Tab("One part"):
            with gr.Row():
                category = gr.Dropdown(available, value=available[0], label="product")
                index = gr.Slider(0, 160, value=0, step=1, label="test part")
                corruption = gr.Dropdown(names, value=NONE, label="break a sensor")
                severity = gr.Slider(1, 5, value=3, step=1, label="severity")
            go = gr.Button("Inspect")
            with gr.Row():
                rgb = gr.Image(label="RGB as the camera saw it")
                depth = gr.Image(label="3D (shaded depth)")
            with gr.Row():
                rgb_map = gr.Image(label="RGB anomaly map")
                map_3d = gr.Image(label="3D anomaly map")
                fused = gr.Image(label="fused anomaly map")
            text = gr.Markdown()
            go.click(
                single,
                [category, index, corruption, severity],
                [rgb, depth, rgb_map, map_3d, fused, text],
            )
        with gr.Tab("A sensor fails"):
            with gr.Row():
                s_category = gr.Dropdown(available, value=available[0], label="product")
                s_corruption = gr.Dropdown(names[1:], value="gaussian_noise", label="what breaks")
                s_severity = gr.Slider(1, 5, value=3, step=1, label="severity")
                s_seed = gr.Number(0, precision=0, label="order")
            run = gr.Button("Run the line")
            plot = gr.Plot()
            s_text = gr.Markdown()
            run.click(stream, [s_category, s_corruption, s_severity, s_seed], [plot, s_text])
        gr.Markdown(
            "Data: MVTec 3D-AD (Bergmann et al., 2022), CC BY-NC-SA 4.0. Code: "
            "[github.com/borabek/factorylens](https://github.com/borabek/factorylens)."
        )
    return app
