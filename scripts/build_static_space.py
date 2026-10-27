"""Precompute the demo and assemble a static Hugging Face Space.

Gradio Spaces need a paid account; static ones do not. So everything the demo
would compute live is computed here, with the same functions
(`factorylens.demo`), and saved as images and one JSON file next to
space-static/index.html:

  - per product, one good part and one part of each of three defect types,
    clean and under nine FactoryLens-C corruptions at severity 3: both
    sensors' inputs, both anomaly maps, the fused map and the decision;
  - per product, a run in which a sensor fails halfway, for five corruptions:
    the figure and the alarm rates of plain late fusion and reliability
    fusion after the fault.

The images are MVTec 3D-AD derivatives (CC BY-NC-SA 4.0); the Space carries
that licence and the attribution. Nothing generated here goes into git.

usage: OMP_NUM_THREADS=4 uv run python scripts/build_static_space.py --cache data/processed
       ... --upload                      # needs hf auth login
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from factorylens import demo
from factorylens.data import cache
from factorylens.inspector import Inspector

SEVERITY = 3
CONDITIONS = [
    demo.NONE,
    "gaussian_noise",
    "defocus_blur",
    "jpeg",
    "occlusion",
    "rgb_missing",
    "depth_noise",
    "holes",
    "quantise",
    "xyz_missing",
]
STREAMS = ["gaussian_noise", "defocus_blur", "jpeg", "depth_noise", "xyz_missing"]
IMAGES = ("rgb", "depth", "rgb_map", "map_3d", "fused")


def pick_parts(defects: list[str], n_defects: int = 3) -> list[int]:
    """The first good part and the first part of up to n defect types ('combined' last)."""
    kinds = sorted({d for d in defects if d != "good"}, key=lambda k: (k == "combined", k))
    return [defects.index("good"), *(defects.index(k) for k in kinds[:n_defects])]


def save_jpeg(img: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=85)
    return path.as_posix()


def rate(rows: list[dict], key: str, defective: bool) -> float:
    sel = [r[key] for r in rows if r["after"] and r["defective"] == defective]
    return round(float(np.mean(sel)), 3) if sel else float("nan")


def product(name: str, models: Path, cache_dir: Path, out: Path) -> dict:
    inspector = Inspector.load(models / f"{name}.pt")
    test = cache.get("unused", cache_dir, name, "test")
    validation = cache.get("unused", cache_dir, name, "validation")
    parts = []
    for index in pick_parts(test.defect):
        part = {"index": index, "truth": test.defect[index], "conditions": {}}
        for condition in CONDITIONS:
            r = demo.inspect_one(inspector, test, validation, index, condition, SEVERITY)
            images = {
                key: save_jpeg(
                    getattr(r, key), out / "img" / name / f"{index}_{condition}_{key}.jpg"
                )
                for key in IMAGES
            }
            res = r.result
            part["conditions"][condition] = {
                "images": {k: Path(v).relative_to(out).as_posix() for k, v in images.items()},
                "score": round(res.score, 3),
                "p_value": round(res.p_value, 3),
                "alarm": bool(res.alarm),
                "weight_rgb": round(res.weight_rgb, 3),
                "weight_3d": round(res.weight_3d, 3),
                "health_rgb": round(res.health_rgb, 2),
                "health_3d": round(res.health_3d, 2),
            }
        parts.append(part)
        print(f"{name} part {index} ({test.defect[index]})", flush=True)

    streams = {}
    for corruption in STREAMS:
        s = demo.run_stream(inspector, test, corruption, SEVERITY, validation=validation)
        path = out / "img" / name / f"stream_{corruption}.png"
        s.figure.savefig(path, dpi=110)
        demo.plt.close(s.figure)
        streams[corruption] = {
            "figure": path.relative_to(out).as_posix(),
            "late_false_alarm": rate(s.rows, "late_alarm", False),
            "late_caught": rate(s.rows, "late_alarm", True),
            "reliable_false_alarm": rate(s.rows, "alarm", False),
            "reliable_caught": rate(s.rows, "alarm", True),
        }
        print(f"{name} stream {corruption}", flush=True)
    return {"parts": parts, "streams": streams, "threshold": round(float(inspector.threshold), 3)}


MODELS = {
    "rgb": "RGB only",
    "fpfh": "3D only",
    "early": "early fusion",
    "late": "late fusion",
    "reliable_window": "reliability fusion",
}


def summary(analysis: Path) -> dict:
    """Headline numbers for the Results tab, read from the repository's result files."""
    rob = pd.read_csv(analysis / "robustness_summary.csv")
    clean = rob.drop_duplicates("model").set_index("model")["clean"]
    by = rob.groupby(["sensor", "model"])["mean"].mean()
    detection = [
        {
            "model": MODELS[m],
            "clean": round(float(clean[m]), 3),
            "rgb_broken": round(float(by[("rgb", m)]), 3),
            "3d_broken": round(float(by[("xyz", m)]), 3),
        }
        for m in MODELS
    ]
    healthy = {
        "rgb_broken": round(float(clean["fpfh"]), 3),
        "3d_broken": round(float(clean["rgb"]), 3),
    }
    cc = pd.read_csv(analysis / "conformal_clean.csv")
    cc = cc[cc["alpha"] == 0.1].groupby("detector")[["false_alarm", "detection"]].mean()
    cb = pd.read_csv(analysis / "conformal_broken.csv")
    cb = cb.groupby(["sensor", "detector"])[["false_alarm", "detection"]].mean()
    alarms = [
        {
            "model": MODELS[m],
            "clean": [round(float(cc.loc[m, k]), 3) for k in ("false_alarm", "detection")],
            "rgb_broken": [
                round(float(cb.loc[("rgb", m), k]), 3) for k in ("false_alarm", "detection")
            ],
            "3d_broken": [
                round(float(cb.loc[("xyz", m), k]), 3) for k in ("false_alarm", "detection")
            ],
        }
        for m in ("late", "reliable_window")
    ]
    return {"detection": detection, "healthy": healthy, "alarms": alarms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("build/static-space"))
    parser.add_argument("--products", nargs="*", help="default: every model in --models")
    parser.add_argument("--repo-id", help="default: <your user>/factorylens")
    parser.add_argument("--analysis", type=Path, default=Path("results/analysis"))
    parser.add_argument(
        "--page-only",
        action="store_true",
        help="keep the images and per-part results in --out, refresh the page and the summary",
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if args.page_only:
        data = json.loads((args.out / "data.json").read_text())
    else:
        if args.out.exists():
            shutil.rmtree(args.out)
        args.out.mkdir(parents=True)
        names = args.products or sorted(p.stem for p in args.models.glob("*.pt"))
        data = {
            "severity": SEVERITY,
            "conditions": CONDITIONS,
            "streams": STREAMS,
            "products": {n: product(n, args.models, args.cache, args.out) for n in names},
        }
    for f in Path("space-static").iterdir():
        shutil.copy(f, args.out / f.name)
    readme = args.out / "README.md"
    readme.write_text(readme.read_text().replace("COMMIT", commit))
    data["commit"] = commit
    data["summary"] = summary(args.analysis)
    names = list(data["products"])
    (args.out / "data.json").write_text(json.dumps(data))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"{args.out}: {len(names)} products, {size:.0f} MB, code at {commit}")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_id = args.repo_id or f"{api.whoami()['name']}/factorylens"
        api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(
            folder_path=args.out,
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"FactoryLens static demo at {commit}",
        )
        print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
