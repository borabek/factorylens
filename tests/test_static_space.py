import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import build_static_space as space  # noqa: E402


def test_one_good_part_and_one_of_each_defect_type_combined_last():
    defects = ["combined", "crack", "good", "hole", "good", "crack", "cut", "contamination"]
    assert space.pick_parts(defects) == [2, 7, 1, 6]  # good, contamination, crack, cut
    assert space.pick_parts(["good", "combined", "hole"]) == [0, 2, 1]


def test_summary_reads_the_result_files(tmp_path):
    rows = []
    for m, clean in {
        "rgb": 0.8,
        "fpfh": 0.79,
        "early": 0.91,
        "late": 0.88,
        "reliable_window": 0.88,
    }.items():
        for c, sensor, mean in (("jpeg", "rgb", 0.7), ("holes", "xyz", 0.75)):
            rows.append(
                {"model": m, "corruption": c, "sensor": sensor, "clean": clean, "mean": mean}
            )
    pd.DataFrame(rows).to_csv(tmp_path / "robustness_summary.csv", index=False)
    clean = [
        {"detector": d, "alpha": a, "false_alarm": 0.1, "detection": 0.6}
        for d in ("late", "reliable_window")
        for a in (0.05, 0.1)
    ]
    pd.DataFrame(clean).to_csv(tmp_path / "conformal_clean.csv", index=False)
    broken = [
        {"sensor": s, "detector": d, "false_alarm": fa, "detection": 0.5}
        for s in ("rgb", "xyz")
        for d, fa in (("late", 0.9), ("reliable_window", 0.2))
    ]
    pd.DataFrame(broken).to_csv(tmp_path / "conformal_broken.csv", index=False)

    s = space.summary(tmp_path)
    assert [d["model"] for d in s["detection"]][-1] == "reliability fusion"
    assert s["healthy"] == {"rgb_broken": 0.79, "3d_broken": 0.8}
    assert s["alarms"][0]["rgb_broken"] == [0.9, 0.5]
    assert s["alarms"][1]["clean"] == [0.1, 0.6]
