"""What the FF detectors rely on: permutation importance, SHAP, partial dependence.

Two kinds of model, both trained on day 1 and tested on day 2 (NoNose vs normal):

1. Supervised random forests on per-cycle features (mean and std of every
   logged robot channel), with and without two time features (hour of day,
   hours since the run started). Each model is explained three ways:
     - impurity importance, which only reflects the training day,
     - permutation importance of the test AUROC on day 2,
     - mean |SHAP| on day 2 (TreeExplainer).
   Partial dependence on the two gripper-load features shows why the R01
   feature, which the forest uses heavily on day 1, does nothing on day 2.
2. The unsupervised forecast detector over all robots (docs/data.md). Its
   score is a mean of per-channel squared residuals, so each robot's
   contribution is exact (for an additive score this *is* the SHAP value,
   up to the background mean). Permuting one robot's profiles across test
   cycles shows which robots help the detector and which hurt it.

usage: uv run --group analysis python scripts/ff/explain.py "data/raw/ff_analog/Analog Dataset"
Writes results/analysis/ff_explain_*.csv and docs/figures/ff_explain.png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance

from factorylens.data import ff
from factorylens.eval.metrics import image_auroc
from factorylens.models import sensor

sys.path.insert(0, str(Path(__file__).parent))
from figures import INK, INK_2, LABEL_COLORS, MUTED, plt  # noqa: E402

TIME = ["hour of day", "hours since start"]
BLUE, ORANGE, GREEN = LABEL_COLORS["normal"], LABEL_COLORS["NoNose"], "#1baf7a"


def short(channel: str) -> str:
    """M_R03_UJointAngle_Degree -> R03 U joint, I_R03_Gripper_Load -> R03 load."""
    robot = channel.split("_")[1]
    if "Gripper_Load" in channel:
        return f"{robot} load"
    if "Gripper_Pot" in channel:
        return f"{robot} gripper pos"
    return f"{robot} {channel.split('_')[2][0]} joint"


def features(
    profiles: np.ndarray, cycles: pd.DataFrame, channels: list[str], t0: pd.Timestamp
) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {}
    for i, c in enumerate(channels):
        cols[f"{short(c)} mean"] = profiles[:, :, i].mean(axis=1)
        cols[f"{short(c)} std"] = profiles[:, :, i].std(axis=1)
    start = cycles["start"]
    cols["hour of day"] = (start.dt.hour + start.dt.minute / 60).to_numpy()
    cols["hours since start"] = ((start - t0).dt.total_seconds() / 3600).to_numpy()
    return pd.DataFrame(cols)


def supervised(xtr: pd.DataFrame, ytr: np.ndarray, xte: pd.DataFrame, yte: np.ndarray):
    sets = {
        "time only": TIME,
        "time + sensors": list(xtr.columns),
        "sensors": [c for c in xtr.columns if c not in TIME],
        "R03 only": [c for c in xtr.columns if c.startswith("R03")],
    }
    models, rows, importance = {}, [], []
    for name, cols in sets.items():
        forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=0)
        forest.fit(xtr[cols], ytr)
        models[name] = (forest, cols)
        rows.append(
            {
                "features": name,
                "test_auroc": image_auroc(yte, forest.predict_proba(xte[cols])[:, 1]),
            }
        )
        perm = permutation_importance(
            forest, xte[cols], yte, scoring="roc_auc", n_repeats=30, random_state=0
        )
        values = shap.TreeExplainer(forest).shap_values(xte[cols])
        # older shap returns one array per class, newer a (n, features, classes) array
        positive = values[1] if isinstance(values, list) else values[..., 1]
        importance.append(
            pd.DataFrame(
                {
                    "model": name,
                    "feature": cols,
                    "impurity_train": forest.feature_importances_,
                    "permutation_test": perm.importances_mean,
                    "permutation_test_sd": perm.importances_std,
                    "mean_abs_shap_test": np.abs(positive).mean(axis=0),
                }
            )
        )
    return models, pd.DataFrame(rows), pd.concat(importance, ignore_index=True)


def detector(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, seed: int = 0):
    """Per-robot contribution to, and permutation importance for, the pooled forecast detector."""
    channels = sensor.ROBOT_CHANNELS
    fit_on = sensor.cycle_profiles(df, train, channels)[~train["anomalous"].to_numpy()]
    scored = sensor.cycle_profiles(df, test, channels)
    keep = sensor.usable(fit_on) & sensor.usable(scored)
    fit_on, scored = fit_on[..., keep], scored[..., keep]
    kept = [c for c, k in zip(channels, keep, strict=True) if k]
    robots = np.array([c.split("_")[1] for c in kept])
    labels = test["anomalous"].to_numpy()

    det = sensor.Forecast().fit(fit_on)
    per_channel = (det.residuals(scored) ** 2).mean(axis=1) / len(kept)  # sums to the score
    base = image_auroc(labels, per_channel.sum(axis=1))
    rng = np.random.default_rng(seed)
    rows = []
    for robot in sensor.ROBOTS:
        idx = np.flatnonzero(robots == robot)
        contribution = per_channel[:, idx].sum(axis=1)
        drops = []
        for _ in range(30):
            shuffled = scored.copy()
            shuffled[:, :, idx] = scored[rng.permutation(len(scored))][:, :, idx]
            drops.append(base - image_auroc(labels, det.score(shuffled)))
        rows.append(
            {
                "robot": robot,
                "channels": len(idx),
                "share_of_score": contribution.mean() / per_channel.sum(axis=1).mean(),
                "contribution_nonose_minus_normal": contribution[labels].mean()
                - contribution[~labels].mean(),
                "auroc_alone": image_auroc(labels, contribution),
                "permutation_drop": float(np.mean(drops)),
                "permutation_drop_sd": float(np.std(drops)),
            }
        )
    return base, pd.DataFrame(rows)


def figure(
    importance: pd.DataFrame,
    models: dict,
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xte: pd.DataFrame,
    yte: np.ndarray,
    det_base: float,
    det: pd.DataFrame,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), width_ratios=[1.25, 1, 0.9])

    # a: what the sensors-only forest uses, by training impurity vs test permutation
    ax = axes[0]
    imp = importance[importance["model"] == "sensors"].copy()
    top = imp.nlargest(4, "impurity_train")["feature"].tolist()
    top += [f for f in imp.nlargest(4, "permutation_test")["feature"] if f not in top]
    imp = imp.set_index("feature").loc[top[::-1]]
    y = np.arange(len(imp))
    bars = [
        ("impurity_train", "impurity (training day)", MUTED),
        ("mean_abs_shap_test", "mean |SHAP| (test day)", ORANGE),
        ("permutation_test", "permutation, AUROC (test day)", BLUE),
    ]
    for j, (col, label, color) in enumerate(bars):
        ax.barh(y + 0.27 * (1 - j), imp[col] / imp[col].max(), 0.26, color=color, label=label)
    ax.set_yticks(y, imp.index)
    ax.set_title("sensor-only forest: what does it use?", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_xlabel("importance, relative to the largest of each kind")

    # b: partial dependence of the sensor forest on R01 and R03 load, with both days' values
    ax = axes[1]
    forest, cols = models["sensors"]
    for row, (feature, color) in enumerate([("R01 load mean", INK), ("R03 load mean", GREEN)]):
        both = pd.concat([xtr[cols], xte[cols]])
        pd_result = partial_dependence(
            forest, both, [feature], grid_resolution=60, percentiles=(0, 1)
        )
        grid, avg = pd_result["grid_values"][0], pd_result["average"][0]
        lo, hi = both[feature].min(), both[feature].max()
        ax.plot((grid - lo) / (hi - lo), avg, color=color, lw=1.6, label=feature)
        # rugs below the curves: one per feature and day, coloured by the true class
        for x, labels, day in [(xtr, ytr, 1), (xte, yte, 2)]:
            base_y = -0.07 * (1 + day + 2 * row)
            vals = ((x[feature] - lo) / (hi - lo)).to_numpy()
            for flag, c in [(False, BLUE), (True, ORANGE)]:
                ax.scatter(
                    vals[labels == flag],
                    np.full((labels == flag).sum(), base_y),
                    s=9,
                    marker="|",
                    color=c,
                    alpha=0.7,
                )
            ax.text(
                1.02, base_y, f"{feature[:3]}, day {day}", fontsize=6.5, va="center", color=color
            )
    ax.set_xlabel("feature value (scaled to its range on both days)")
    ax.set_ylabel("P(no nose), partial dependence", color=INK_2)
    ax.set_title("partial dependence, values seen on each day", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="center left")

    # c: the unsupervised forecast detector, permuting one robot at a time
    ax = axes[2]
    d = det.set_index("robot").loc[list(sensor.ROBOTS)[::-1]]
    colors = [BLUE if v > 0 else ORANGE for v in d["permutation_drop"]]
    ax.barh(np.arange(len(d)), d["permutation_drop"], xerr=d["permutation_drop_sd"], color=colors)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(np.arange(len(d)), d.index)
    ax.set_xlabel("AUROC lost when this robot is shuffled")
    ax.set_title(
        f"forecast detector, all robots\n(AUROC {det_base:.2f})", loc="left", fontsize=9, color=INK
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/ff_explain.png"))
    args = parser.parse_args()

    df = ff.load_analog(args.root)
    cycles = ff.segment_cycles(df)
    train, test = ff.cross_day_split(cycles)
    channels = sensor.ROBOT_CHANNELS
    p_train = sensor.cycle_profiles(df, train, channels)
    p_test = sensor.cycle_profiles(df, test, channels)
    keep = sensor.usable(p_train) & sensor.usable(p_test)
    kept = [c for c, k in zip(channels, keep, strict=True) if k]
    t0 = cycles["start"].min()
    xtr = features(p_train[..., keep], train, kept, t0)
    xte = features(p_test[..., keep], test, kept, t0)
    ytr, yte = train["anomalous"].to_numpy(), test["anomalous"].to_numpy()

    models, scores, importance = supervised(xtr, ytr, xte, yte)
    det_base, det = detector(df, train, test)

    args.out.mkdir(parents=True, exist_ok=True)
    scores.round(4).to_csv(args.out / "ff_explain_models.csv", index=False)
    importance.round(4).to_csv(args.out / "ff_explain_importance.csv", index=False)
    det.round(4).to_csv(args.out / "ff_explain_detector.csv", index=False)
    figure(importance, models, xtr, ytr, xte, yte, det_base, det, args.figure)

    print(scores.round(3).to_string(index=False))
    for name in models:
        part = importance[importance["model"] == name]
        print(f"\n{name}")
        print(part.nlargest(4, "mean_abs_shap_test").round(3).to_string(index=False))
    print(f"\nforecast detector, all robots: AUROC {det_base:.3f}")
    print(det.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
