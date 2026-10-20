"""Which backbone layers to use for the RGB memory bank (issue #18).

Every combination runs on the same 28x28 patch grid with the full training
set, so the only thing that changes is which feature levels go into a patch.
layer1 is averaged down from 56x56, layer3 interpolated up from 14x14.

usage: uv run python scripts/run_layer_ablation.py data/raw/mvtec3d --cache data/processed
Takes 1.5-2 hours on an M4. Results go to results/ablation/layers/.
"""

import argparse

from factorylens import experiments
from factorylens.data import mvtec3d
from factorylens.models.patchcore import PatchCoreConfig

LAYER_SETS = [
    ("layer1",),
    ("layer2",),
    ("layer3",),
    ("layer1", "layer2"),
    ("layer2", "layer3"),
    ("layer1", "layer2", "layer3"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--results", default="results/ablation/layers")
    args = parser.parse_args()

    for layers in LAYER_SETS:
        config = PatchCoreConfig(layers=layers, grid=28)
        for category in mvtec3d.CATEGORIES:
            r = experiments.run_rgb(
                args.root,
                args.cache,
                category,
                model_config=config,
                results_dir=args.results,
                scores_dir=None,
            )
            print(
                f"{'+'.join(layers):22s} {category:12s} image {r['image_auroc']:.3f} "
                f"aupro {r['aupro']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
