"""The demo's 'a sensor fails' view as a static figure, for the docs.

usage: uv run python scripts/demo_figure.py data/raw/mvtec3d --cache data/processed
Needs models/bagel.pt (factorylens fit --category bagel). Writes docs/figures/demo_stream.png.
"""

import argparse
from pathlib import Path

from factorylens import demo
from factorylens.data import cache
from factorylens.inspector import Inspector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--category", default="bagel")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/demo_stream.png"))
    args = parser.parse_args()

    inspector = Inspector.load(f"models/{args.category}.pt")
    test = cache.get(args.root, args.cache, args.category, "test")
    validation = cache.get(args.root, args.cache, args.category, "validation")
    s = demo.run_stream(inspector, test, args.corruption, 3, validation=validation)
    s.figure.savefig(args.out, dpi=150)
    print(s.text)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
