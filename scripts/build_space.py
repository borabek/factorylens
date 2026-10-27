"""Assemble the Hugging Face Space for the demo, and upload it.

The Space gets the files in space/, the fitted models (factorylens fit
--half) and the prepared test and validation parts of every category with a
model. requirements.txt pins this commit of the code from GitHub, so the
commit has to be pushed first. Models and data never go into the git
repository: they are derived from MVTec 3D-AD (CC BY-NC-SA 4.0) and the Space
says so.

usage: uv run python scripts/build_space.py --cache data/processed            # build only
       uv run python scripts/build_space.py --cache data/processed --upload   # needs hf auth login
"""

import argparse
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

from factorylens.data import cache

REPO = "https://github.com/borabek/factorylens"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("build/space"))
    parser.add_argument("--repo-id", help="default: <your user>/factorylens")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit your changes first: the Space installs the code from GitHub")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"push {commit[:7]} first: the Space installs it from GitHub")

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "models").mkdir(parents=True)
    shutil.copy("space/app.py", args.out / "app.py")
    readme = Path("space/README.md").read_text()
    readme = readme.replace("SDK_VERSION", version("gradio")).replace("COMMIT", commit[:7])
    (args.out / "README.md").write_text(readme)
    (args.out / "requirements.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        f"torch=={version('torch').split('+')[0]}\n"
        f"torchvision=={version('torchvision').split('+')[0]}\n"
        f"open3d=={version('open3d')}\n"
        f"factorylens[demo] @ git+{REPO}@{commit}\n"
    )

    settings = cache.Settings()
    data = args.out / "data" / settings.dirname()
    data.mkdir(parents=True)
    models = sorted(args.models.glob("*.pt"))
    for model in models:
        shutil.copy(model, args.out / "models" / model.name)
        for split in ("test", "validation"):
            src = cache.cache_path(args.cache, model.stem, split, settings)
            shutil.copy(src, data / src.name)
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"{args.out}: {len(models)} products, {size:.0f} MB, code at {commit[:7]}")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        repo_id = args.repo_id or f"{api.whoami()['name']}/factorylens"
        api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(
            folder_path=args.out,
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"FactoryLens demo at {commit[:7]}",
        )
        print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
