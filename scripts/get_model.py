"""Fetch a pretrained dental-caries YOLO model into backend/models/caries.pt.

Two supported sources:

1. Roboflow Universe (recommended — many open dental-caries detection models).
   Requires a free Roboflow account + API key.

     pip install roboflow
     export ROBOFLOW_API_KEY=xxxxxxxx
     python scripts/get_model.py --source roboflow \
         --workspace <workspace> --project <project> --version <n>

   Find a model at https://universe.roboflow.com (search "dental caries" /
   "tooth decay" / "cavity" detection). Open it, click *Download this Dataset* or
   use the model snippet to read the workspace/project/version values.

2. Direct URL (e.g. a HuggingFace-hosted .pt you trust).

     python scripts/get_model.py --source url --url https://.../caries.pt

Use --name to install several models side by side (e.g. caries + caries_photo
for auto X-ray/photo routing):

     python scripts/get_model.py --source url --url https://.../best.pt --name caries_photo

If you already have weights, just copy them to backend/models/<name>.pt — no
script needed.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _dest(name: str) -> Path:
    return MODELS_DIR / f"{name}.pt"


def from_url(url: str, dest: Path) -> None:
    print(f"Downloading {url}\n         -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def from_roboflow(workspace: str, project: str, version: int, dest: Path) -> None:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Install roboflow first:  pip install roboflow")

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set ROBOFLOW_API_KEY (get a free key at https://roboflow.com).")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    model = proj.version(version).model
    # Roboflow caches the weights locally; copy them to our models dir.
    weights = getattr(model, "weights_path", None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if weights and Path(weights).exists():
        import shutil

        shutil.copy(weights, dest)
        print(f"Saved {dest}")
    else:
        print(
            "Roboflow returned a hosted model. Export it as YOLOv8 weights from the "
            "Universe UI (Deploy > Download Weights) and copy the .pt to "
            f"{dest}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a caries YOLO model.")
    parser.add_argument("--source", choices=["roboflow", "url"], required=True)
    parser.add_argument("--url", help="Direct .pt URL (for --source url)")
    parser.add_argument("--name", default="caries",
                        help="Model name / filename stem (default: caries). Use a "
                             "distinct name to keep several models switchable.")
    parser.add_argument("--workspace")
    parser.add_argument("--project")
    parser.add_argument("--version", type=int)
    args = parser.parse_args()

    dest = _dest(args.name)
    if args.source == "url":
        if not args.url:
            sys.exit("--url is required for --source url")
        from_url(args.url, dest)
    else:
        if not (args.workspace and args.project and args.version):
            sys.exit("--workspace, --project and --version are required for roboflow")
        from_roboflow(args.workspace, args.project, args.version, dest)


if __name__ == "__main__":
    main()
