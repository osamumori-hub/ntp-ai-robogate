"""Fine-tune YOLOv8n on the labeled kubiaka/other_kamikiri dataset.

Two input paths supported:

A) Local Roboflow export — pass --dataset PATH pointing at the extracted ZIP
   folder. Expected structure:
       PATH/
       ├── data.yaml          (Roboflow provides this)
       ├── train/{images,labels}/
       ├── valid/{images,labels}/
       └── test/{images,labels}/

B) Roboflow API download — pass --rf-key, --rf-workspace, --rf-project, --rf-version
   to fetch the dataset programmatically. Requires `pip install roboflow`.

After training, copies the best weights to models/robogate_v1.pt so the demo
picks them up on next start.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
TARGET_WEIGHTS = MODELS_DIR / "robogate_v1.pt"


def fetch_roboflow(api_key: str, workspace: str, project: str, version: int) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("roboflow not installed. Run: pip install roboflow")
    rf = Roboflow(api_key=api_key)
    p = rf.workspace(workspace).project(project)
    dataset = p.version(version).download("yolov8", location=str(PROJECT_ROOT / "dataset"))
    return Path(dataset.location)


def patch_data_yaml(dataset_dir: Path) -> Path:
    """Rewrite data.yaml so paths are absolute (Roboflow exports relative paths).

    Also enforces our class order: kubiaka=0, other_kamikiri=1. If Roboflow
    exported them in a different order, this surfaces a clear error.
    """
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        sys.exit(f"data.yaml missing from dataset: {yaml_path}")

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    expected = ["kubiaka", "other_kamikiri"]
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    if names != expected:
        print(
            f"WARN: data.yaml class order is {names}, expected {expected}.\n"
            "      The demo's config.py assumes id 0 = kubiaka, 1 = other_kamikiri.\n"
            "      Re-export from Roboflow with that class order, or update config.py."
        )

    for split in ("train", "val", "valid", "test"):
        if split in cfg:
            cfg[split] = str((dataset_dir / cfg[split]).resolve())
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, help="Path to extracted Roboflow YOLOv8 export")
    parser.add_argument("--rf-key", help="Roboflow API key")
    parser.add_argument("--rf-workspace", help="Roboflow workspace slug")
    parser.add_argument("--rf-project", help="Roboflow project slug")
    parser.add_argument("--rf-version", type=int, help="Roboflow version number")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base", default="yolov8n.pt", help="Base weights to fine-tune from")
    args = parser.parse_args()

    if args.dataset:
        dataset_dir = args.dataset.resolve()
    elif args.rf_key and args.rf_workspace and args.rf_project and args.rf_version:
        dataset_dir = fetch_roboflow(args.rf_key, args.rf_workspace, args.rf_project, args.rf_version)
    else:
        sys.exit("Pass either --dataset PATH or all four --rf-* flags.")

    if not dataset_dir.exists():
        sys.exit(f"Dataset folder does not exist: {dataset_dir}")

    yaml_path = patch_data_yaml(dataset_dir)
    print(f"Dataset: {dataset_dir}")
    print(f"data.yaml: {yaml_path}")

    model = YOLO(args.base)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RUNS_DIR),
        name="robogate",
        exist_ok=True,
    )

    # Find the best weights
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"best.pt not found at {best}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, TARGET_WEIGHTS)
    print(f"\nCopied {best} -> {TARGET_WEIGHTS}")
    print("Restart streamlit to pick up the new weights.")


if __name__ == "__main__":
    main()
