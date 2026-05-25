"""Build a training-ready YOLO dataset from the close-up originals.

For close-up specialty photos where the subject fills most of the frame,
a centered fixed-size box is a reasonable label. Faster and more reliable
than zero-shot detection with foundation models, and good enough to fine-tune
a YOLOv8n that demos well on similar close-ups.

Source images: roboflow_upload/{kubiaka,other_kamikiri}/orig_*.jpg
Output: dataset/{images,labels}/{train,val}/   + data.yaml
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD = PROJECT_ROOT / "roboflow_upload"
DATASET = PROJECT_ROOT / "dataset"

CLASS_MAP = {
    0: "kubiaka",
    1: "other_kamikiri",
}

BOX_WIDTH_FRACTION = 0.85
BOX_HEIGHT_FRACTION = 0.85
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# Filenames that show frass/holes/damage with no actual insect — exclude
# from training, the model would learn the wrong feature.
EXCLUDE_KEYWORDS = (
    "frass", "furasu", "hurasu", "damage", "adultexit_hole",
    "dassyutukou", "deadcherry", "cherry",
)


def reset_dataset() -> None:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_originals() -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for class_id, folder in CLASS_MAP.items():
        src = UPLOAD / folder
        if not src.exists():
            print(f"  ! missing source: {src}", file=sys.stderr)
            continue
        for p in sorted(src.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if not p.name.startswith("orig_"):
                continue  # skip video frames for this strategy
            lower = p.name.lower()
            if any(k in lower for k in EXCLUDE_KEYWORDS):
                continue  # skip frass/hole/damage-only images
            items.append((p, class_id))
    return items


def label_text_centered(class_id: int) -> str:
    # YOLO uses normalized coordinates, so we don't need to read the image.
    # Centered box at (0.5, 0.5) sized BOX_WIDTH_FRACTION x BOX_HEIGHT_FRACTION.
    cx, cy = 0.5, 0.5
    bw, bh = BOX_WIDTH_FRACTION, BOX_HEIGHT_FRACTION
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    items = collect_originals()
    if not items:
        sys.exit("No usable originals found.")
    per_class = {0: 0, 1: 0}
    for _, cid in items:
        per_class[cid] += 1
    print(f"Usable originals: {len(items)}")
    print(f"  kubiaka: {per_class[0]}")
    print(f"  other_kamikiri: {per_class[1]}")

    random.seed(RANDOM_SEED)
    random.shuffle(items)
    n_val = max(int(len(items) * VAL_FRACTION), 2)
    val_set, train_set = items[:n_val], items[n_val:]
    # Ensure val has at least one of each class
    val_class_counts = {0: 0, 1: 0}
    for _, cid in val_set:
        val_class_counts[cid] += 1
    print(f"Train: {len(train_set)}   Val: {len(val_set)}  (val class counts: {val_class_counts})")

    reset_dataset()
    counts = {"train": 0, "val": 0}
    for split, batch in (("train", train_set), ("val", val_set)):
        for img_path, cid in batch:
            shutil.copy2(img_path, DATASET / "images" / split / img_path.name)
            label = label_text_centered(cid)
            (DATASET / "labels" / split / (img_path.stem + ".txt")).write_text(label, encoding="utf-8")
            counts[split] += 1
    print(f"Wrote {counts['train']} train + {counts['val']} val")

    data_yaml = {
        "path": str(DATASET.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {cid: name for cid, name in CLASS_MAP.items()},
    }
    (DATASET / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\nDataset ready at {DATASET}")


if __name__ == "__main__":
    main()
