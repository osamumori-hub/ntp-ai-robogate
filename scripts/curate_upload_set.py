"""Build a curated upload set for Roboflow.

Picks all original stills + an even-sampled subset of training_pool frames.
Output: roboflow_upload/{kubiaka,other_kamikiri}/*.jpg

Re-running clears prior output.
"""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path(r"E:/task/26-5-22/New folder/dataset-20260522T071458Z-3-001/dataset")
KUBIAKA_ORIGINALS = SOURCE_DIR / "right image and video" / "images"
NG_ORIGINALS = SOURCE_DIR / "NG"
TRAINING_POOL = PROJECT_ROOT / "training_pool"
UPLOAD = PROJECT_ROOT / "roboflow_upload"

TARGET_PER_CLASS = 150  # frames to sample (originals are added on top)


def reset(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def even_sample(items: list[Path], k: int) -> list[Path]:
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _copy_image_dir(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    n = 0
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(p, dst / f"orig_{p.name}")
        n += 1
    return n


def copy_originals_kubiaka(dst: Path) -> int:
    return _copy_image_dir(KUBIAKA_ORIGINALS, dst)


def copy_originals_other(dst: Path) -> int:
    return _copy_image_dir(NG_ORIGINALS, dst)


def copy_sampled_frames(label: str, dst: Path, k: int) -> int:
    src_dir = TRAINING_POOL / label
    if not src_dir.exists():
        return 0
    frames = sorted(src_dir.glob("*.jpg"))
    picked = even_sample(frames, k)
    for p in picked:
        shutil.copy2(p, dst / p.name)
    return len(picked)


def main() -> None:
    reset(UPLOAD)
    for label, originals_fn in [
        ("kubiaka", copy_originals_kubiaka),
        ("other_kamikiri", copy_originals_other),
    ]:
        out = UPLOAD / label
        out.mkdir(parents=True, exist_ok=True)
        n_orig = originals_fn(out)
        n_frames = copy_sampled_frames(label, out, TARGET_PER_CLASS)
        total = n_orig + n_frames
        print(f"[{label}] {n_orig} originals + {n_frames} sampled frames = {total}")

    print(f"\nUpload contents of {UPLOAD.relative_to(PROJECT_ROOT)}/ to Roboflow.")
    print("Each subfolder maps to one Roboflow class. Use Roboflow's 'Upload' UI:")
    print("  1) Drag roboflow_upload/kubiaka/ -> assign class 'kubiaka' when prompted")
    print("  2) Drag roboflow_upload/other_kamikiri/ -> assign class 'other_kamikiri'")


if __name__ == "__main__":
    main()
