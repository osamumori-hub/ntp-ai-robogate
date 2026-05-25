"""Build label_queue/ — the focused set the user will manually label.

Pulls directly from source dataset (no dependency on the curated upload set):
  - All originals (filtered for visible beetles)
  - 5 evenly-sampled frames from each video

Re-running wipes prior queue.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path(r"E:/task/26-5-22/New folder/dataset-20260522T071458Z-3-001/dataset")
KUBIAKA_IMG_SRC = SOURCE_DIR / "right image and video" / "images"
KUBIAKA_VIDEO_SRC = SOURCE_DIR / "right image and video" / "video"
NG_IMG_SRC = SOURCE_DIR / "NG"
NG_VIDEO_SRC = SOURCE_DIR / "NG" / "NG_video"

QUEUE = PROJECT_ROOT / "label_queue"

EXCLUDE_KEYWORDS = ("frass", "furasu", "hurasu", "damage", "adultexit_hole",
                    "dassyutukou", "deadcherry")
FRAMES_PER_VIDEO = 5


def reset() -> None:
    if QUEUE.exists():
        shutil.rmtree(QUEUE)
    (QUEUE / "kubiaka").mkdir(parents=True, exist_ok=True)
    (QUEUE / "other_kamikiri").mkdir(parents=True, exist_ok=True)


def copy_originals(src: Path, dst: Path, filter_frass: bool = True) -> int:
    if not src.exists():
        return 0
    n = 0
    for p in sorted(src.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if filter_frass and any(k in p.name.lower() for k in EXCLUDE_KEYWORDS):
            continue
        shutil.copy2(p, dst / f"orig_{p.name}")
        n += 1
    return n


def extract_evenly(video_path: Path, out_dir: Path, stem: str, n_frames: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return 0
    step = max(total // (n_frames + 1), 1)
    written = 0
    try:
        for i in range(1, n_frames + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ok, frame = cap.read()
            if not ok:
                continue
            out_path = out_dir / f"frame_{stem}_{i:02d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            written += 1
    finally:
        cap.release()
    return written


def safe_stem(p: Path) -> str:
    s = p.stem
    if "kensyokin" in s.lower() or "懸賞金" in s:
        return "kensyokin"
    if "100" in s:
        return "100tai"
    if "駆除作戦" in s:
        return "kujosakusen"
    if "kamikiri" in s.lower():
        return "ng_kamikiri"
    if "ゴマダラ" in s:
        return "ng_gomadara"
    if "ルリボシ" in s or "昆虫探検" in s:
        return "ng_ruriboshi"
    return s[:10]


def main() -> None:
    reset()

    # Kubiaka
    out = QUEUE / "kubiaka"
    n_orig = copy_originals(KUBIAKA_IMG_SRC, out, filter_frass=True)
    n_frames = 0
    if KUBIAKA_VIDEO_SRC.exists():
        for vp in sorted(KUBIAKA_VIDEO_SRC.iterdir()):
            if vp.suffix.lower() in {".mp4", ".mov"}:
                n_frames += extract_evenly(vp, out, safe_stem(vp), FRAMES_PER_VIDEO)
    print(f"[kubiaka] {n_orig} originals + {n_frames} video frames")

    # Other kamikiri
    out = QUEUE / "other_kamikiri"
    n_orig = copy_originals(NG_IMG_SRC, out, filter_frass=False)
    n_frames = 0
    if NG_VIDEO_SRC.exists():
        for vp in sorted(NG_VIDEO_SRC.iterdir()):
            if vp.suffix.lower() in {".mp4", ".mov"}:
                n_frames += extract_evenly(vp, out, safe_stem(vp), FRAMES_PER_VIDEO)
    print(f"[other_kamikiri] {n_orig} originals + {n_frames} video frames")

    total = sum(len(list((QUEUE / c).iterdir())) for c in ("kubiaka", "other_kamikiri"))
    print(f"\nQueue: {total} images at {QUEUE}")


if __name__ == "__main__":
    main()
