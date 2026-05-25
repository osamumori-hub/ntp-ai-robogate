"""One-shot data prep for the RoboGate demo.

Run once after placing source videos. Produces:
  - assets/videos/demo_kubiaka.mp4, assets/videos/demo_NG.mp4 (short demo clips)
  - training_pool/{kubiaka,other_kamikiri}/*.jpg  (1 fps frame extracts from videos)

Re-running overwrites prior output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path(r"E:/task/26-5-22/New folder/dataset-20260522T071458Z-3-001/dataset")
ASSETS_VIDEOS = PROJECT_ROOT / "assets" / "videos"
TRAINING_POOL = PROJECT_ROOT / "training_pool"

# (source-relative path, output stem, label folder under training_pool, demo trim seconds or None)
# Kubiaka videos live under: dataset/right image and video/video/
# NG (other_kamikiri lookalike) videos live under: dataset/NG_video/
VIDEOS = [
    (
        "right image and video/video/【昆虫採集】ある地域で懸賞金がかけられている外来昆虫を乱獲してきました【クビアカツヤカミキリ】.mp4",
        "kubiaka_kensyokin",
        "kubiaka",
        15,
    ),
    (
        "right image and video/video/クビアカツヤカミキリを100体駆除、特定外来生物の見分け方とその生態.mp4",
        "kubiaka_100tai",
        "kubiaka",
        None,
    ),
    (
        "right image and video/video/クビアカツヤカミキリ駆除作戦1.mp4",  # full-width 1 normalized in code below
        "kubiaka_kujosakusen",
        "kubiaka",
        None,
    ),
    (
        "NG/NG_video/NG_kamikiri.mp4",
        "NG_kamikiri",
        "other_kamikiri",
        15,
    ),
    (
        "NG/NG_video/NG_gomadara.mp4",  # ゴマダラ — normalized
        "NG_gomadara",
        "other_kamikiri",
        None,
    ),
    (
        "NG/NG_video/NG_ruriboshi.mp4",  # ルリボシ — normalized
        "NG_ruriboshi",
        "other_kamikiri",
        None,
    ),
]


def find_video(rel: str, simplified_stem: str = "") -> Path:
    """Match a video by literal path first, then fall back to keyword search."""
    p = SOURCE_DIR / rel
    if p.exists():
        return p
    parent = (SOURCE_DIR / rel).parent
    if not parent.exists():
        raise FileNotFoundError(f"Folder missing: {parent}")
    keywords = {
        "kubiaka_kensyokin": ["懸賞金"],
        "kubiaka_100tai": ["100"],
        "kubiaka_kujosakusen": ["駆除作戦"],
        "NG_kamikiri": ["NG_kamikiri"],
        "NG_gomadara": ["ゴマダラ"],
        "NG_ruriboshi": ["ルリボシ", "昆虫探検"],
    }
    needles = keywords.get(simplified_stem) or [Path(rel).stem]
    for cand in parent.iterdir():
        if any(n in cand.name for n in needles):
            return cand
    raise FileNotFoundError(f"No video matched keywords {needles} in {parent}")


def extract_frames(video_path: Path, out_dir: Path, stem: str, target_fps: float = 1.0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! cannot open {video_path.name}", file=sys.stderr)
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / target_fps)), 1)
    idx, saved = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                out_path = out_dir / f"{stem}_f{saved:05d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved += 1
            idx += 1
    finally:
        cap.release()
    return saved


def trim_video(video_path: Path, out_path: Path, seconds: float) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! cannot open {video_path.name}", file=sys.stderr)
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    target_frames = int(fps * seconds)
    written = 0
    try:
        while written < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()
    return written > 0


def main() -> None:
    print(f"Source : {SOURCE_DIR}")
    print(f"Project: {PROJECT_ROOT}")
    if not SOURCE_DIR.exists():
        sys.exit(f"Source folder does not exist: {SOURCE_DIR}")

    total_frames = 0
    for rel, stem, label, trim_sec in VIDEOS:
        try:
            vp = find_video(rel, simplified_stem=stem)
        except FileNotFoundError as e:
            print(f"[skip] {rel} ({e})")
            continue
        print(f"\n[{stem}] {vp.name}")

        # Frame extraction for training
        out_dir = TRAINING_POOL / label
        n = extract_frames(vp, out_dir, stem)
        total_frames += n
        print(f"  + extracted {n} frames into {out_dir.relative_to(PROJECT_ROOT)}")

        # Demo trim (only first kubiaka + first NG)
        if trim_sec is not None:
            demo_name = "demo_kubiaka.mp4" if label == "kubiaka" else "demo_NG.mp4"
            demo_path = ASSETS_VIDEOS / demo_name
            ok = trim_video(vp, demo_path, trim_sec)
            if ok:
                print(f"  + trimmed {trim_sec}s demo clip -> {demo_path.relative_to(PROJECT_ROOT)}")

    print(f"\nDone. Total training frames extracted: {total_frames}")
    print(f"Upload contents of {TRAINING_POOL.relative_to(PROJECT_ROOT)}/ to Roboflow.")


if __name__ == "__main__":
    main()
