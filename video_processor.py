from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import cv2

from detector import Detector


def process_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    detector: Detector,
    frame_skip: int = 3,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> dict:
    input_path = str(input_path)
    output_path = str(output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_detections: list[dict] = []
    total_detections = 0
    confidence_sum = 0.0
    per_second_counts: dict[int, int] = {}
    frames_analyzed = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(frame_skip, 1) == 0:
                last_detections = detector.detect(frame)
                frames_analyzed += 1
                second = int(frame_idx / fps) if fps > 0 else 0
                per_second_counts[second] = (
                    per_second_counts.get(second, 0) + len(last_detections)
                )
                total_detections += len(last_detections)
                confidence_sum += sum(d["confidence"] for d in last_detections)

            annotated = detector.annotate(frame, last_detections)
            writer.write(annotated)

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(min(frame_idx / total_frames, 1.0))
    finally:
        cap.release()
        writer.release()

    avg_conf = (confidence_sum / total_detections) if total_detections else 0.0

    return {
        "total_frames": frame_idx,
        "frames_analyzed": frames_analyzed,
        "total_detections": total_detections,
        "average_confidence": avg_conf,
        "fps": fps,
        "per_second_counts": per_second_counts,
        "output_path": output_path,
    }
