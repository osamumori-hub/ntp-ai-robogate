# RoboGate Demo — 害虫検知デモ

VC pitch demo for NTP's RoboGate anomaly judgment engine. Detects クビアカツヤカミキリ (red-necked longhorn beetle) in photos and videos.

> This is a demo, not production code.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Model weights

Place a YOLOv8 `.pt` file at `models/robogate_v1.pt`. If no fine-tuned weights are available yet, the app falls back to `models/yolov8n.pt` (COCO pretrained, auto-downloaded by ultralytics on first run). See [models/README.md](models/README.md).

## Sample data

Drop test images into `assets/images/` (JPG/PNG) and test videos into `assets/videos/` (MP4/MOV). The "サンプル画像で試す" / "サンプル動画で試す" buttons pick one at random.

## Run

```bash
streamlit run app.py
```

## Tabs

1. **フォト検知** — upload a photo, see bounding boxes and a detection table.
2. **ビデオ検知** — upload a video, get a frame-annotated output plus timeline and stats.
3. **ロードマップ** — Phase 1–4 vision slide.
