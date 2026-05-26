from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import config


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    # Try each configured path
    for font_path_str in config.JAPANESE_FONT_PATHS:
        font_path = Path(font_path_str)
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except Exception:
                continue
    
    # Last resort: download NotoSansJP at runtime (requires requests + internet)
    try:
        import urllib.request, tempfile
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
        tmp = Path(tempfile.gettempdir()) / "NotoSansCJKjp.otf"
        if not tmp.exists():
            urllib.request.urlretrieve(url, tmp)
        return ImageFont.truetype(str(tmp), size=size)
    except Exception:
        pass
    
    return ImageFont.load_default()  # garbled, but won't crash

class Detector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> list[dict]:
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                })
        return detections

    def annotate(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        # Convert BGR (OpenCV) to RGB (Pillow) for text rendering, then back.
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        h, w = image.shape[:2]
        font_size = max(int(min(h, w) * 0.035), 18)
        font = _load_font(font_size)
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            # config.CLASS_COLORS is BGR for OpenCV; convert to RGB for Pillow.
            b, g, r = config.CLASS_COLORS.get(det["class_name"], config.DEFAULT_COLOR)
            color_rgb = (r, g, b)
            label = config.CLASS_LABELS_SHORT_JP.get(det["class_name"], config.DEFAULT_LABEL_SHORT_JP)
            text = f"{label} {det['confidence']:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
            # Text background for readability
            try:
                left, top, right, bottom = font.getbbox(text)
                tw, th = right - left, bottom - top
            except AttributeError:
                tw, th = draw.textsize(text, font=font)
            pad = 4
            ty1 = max(y1 - th - pad * 2, 0)
            tx2 = min(x1 + tw + pad * 2, w - 1)
            draw.rectangle([x1, ty1, tx2, ty1 + th + pad * 2], fill=color_rgb)
            draw.text((x1 + pad, ty1 + pad), text, fill=(255, 255, 255), font=font)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
