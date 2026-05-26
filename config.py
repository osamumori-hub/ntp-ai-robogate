from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
VIDEOS_DIR = ASSETS_DIR / "videos"

# Update this when the fine-tuned model is ready
DEFAULT_MODEL_PATH = MODELS_DIR / "robogate_v1.pt"
# COCO pretrained, auto-downloaded by ultralytics on first use
FALLBACK_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

DEFAULT_CONFIDENCE = 0.5
DEFAULT_FRAME_SKIP = 3

# Class name -> Japanese label (shown in Streamlit tables and headings)
# kubiaka          : クビアカツヤカミキリ (the invasive target species, Aromia bungii)
# other_kamikiri   : native longhorn-beetle lookalikes (ホタルカミキリ etc.) — NOT the target
CLASS_LABELS_JP = {
    "kubiaka": "クビアカツヤカミキリ",
    "other_kamikiri": "他のカミキリムシ (対象外)",
}

# Class name -> BGR color for OpenCV
CLASS_COLORS = {
    "kubiaka": (0, 0, 255),         # red — target
    "other_kamikiri": (255, 0, 0),  # blue — not a target
}

# Class name -> short Japanese label drawn ON the image (kept short for readability)
CLASS_LABELS_SHORT_JP = {
    "kubiaka": "クビアカ",
    "other_kamikiri": "対象外",
}

DEFAULT_COLOR = (128, 128, 128)  # gray for unknown classes
DEFAULT_LABEL_JP = "その他"
DEFAULT_LABEL_SHORT_JP = "その他"

# Japanese-capable font (Google Noto Sans JP, ships with Windows 10/11)
JAPANESE_FONT_PATHS = [
    r"C:\Windows\Fonts\NotoSansJP-VF.ttf",       # Windows, Noto
    r"C:\Windows\Fonts\msgothic.ttc",              # Windows, MS Gothic (always present)
    r"C:\Windows\Fonts\meiryo.ttc",                # Windows, Meiryo
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",          # macOS
]