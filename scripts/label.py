"""Tiny in-browser annotator. Run: streamlit run scripts/label.py --server.port 8502

Reads images from a queue (label_queue/{kubiaka,other_kamikiri}/),
shows them one at a time, lets you draw rectangles, writes YOLO-format
labels to dataset/{images,labels}/train/  (no val split — we'll move some
to val at the end).

Controls:
  - Draw rectangles by clicking and dragging
  - Class is set by the source folder (kubiaka/other_kamikiri)
  - 「保存して次へ」 saves YOLO label and goes to next image
  - 「スキップ」 skips (no label written) and goes to next
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# streamlit-drawable-canvas 0.9.3 calls streamlit.elements.image.image_to_url,
# which was removed in streamlit >= 1.40. Shim it using the new media file manager
# (which produces short /media/<hash>.png URLs the canvas component can load).
import streamlit.elements.image as _st_image
if not hasattr(_st_image, "image_to_url"):
    from streamlit.runtime import get_instance, exists as _runtime_exists

    def image_to_url(image, width=None, clamp=False, channels="RGB",
                     output_format="auto", image_id="", allow_emoji=False):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        buf = io.BytesIO()
        fmt = "PNG" if output_format.lower() in {"auto", "png"} else output_format.upper()
        image.save(buf, format=fmt)
        data = buf.getvalue()
        mime = f"image/{fmt.lower()}"
        if _runtime_exists():
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                ctx = get_script_run_ctx()
                session_id = ctx.session_id if ctx else "main"
                mfm = get_instance().media_file_mgr
                file_id = image_id or f"img_{hash(data)}"
                url = mfm.add(data, mime, session_id, file_name=f"{file_id}.{fmt.lower()}")
                return url
            except Exception:
                pass
        # Fallback to data URL
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    _st_image.image_to_url = image_to_url

from streamlit_drawable_canvas import st_canvas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUEUE = PROJECT_ROOT / "label_queue"
DATASET = PROJECT_ROOT / "dataset"
IMG_OUT = DATASET / "images" / "train"
LBL_OUT = DATASET / "labels" / "train"

CLASS_MAP = {"kubiaka": 0, "other_kamikiri": 1}
CLASS_JP = {"kubiaka": "クビアカ", "other_kamikiri": "他のカミキリ (対象外)"}
CLASS_COLOR = {"kubiaka": "#ff0000", "other_kamikiri": "#0066ff"}

MAX_DISPLAY_W = 900


def build_queue() -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for folder, cid in CLASS_MAP.items():
        d = QUEUE / folder
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((p, cid))
    return items


def already_labeled(stem: str) -> bool:
    return (LBL_OUT / f"{stem}.txt").exists()


def main() -> None:
    st.set_page_config(page_title="ラベリングツール", layout="wide")
    st.title("ラベリングツール")

    IMG_OUT.mkdir(parents=True, exist_ok=True)
    LBL_OUT.mkdir(parents=True, exist_ok=True)

    if "queue" not in st.session_state:
        st.session_state.queue = build_queue()
        st.session_state.idx = 0

    queue = st.session_state.queue
    if not queue:
        st.error(
            f"`label_queue/` にラベル候補がありません。\n"
            f"`{QUEUE}` に kubiaka/ と other_kamikiri/ サブフォルダを作成し、画像を配置してください。"
        )
        return

    # Skip already labeled
    while st.session_state.idx < len(queue) and already_labeled(queue[st.session_state.idx][0].stem):
        st.session_state.idx += 1

    if st.session_state.idx >= len(queue):
        n = sum(1 for p in LBL_OUT.glob("*.txt") if p.stat().st_size > 0)
        st.success(f"完了! {n} 枚のラベル付き画像が dataset/ に保存されました。")
        st.write("次のステップ:  ターミナルで")
        st.code("python scripts/train.py --dataset dataset --epochs 100 --imgsz 640 --batch 8", language="bash")
        return

    img_path, class_id = queue[st.session_state.idx]
    class_name = [k for k, v in CLASS_MAP.items() if v == class_id][0]

    # Progress
    done = st.session_state.idx
    total = len(queue)
    st.progress(done / total, text=f"{done}/{total}  ({img_path.name})")

    # Class info
    st.markdown(
        f"### 現在のクラス: **{CLASS_JP[class_name]}** "
        f"<span style='color:{CLASS_COLOR[class_name]}'>■</span>",
        unsafe_allow_html=True,
    )
    st.caption("画像上で虫の周りにマウスドラッグで四角を描いてください。複数描いてもOK。")

    # Load image and downscale for display
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = min(MAX_DISPLAY_W / orig_w, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)
    disp_img = img.resize((disp_w, disp_h))

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.15)",
        stroke_width=3,
        stroke_color=CLASS_COLOR[class_name],
        background_image=disp_img,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key=f"canvas_{st.session_state.idx}",
    )

    c1, c2, c3 = st.columns([1, 1, 4])
    save_btn = c1.button("保存して次へ ▶", type="primary")
    skip_btn = c2.button("スキップ")
    c3.write("")

    if skip_btn:
        st.session_state.idx += 1
        st.rerun()

    if save_btn:
        rects = []
        if canvas_result.json_data is not None:
            for obj in canvas_result.json_data.get("objects", []):
                if obj.get("type") != "rect":
                    continue
                # streamlit-drawable-canvas returns left/top/width/height in display coords
                left = obj["left"]
                top = obj["top"]
                w = obj["width"] * obj.get("scaleX", 1)
                h = obj["height"] * obj.get("scaleY", 1)
                # Convert to YOLO normalized (using original image dims)
                cx = (left + w / 2) / disp_w
                cy = (top + h / 2) / disp_h
                bw = w / disp_w
                bh = h / disp_h
                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                if bw > 0.01 and bh > 0.01:
                    rects.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not rects:
            st.warning("ボックスが描かれていません。スキップしますか? それとも描いて保存しますか?")
        else:
            # Copy image to dataset/images/train/
            out_img_path = IMG_OUT / img_path.name
            out_img_path.write_bytes(img_path.read_bytes())
            # Write label
            out_lbl_path = LBL_OUT / f"{img_path.stem}.txt"
            out_lbl_path.write_text("\n".join(rects), encoding="utf-8")
            st.session_state.idx += 1
            st.rerun()


if __name__ == "__main__":
    main()
