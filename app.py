from __future__ import annotations

import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from detector import Detector
from video_processor import process_video

st.set_page_config(page_title="ロボゲート ── 害虫検知デモ", layout="wide")


@st.cache_resource(show_spinner="モデルを読み込み中...")
def load_detector() -> Detector:
    model_path = config.DEFAULT_MODEL_PATH
    if model_path.exists():
        model_arg = str(model_path)
    else:
        # ultralytics auto-downloads yolov8n.pt as fallback
        model_arg = "yolov8n.pt"
    return Detector(model_arg)


def label_jp(class_name: str) -> str:
    return config.CLASS_LABELS_JP.get(class_name, config.DEFAULT_LABEL_JP)


def detections_to_df(detections: list[dict]) -> pd.DataFrame:
    rows = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        rows.append({
            "クラス": label_jp(d["class_name"]),
            "信頼度": round(d["confidence"], 3),
            "始点X": round(x1, 1),
            "始点Y": round(y1, 1),
            "終点X": round(x2, 1),
            "終点Y": round(y2, 1),
        })
    return pd.DataFrame(rows)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pick_random_asset(directory: Path, extensions: set[str]) -> Path | None:
    if not directory.exists():
        return None
    files = [p for p in directory.iterdir() if p.suffix.lower() in extensions]
    if not files:
        return None
    return random.choice(files)


def run_photo_tab(detector: Detector) -> None:
    st.subheader("フォト検知")
    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "画像をアップロード",
            type=["jpg", "jpeg", "png"],
            key="photo_upload",
        )
    with col_sample:
        st.write("")
        st.write("")
        sample_clicked = st.button(
            "サンプル画像で試す", use_container_width=True, key="photo_sample"
        )

    image_bgr: np.ndarray | None = None
    if uploaded is not None:
        pil = Image.open(uploaded)
        image_bgr = pil_to_bgr(pil)
    elif sample_clicked:
        sample = pick_random_asset(config.IMAGES_DIR, {".jpg", ".jpeg", ".png"})
        if sample is None:
            st.warning("assets/images/ にサンプル画像がありません。")
        else:
            image_bgr = cv2.imread(str(sample))
            st.caption(f"サンプル: {sample.name}")

    if image_bgr is None:
        st.info("画像をアップロードするか、サンプル画像ボタンを押してください。")
        return

    detections = detector.detect(image_bgr)
    annotated_bgr = detector.annotate(image_bgr, detections)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**元画像**")
        st.image(bgr_to_rgb(image_bgr), use_container_width=True)
    with c2:
        st.markdown("**検知結果**")
        st.image(bgr_to_rgb(annotated_bgr), use_container_width=True)

    st.markdown("### 検知一覧")
    if detections:
        st.dataframe(detections_to_df(detections), use_container_width=True)
    else:
        st.info("検知対象は見つかりませんでした。")


def run_video_tab(detector: Detector, frame_skip: int) -> None:
    st.subheader("ビデオ検知")
    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "動画をアップロード", type=["mp4", "mov"], key="video_upload"
        )
    with col_sample:
        st.write("")
        st.write("")
        sample_clicked = st.button(
            "サンプル動画で試す", use_container_width=True, key="video_sample"
        )

    source_path: Path | None = None
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.close()
        source_path = Path(tmp.name)
    elif sample_clicked:
        sample = pick_random_asset(config.VIDEOS_DIR, {".mp4", ".mov"})
        if sample is None:
            st.warning("assets/videos/ にサンプル動画がありません。")
        else:
            source_path = sample
            st.caption(f"サンプル: {sample.name}")

    if source_path is None:
        st.info("動画をアップロードするか、サンプル動画ボタンを押してください。")
        return

    if st.button("解析開始", type="primary", key="video_run"):
        output_path = Path(tempfile.gettempdir()) / f"robogate_output_{source_path.stem}.mp4"
        progress = st.progress(0.0, text="解析中...")

        def update(p: float) -> None:
            progress.progress(min(p, 1.0), text=f"解析中... {int(p * 100)}%")

        stats = process_video(
            source_path,
            output_path,
            detector,
            frame_skip=frame_skip,
            progress_callback=update,
        )
        progress.empty()

        st.success("解析完了")
        st.video(str(output_path))

        c1, c2, c3 = st.columns(3)
        c1.metric("解析フレーム数", stats["frames_analyzed"])
        c2.metric("検知数合計", stats["total_detections"])
        c3.metric("平均信頼度", f"{stats['average_confidence']:.3f}")

        st.markdown("### タイムライン (秒ごとの検知数)")
        if stats["per_second_counts"]:
            df = pd.DataFrame(
                sorted(stats["per_second_counts"].items()),
                columns=["秒", "検知数"],
            ).set_index("秒")
            st.bar_chart(df)
        else:
            st.info("検知対象は見つかりませんでした。")


def run_roadmap_tab() -> None:
    st.subheader("ロードマップ")
    st.markdown(
        """
| フェーズ | 内容 | 状態 |
| --- | --- | --- |
| **フェーズ 1** | 写真からの検知 | ✅ **このデモ** |
| **フェーズ 2** | 動画からの検知 | ✅ **このデモ** |
| **フェーズ 3** | ドローン映像のストリーミング解析 | 🔜 次フェーズ |
| **フェーズ 4** | 四足歩行ロボット (ユニツリー Go2) 搭載 | 🔜 次フェーズ |
"""
    )

    st.divider()
    st.markdown("### なぜ重要か")

    c1, c2, c3 = st.columns(3)
    c1.metric("和歌山県の梅生産シェア", "60%", help="全国の梅生産に占める和歌山県の割合")
    c2.metric("みなべ町・梅林面積", "2,000 ha", help="次の標的として警戒中")
    c3.metric("被害本数 (日高川町, 4年間)", "3,500 本", help="天敵がいないため一度広がると止まらない")

    st.markdown(
        """
クビアカツヤカミキリは、梅・桃・桜などバラ科の果樹を内部から食い荒らす特定外来生物です。
天敵がいないため一度広がると止まらず、和歌山県では日高川町まで4年余りで3,500本の被害が発生。
**和歌山の梅生産（全国の60%）の中心地・みなべ町が次の標的**として警戒されています。
"""
    )

    st.divider()
    st.markdown("### 既に動いている行政予算")

    c1, c2 = st.columns(2)
    c1.success("**¥10,000 / 件**\n\nフラス発見者への懸賞金 (みなべ町農業士会, 2024年4月〜)")
    c2.success("**¥60,000 / 本**\n\n被害木伐採への県・町支援金")

    st.markdown(
        "RoboGate はこの発見プロセスをロボット × AI で自動化します。"
        "懸賞金 ¥10,000 × 検知数 のスケールでマネタイズ可能な、行政予算 PMF の確立済みドメインです。"
    )


def main() -> None:
    st.title("ロボゲート ── 害虫検知デモ")

    with st.sidebar:
        st.header("設定")
        conf_threshold = st.slider(
            "信頼度しきい値", 0.0, 1.0, config.DEFAULT_CONFIDENCE, 0.05
        )
        st.divider()
        st.subheader("ビデオ解析")
        frame_skip = st.slider("フレームスキップ", 1, 10, config.DEFAULT_FRAME_SKIP)
        st.caption(f"{frame_skip}フレームに1回検知を実行")

    detector = load_detector()
    detector.conf_threshold = conf_threshold

    tab1, tab2, tab3 = st.tabs(["フォト検知", "ビデオ検知", "ロードマップ"])
    with tab1:
        run_photo_tab(detector)
    with tab2:
        run_video_tab(detector, frame_skip)
    with tab3:
        run_roadmap_tab()


if __name__ == "__main__":
    main()
