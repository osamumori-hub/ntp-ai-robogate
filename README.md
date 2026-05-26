# RoboGate デモ — 害虫検知デモ

NTPのRoboGate異常判定エンジンのVCピッチ用デモです。写真および動画からクビアカツヤカミキリを検知します。

> これはデモであり、本番コードではありません。

## セットアップ

```bash
python -m venv .venv
# Windows の場合:  .venv\Scripts\activate
# macOS / Linux の場合:  source .venv/bin/activate
pip install -r requirements.txt
```

## モデルの重み

YOLOv8 の `.pt` ファイルを `models/robogate_v1.pt` に配置してください。ファインチューニング済みの重みがまだ用意できていない場合、アプリは `models/yolov8n.pt`（COCO 事前学習済み。初回実行時に ultralytics が自動ダウンロード）にフォールバックします。詳細は [models/README.md](models/README.md) を参照してください。

## サンプルデータ

テスト用画像（JPG/PNG）を `assets/images/` に、テスト用動画（MP4/MOV）を `assets/videos/` に配置してください。「サンプル画像で試す」/「サンプル動画で試す」ボタンを押すと、ランダムに1つが選ばれます。

## 実行方法

```bash
streamlit run app.py
```

## タブ構成

1. **フォト検知** — 写真をアップロードすると、バウンディングボックスと検知結果テーブルが表示されます。
2. **ビデオ検知** — 動画をアップロードすると、フレームに注釈が付いた出力に加え、タイムラインと統計情報が得られます。
3. **ロードマップ** — フェーズ 1〜4 のビジョンスライド。
