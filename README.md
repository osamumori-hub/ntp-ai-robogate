# RoboGate Demo — 害虫検知デモ

NTP の RoboGate 異常判定エンジン向け VC ピッチ用デモです。  
写真および動画から「クビアカツヤカミキリ（Red-Necked Longhorn Beetle）」を検知します。

> 本プロジェクトはデモ用途であり、本番環境向けコードではありません。

## セットアップ

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt