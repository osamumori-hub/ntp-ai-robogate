# モデルウェイト

YOLOv8 の `.pt` ファイルをこのディレクトリに配置してください。

## 想定ファイル

- `robogate_v1.pt`  
  カミキリムシ / カブトムシ向けにファインチューニングされたメインモデルです。  
  パスは [`config.py`](../config.py) 内の `DEFAULT_MODEL_PATH` で設定されています。

- `yolov8n.pt`  
  COCO 学習済みのフォールバックモデルです。  
  ファイルが存在しない場合、初回使用時に `ultralytics` により自動ダウンロードされます。

## 学習方法（Option A）

Google Colab 上で `ultralytics` ライブラリを使用し、  
Shimamoto のラベル付きデータセットを用いてファインチューニングを行います。

デモ用途であれば、各クラス 50〜100 枚程度の画像でも十分です。

学習後に生成された `best.pt` をエクスポートし、  
このディレクトリへ `robogate_v1.pt` として配置してください。

## フォールバック（Option B）

ファインチューニング済みモデルが未完成の場合は、  
Roboflow Universe 上の昆虫 / カブトムシ向け YOLOv8 コミュニティモデルを利用できます。

クラス名が汎用的な場合がありますが、  
UI 表示用のラベルマッピングは [`config.py`](../config.py) 内の  
`CLASS_LABELS_JP` および `CLASS_LABELS_EN` で管理されています。

## 注意事項

`.pt` ファイルは `.gitignore` に含まれています。  
モデルウェイトを Git にコミットしないでください。