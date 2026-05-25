# Model weights

Place YOLOv8 `.pt` files here.

## Expected files

- `robogate_v1.pt` — fine-tuned weights for カミキリムシ / カブトムシ. Primary model. Path is set by `DEFAULT_MODEL_PATH` in [`config.py`](../config.py).
- `yolov8n.pt` — COCO pretrained fallback. Auto-downloaded by `ultralytics` on first use if missing.

## Training (Option A)

Fine-tune via Google Colab using the `ultralytics` library on Shimamoto's labeled dataset (50–100 images per class is enough for the demo). Export the resulting `best.pt` and copy it here as `robogate_v1.pt`.

## Fallback (Option B)

If a fine-tuned model is not ready, download a community insect/beetle YOLOv8 model from Roboflow Universe and drop it in. Class names may be generic — UI label mapping is in `CLASS_LABELS_JP` / `CLASS_LABELS_EN` in [`config.py`](../config.py).

`.pt` files are gitignored — do not commit weights.
