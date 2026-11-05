#!/usr/bin/env bash
# Скачивает YOLO11n (ONNX, COCO) в models/yolo/ (подсказки для SAM 2 в experiments/sam2_model).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/yolo
curl -L -o models/yolo/yolo11n.onnx https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
ls -la models/yolo
