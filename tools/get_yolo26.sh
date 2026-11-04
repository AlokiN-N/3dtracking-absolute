#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
curl --fail --location --retry 2 -o models/yolo26n-seg.onnx.tmp \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.onnx
mv models/yolo26n-seg.onnx.tmp models/yolo26n-seg.onnx
