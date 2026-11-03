#!/usr/bin/env bash
# Скачивает ONNX-веса MaST (ECCV 2026, github.com/TsingWei/MaST) в models/mast/.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
git clone --depth 1 https://github.com/TsingWei/MaST.git "$tmp/MaST"
mkdir -p models/mast
cp "$tmp"/MaST/models/*.onnx models/mast/
rm -rf "$tmp"
ls -la models/mast
