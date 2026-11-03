"""Конфигурации MaST и путь к ONNX-весам.

Веса лежат в репозитории авторов (github.com/TsingWei/MaST, папка models/).
Скачать: `bash tools/get_mast.sh` -> models/mast/MaST-{nano,tiny,small}.onnx
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "MaST_small": dict(search_size=384, template_size=192, search_factor=5.0, template_factor=2.0, stride=16, use_reparam=True),
    "MaST_tiny": dict(search_size=256, template_size=128, search_factor=4.0, template_factor=2.0, stride=16, use_reparam=True),
    "MaST_nano": dict(search_size=256, template_size=128, search_factor=4.0, template_factor=2.0, stride=16, use_reparam=True),
}


def model_cfg(name: str = "MaST_tiny") -> dict:
    cfg = dict(MODELS[name])
    d = Path(os.environ.get("MAST_DIR", ROOT / "models" / "mast"))
    cfg["onnx"] = str(d / f"{name.replace('_', '-')}.onnx")
    if not Path(cfg["onnx"]).is_file():
        raise FileNotFoundError(f"нет весов {cfg['onnx']}: запустите `bash tools/get_mast.sh`")
    return cfg
