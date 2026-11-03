"""Целеуказание: рамка объекта на первом кадре -> init.json для e02_mast_ekf.

Оператор (интерактивно):
    python -m tools.designate --data data --cams 1 5 --out init.json
    # в окне каждой камеры обвести объект мышью, Enter — подтвердить, c — пропустить камеру

Эмуляция оператора для сцены с GT (проверка пайплайна на Way1):
    python -m tools.designate --data data --from-gt --out init.json
    # рамка = проекция 3D-бокса ObjectModel на кадре 0 в камерах, где объект целиком в кадре
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.camera import Camera  # noqa: E402
from core.dataset import load_scene  # noqa: E402
from core.object_model import ObjectModel, heading_from_positions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="init.json")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--cams", type=int, nargs="*", default=None)
    ap.add_argument("--from-gt", action="store_true", help="эмуляция оператора по GT (только для проверки)")
    a = ap.parse_args()
    scene = load_scene(a.data)
    cams = a.cams or sorted(scene.cameras)
    out = {}
    if a.from_gt:
        om = ObjectModel(); gt = {f: p.loc for f, p in scene.target.items()}
        hd = heading_from_positions(gt, a.frame, span=10)
        for i in cams:
            cam = Camera.from_track(scene.cameras[i], a.frame, scene.width, scene.height)
            b = om.box_2d(cam, gt[a.frame], hd)
            if b is None:
                continue
            x0, y0, x1, y1 = b
            if x0 < 5 or y0 < 5 or x1 > scene.width - 5 or y1 > scene.height - 5:
                continue  # объект виден не целиком — оператор такую камеру не размечает
            out[str(i)] = [a.frame, float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    else:
        for i in cams:
            cap = cv2.VideoCapture(str(scene.cameras[i].video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
            ok, img = cap.read()
            if not ok:
                continue
            x, y, w, h = cv2.selectROI(f"CAM{i}: обведите объект (Enter), c - пропустить", img, showCrosshair=True)
            cv2.destroyAllWindows()
            if w > 0 and h > 0:
                out[str(i)] = [a.frame, float(x), float(y), float(w), float(h)]
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{a.out}: {out}")
    return 0 if len(out) >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
