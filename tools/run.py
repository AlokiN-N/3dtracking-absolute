"""Боевой запуск без GT: видео + камеры -> CSV с 3D-траекторией в формате организаторов.

    python -m tools.run --data <папка> --init init.json --out result.csv
    python -m tools.run --data <папка> --exp e01_bgsub_ekf --out result.csv   # без целеуказания

Выход: `Frame;Loc_X;Loc_Y;Loc_Z` (см, как object_transform_*.csv). Кадры без оценки — пустые.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dataset import load_scene  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--exp", default="e02_mast_ekf", choices=sorted(EXPERIMENTS))
    ap.add_argument("--init", default=None, help="целеуказание (tools/designate.py), нужно для e02")
    ap.add_argument("--model", default="MaST_tiny")
    ap.add_argument("--out", default="result.csv")
    a = ap.parse_args()

    scene = load_scene(a.data, require_gt=False)
    n = max(len(t.poses) for t in scene.cameras.values())
    cls = EXPERIMENTS[a.exp]
    params = inspect.signature(cls.__init__).parameters
    kw = {k: v for k, v in dict(init=a.init, model=a.model).items()
          if k in params or any(p.kind == p.VAR_KEYWORD for p in params.values())}
    pred = cls(**kw).run(scene)
    last = max(list(pred.xyz) + [n - 1])
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
        for f in range(last + 1):
            p = pred.xyz.get(f)
            fh.write(f"{f};;;\n" if p is None else f"{f};{p[0]:.4f};{p[1]:.4f};{p[2]:.4f}\n")
    print(f"{a.out}: {len(pred.xyz)} кадров с оценкой из {last + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
