"""Метрики качества 3D-трека: ошибки по кадрам и сводка."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ErrorTable:
    frames: list[int] = field(default_factory=list)
    gt: list[np.ndarray] = field(default_factory=list)
    pred: list[np.ndarray] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)  # метры
    n_cams: list[int] = field(default_factory=list)

    def add(self, frame: int, gt: np.ndarray, pred: np.ndarray | None, n_cams: int = 0) -> None:
        self.frames.append(frame)
        self.gt.append(np.asarray(gt, dtype=np.float64))
        if pred is None:
            self.pred.append(np.full(3, np.nan))
            self.errors.append(float("nan"))
        else:
            pred = np.asarray(pred, dtype=np.float64)
            self.pred.append(pred)
            self.errors.append(float(np.linalg.norm(pred - gt)) / 100.0)
        self.n_cams.append(n_cams)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(np.array(self.errors))

    def summary(self) -> dict:
        err = np.array(self.errors)
        ok = np.isfinite(err)
        if not ok.any():
            return {"n": 0}
        e = err[ok]
        return {
            "n": int(ok.sum()),
            "coverage": float(ok.mean()),
            "rmse_m": float(np.sqrt(np.mean(e**2))),
            "mae_m": float(np.mean(e)),
            "median_m": float(np.median(e)),
            "p95_m": float(np.percentile(e, 95)),
            "max_m": float(e.max()),
        }

    def as_rows(self) -> list[dict]:
        rows = []
        for i, f in enumerate(self.frames):
            rows.append(
                {
                    "frame": f,
                    "gt_x": self.gt[i][0],
                    "gt_y": self.gt[i][1],
                    "gt_z": self.gt[i][2],
                    "pred_x": self.pred[i][0],
                    "pred_y": self.pred[i][1],
                    "pred_z": self.pred[i][2],
                    "err_m": self.errors[i],
                    "n_cams": self.n_cams[i],
                }
            )
        return rows


def build_errors(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray], n_cams: dict[int, int] | None = None) -> ErrorTable:
    table = ErrorTable()
    for frame in sorted(gt):
        table.add(frame, gt[frame], pred.get(frame), (n_cams or {}).get(frame, 0))
    return table
