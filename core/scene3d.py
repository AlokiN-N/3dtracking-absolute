"""Отрисовка 3D-вида (аксонометрия) для панели валидации.

Своя минимальная проекция вместо matplotlib: рендер кадра занимает
доли миллисекунды, а matplotlib на 1000 кадров был бы непозволительно медленным.
"""

from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera

GREEN = (80, 220, 80)
GREEN_DIM = (45, 105, 45)
RED = (70, 70, 255)
RED_DIM = (55, 40, 110)
GREY = (150, 150, 150)
GREY_DIM = (62, 62, 66)
WHITE = (235, 235, 235)
BG = (26, 26, 30)


class Scene3D:
    """Аксонометрический рендер сцены с фиксированным кадром по данным."""

    def __init__(
        self,
        gt_track: dict[int, np.ndarray],
        pred_track: dict[int, np.ndarray],
        cameras: list[Camera],
        size: tuple[int, int] = (720, 660),
        az_deg: float = 38.0,
        el_deg: float = 26.0,
        margin: int = 34,
        grid_m: float = 100.0,
    ) -> None:
        self.w, self.h = size
        self.cameras = cameras
        self._gt = gt_track
        self._pred = pred_track
        self.az = np.radians(az_deg)
        self.el = np.radians(el_deg)
        self.grid_m = grid_m

        pts = [np.asarray(p, dtype=np.float64) for p in gt_track.values()]
        pts += [np.asarray(p, dtype=np.float64) for p in pred_track.values() if p is not None]
        pts += [c.loc for c in cameras]
        P = np.stack(pts)

        self.right = np.array([-np.sin(self.az), np.cos(self.az), 0.0])
        self.up = np.array(
            [-np.cos(self.az) * np.sin(self.el), -np.sin(self.az) * np.sin(self.el), np.cos(self.el)]
        )

        self.zfloor = float(P[:, 2].min() - 200.0)
        self.center = np.array([P[:, 0].mean(), P[:, 1].mean(), self.zfloor])
        self.xy_min = P[:, :2].min(axis=0) - 15000.0
        self.xy_max = P[:, :2].max(axis=0) + 15000.0

        rel = P - self.center
        u = rel @ self.right
        v = -(rel @ self.up)
        du = max(u.max() - u.min(), 1.0)
        dv = max(v.max() - v.min(), 1.0)
        self.scale = min((self.w - 2 * margin) / du, (self.h - 2 * margin) / dv)
        self.ox = self.w / 2 - (u.min() + u.max()) / 2 * self.scale
        self.oy = self.h / 2 - (v.min() + v.max()) / 2 * self.scale

        self._bg = self._build_background()

    # ------------------------------------------------------------ проекция
    def project(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        rel = pts - self.center
        u = (rel @ self.right) * self.scale + self.ox
        v = -(rel @ self.up) * self.scale + self.oy
        return np.stack([u, v], axis=1)

    def _poly(self, track: dict[int, np.ndarray], frames: list[int]) -> np.ndarray:
        return self.project(np.stack([track[f] for f in frames])).astype(np.int32)

    # -------------------------------------------------------- статический фон
    def _build_background(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), BG, dtype=np.uint8)
        z = self.zfloor
        step = self.grid_m * 100.0
        for x in np.arange(np.floor(self.xy_min[0] / step) * step, self.xy_max[0] + step, step):
            p = self.project(np.array([[x, self.xy_min[1], z], [x, self.xy_max[1], z]]))
            cv2.line(img, tuple(np.int32(p[0])), tuple(np.int32(p[1])), GREY_DIM, 1)
        for y in np.arange(np.floor(self.xy_min[1] / step) * step, self.xy_max[1] + step, step):
            p = self.project(np.array([[self.xy_min[0], y, z], [self.xy_max[0], y, z]]))
            cv2.line(img, tuple(np.int32(p[0])), tuple(np.int32(p[1])), GREY_DIM, 1)

        for cam in self.cameras:
            pt = self.project(cam.loc)[0]
            yaw = float(cam.rot[2])
            fov = 2 * np.degrees(np.arctan(cam.sensor_mm[0] / (2 * cam.focal_mm)))
            wedge = [cam.loc]
            for a in (yaw - fov / 2, yaw + fov / 2):
                r = np.radians(a)
                wedge.append(
                    np.array([cam.loc[0] + 20000 * np.cos(r), cam.loc[1] + 20000 * np.sin(r), z])
                )
            scr = self.project(np.stack(wedge)).astype(np.int32)
            cv2.line(img, tuple(scr[0]), tuple(scr[1]), (70, 70, 74), 1, cv2.LINE_AA)
            cv2.line(img, tuple(scr[0]), tuple(scr[2]), (70, 70, 74), 1, cv2.LINE_AA)
            cv2.rectangle(img, (int(pt[0]) - 4, int(pt[1]) - 4), (int(pt[0]) + 4, int(pt[1]) + 4), (200, 200, 120), -1)
            cv2.putText(img, cam.name, (int(pt[0]) + 7, int(pt[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 120), 1, cv2.LINE_AA)

        frames = [f for f in sorted(self._gt) if self._gt[f] is not None]
        if len(frames) > 1:
            cv2.polylines(img, [self._poly(self._gt, frames)], False, GREEN_DIM, 2, cv2.LINE_AA)
        frames = [f for f in sorted(self._pred) if self._pred.get(f) is not None]
        if len(frames) > 1:
            cv2.polylines(img, [self._poly(self._pred, frames)], False, RED_DIM, 2, cv2.LINE_AA)

        cv2.putText(img, "3D VIEW", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, "GT", (110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
        cv2.putText(img, "pred", (150, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
        cv2.putText(img, f"grid {int(self.grid_m)} m", (self.w - 110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
        return img

    # ---------------------------------------------------------------- кадр
    def render(self, frame: int, gt_pt: np.ndarray | None, pred_pt: np.ndarray | None) -> np.ndarray:
        img = self._bg.copy()

        past = [f for f in sorted(self._gt) if f <= frame and self._gt[f] is not None]
        if len(past) > 1:
            cv2.polylines(img, [self._poly(self._gt, past)], False, GREEN, 3, cv2.LINE_AA)
        past = [f for f in sorted(self._pred) if f <= frame and self._pred.get(f) is not None]
        if len(past) > 1:
            # предсказание тонким красным поверх толстого GT: видно оба
            cv2.polylines(img, [self._poly(self._pred, past)], False, RED, 1, cv2.LINE_AA)

        if gt_pt is not None:
            g = self.project(gt_pt)[0].astype(int)
            if pred_pt is not None:
                p = self.project(pred_pt)[0].astype(int)
                cv2.line(img, tuple(g), tuple(p), (210, 210, 210), 1, cv2.LINE_AA)
                cv2.circle(img, tuple(p), 6, RED, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(p), 6, WHITE, 1, cv2.LINE_AA)
            cv2.circle(img, tuple(g), 6, GREEN, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(g), 6, WHITE, 1, cv2.LINE_AA)
        return img
