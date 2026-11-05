"""Модель объекта: габариты и точка привязки GT.

GT-точка в сцене — не центр фургона, а его **передний нижний центр**
(проверено наложением 3D-бокса на кадры CAM4/CAM5: при смещении
(−11 м по курсу, +2 м вверх) от центра бокс ложится на кузов).

Габариты подобраны по рендерам: седельный тягач с прицепом,
≈20 м длина, 2.6 м ширина, 4.0 м высота.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import box_corners, heading_from_velocity


@dataclass(frozen=True)
class ObjectModel:
    length_cm: float = 2000.0
    width_cm: float = 260.0
    height_cm: float = 400.0
    # положение точки привязки (GT) в локальной системе объекта:
    # (вперёд, вправо, вверх) от центра бокса
    anchor_cm: tuple[float, float, float] = (1000.0, 0.0, -200.0)

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.length_cm, self.width_cm, self.height_cm)

    def center(self, anchor_pos: np.ndarray, heading_deg: float) -> np.ndarray:
        """Центр бокса по точке привязки и курсу."""
        y = np.radians(heading_deg)
        fwd = np.array([np.cos(y), np.sin(y), 0.0])
        right = np.array([-np.sin(y), np.cos(y), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        a = self.anchor_cm
        return anchor_pos - a[0] * fwd - a[1] * right - a[2] * up

    def corners(self, anchor_pos: np.ndarray, heading_deg: float) -> np.ndarray:
        return box_corners(self.center(anchor_pos, heading_deg), heading_deg, self.size)

    def box_2d(self, camera, anchor_pos: np.ndarray, heading_deg: float):
        return camera.project_box(self.corners(anchor_pos, heading_deg))


def heading_from_positions(track: dict[int, np.ndarray], frame: int, span: int = 2) -> float:
    """Курс по соседним кадрам траектории; fallback — на ближайший кадр."""
    lo = frame - span
    hi = frame + span
    if lo in track and hi in track:
        return heading_from_velocity(track[hi] - track[lo])
    for d in range(1, span + 3):
        if lo in track and frame in track:
            return heading_from_velocity(track[frame] - track[lo])
        if hi in track and frame in track:
            return heading_from_velocity(track[hi] - track[frame])
        lo -= 1
        hi += 1
    return 0.0
