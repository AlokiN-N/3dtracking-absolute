"""E00 — oracle-триангуляция: проверка геометрии без детектора.

Берём GT-точку, проецируем её в каждую камеру (это «идеальная детекция»),
добавляем шум в пикселях, обратно строим лучи и триангулируем.
Если этот эксперимент даёт большую ошибку — виновата калибровка,
а не модель детекции. Это «потолок» для всех остальных подходов.
"""

from __future__ import annotations

import numpy as np

from core.camera import Camera
from core.dataset import Scene
from core.geometry import triangulate
from core.validation import Prediction
from experiments.base import Experiment


class OracleTriangulation(Experiment):
    name = "e00_oracle"

    def __init__(
        self,
        sigma_px: float = 1.5,
        min_cameras: int = 2,
        seed: int = 0,
    ) -> None:
        self.sigma_px = float(sigma_px)
        self.min_cameras = int(min_cameras)
        self.seed = int(seed)

    def run(self, scene: Scene) -> Prediction:
        rng = np.random.default_rng(self.seed)
        xyz: dict[int, np.ndarray] = {}
        n_cams: dict[int, int] = {}
        used: list[int] = []

        for frame in sorted(scene.target):
            target = scene.target[frame].loc
            origins, directions, sigmas = [], [], []
            for idx in sorted(scene.cameras):
                cam = Camera.from_track(scene.cameras[idx], frame, scene.width, scene.height)
                uv, depth = cam.project(target)
                if not cam.in_image(uv, depth)[0]:
                    continue
                uv = uv[0]
                if self.sigma_px > 0:
                    uv = uv + rng.normal(0.0, self.sigma_px, size=2)
                origin, direction = cam.ray(uv)
                origins.append(origin)
                directions.append(direction)
                sigmas.append(cam.angular_sigma(max(self.sigma_px, 1e-6)))

            if len(origins) < self.min_cameras:
                continue
            point, _ = triangulate(
                np.stack(origins), np.stack(directions), np.array(sigmas)
            )
            xyz[frame] = point
            n_cams[frame] = len(origins)
            used.append(len(origins))

        return Prediction(
            name=self.name,
            xyz=xyz,
            n_cams=n_cams,
            meta={
                "sigma_px": self.sigma_px,
                "min_cameras": self.min_cameras,
                "mean_cameras_used": float(np.mean(used)) if used else 0.0,
                "frames": len(xyz),
            },
        )
