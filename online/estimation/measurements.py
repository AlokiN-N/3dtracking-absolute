"""Shared robust 6-DoF fit for visible CAD-to-image correspondences."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from online.geometry import Rot, center_of, pose_angles, skew
from online.estimation.surface_pose import spatial_partition


@dataclass
class Correspondences:
    view: object
    points: np.ndarray
    pixels: np.ndarray
    confidence: np.ndarray

    def filtered(self):
        good = np.isfinite(self.points).all(1) & np.isfinite(self.pixels).all(1)
        good &= np.isfinite(self.confidence) & (self.confidence > 0)
        ids = np.flatnonzero(good)
        if len(ids) > 256:
            ids = ids[np.linspace(0, len(ids) - 1, 256, dtype=int)]
        return Correspondences(self.view, self.points[ids], self.pixels[ids], self.confidence[ids])


@dataclass
class PoseMeasurement:
    pose: np.ndarray
    covariance: np.ndarray
    cameras: tuple
    reprojection_px: float
    source: str = ""

    def reference_variance(self, model):
        offset = Rot(*self.pose[3:]) @ model.center
        jacobian = np.hstack([np.eye(3), skew(offset)])
        return float(np.trace(jacobian @ self.covariance @ jacobian.T))


def fit_measurement(initial, observations, model):
    observations = [group.filtered() for group in observations]
    observations = [group for group in observations if len(group.points) >= 16]
    if not observations:
        return None
    partitions = [spatial_partition(group.pixels, group.view.instance.box) for group in observations]
    if any(min(part.sum(), (~part).sum()) < 6 for part in partitions):
        return None
    rotation = Rot(*initial[3:])
    center = center_of(initial, model)

    def transform(delta):
        matrix = Rotation.from_rotvec(delta[3:]).as_matrix() @ rotation
        return matrix, center + delta[:3] - matrix @ model.center

    def residual(delta, validation=False):
        matrix, origin = transform(delta)
        values = []
        for group, partition in zip(observations, partitions):
            use = partition if validation else ~partition
            uv, depth = group.view.project(group.points[use] @ matrix.T + origin)
            error = uv - group.pixels[use]
            error[depth <= 0.05] = 1000
            values.append(error * np.sqrt(np.clip(group.confidence[use, None], 0.05, 1.0)))
        return np.concatenate(values).ravel()

    scale = [max(model.size * 0.1, 0.01)] * 3 + [0.1] * 3
    fit = least_squares(residual, np.zeros(6), loss="soft_l1", f_scale=2,
                        x_scale=scale, max_nfev=30)
    matrix, origin = transform(fit.x)
    if np.linalg.norm(fit.x[:3]) > model.size or np.linalg.norm(fit.x[3:]) > 0.65:
        return None
    before = np.linalg.norm(residual(np.zeros(6), True).reshape(-1, 2), axis=1)
    after = np.linalg.norm(residual(fit.x, True).reshape(-1, 2), axis=1)
    if np.median(after) > 3 or np.median(before) - np.median(after) < 0.5:
        return None
    _, singular, vectors = np.linalg.svd(fit.jac, full_matrices=False)
    if singular[-1] < 1e-6:
        return None
    noise = max(1., 1.48 * np.median(np.abs(fit.fun)))
    covariance = (vectors.T / singular**2) @ vectors * noise**2
    if np.sqrt(np.max(np.linalg.eigvalsh(covariance[:3, :3]))) > 0.25 * model.size:
        return None
    pose = np.r_[origin, pose_angles(matrix)]
    return PoseMeasurement(pose, covariance, tuple(g.view.camera.index for g in observations),
                           float(np.median(after)))
