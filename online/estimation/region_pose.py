"""Occlusion-aware registration of a metric CAD silhouette."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from online.geometry import Observation, Rot, View, center_of, with_center
from online.tracker import Estimate

ANGLES = np.arange(24) * (2 * np.pi / 24)
DIRECTIONS = np.stack((np.cos(ANGLES), np.sin(ANGLES)), axis=1)
TRAIN = np.arange(len(DIRECTIONS)) % 2 == 0


@dataclass
class Region:
    frame: int
    pose: np.ndarray
    bias: np.ndarray


def clipped_silhouette(view, pose):
    pixels, depth = view.project(view.model.vertices @ Rot(*pose[3:]).T + pose[:3])
    if depth.min() <= .05:
        return ()
    width, height = view.camera.width - 1, view.camera.height - 1
    boundary = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    parts = []
    start = 0
    for part in view.model.parts:
        projected = np.ascontiguousarray(pixels[start:start + len(part)], dtype=np.float32)
        start += len(part)
        if len(projected) < 3:
            continue
        hull = cv2.convexHull(projected).reshape(-1, 2)
        if len(hull) < 3:
            continue
        if (hull >= 0).all() and (hull <= [width, height]).all():
            parts.append(hull)
        else:
            area, visible = cv2.intersectConvexConvex(hull, boundary)
            if visible is not None and area > 0:
                parts.append(visible.reshape(-1, 2))
    return parts


def support(parts):
    return (np.vstack(parts) @ DIRECTIONS.T).max(axis=0) if len(parts) else None


def region_overlap(observation, parts):
    rendered = np.zeros_like(observation.mask)
    for part in parts:
        q = (part - observation.origin) * observation.ratio
        cv2.fillConvexPoly(rendered, np.clip(np.rint(q), -10000, 10000).astype(np.int32), 1)
    match = np.count_nonzero(rendered & observation.mask)
    return match / max(1, np.count_nonzero(rendered | observation.mask))


def fit_region(anchor, observation, prior, reach, tilt):
    target = support([observation.view.instance.contour])
    target = target - anchor.bias
    model = observation.view.model
    base_center = center_of(prior, model)
    proposals = []
    for yaw, pitch in ((0, 0), (tilt, 0), (-tilt, 0), (0, tilt), (0, -tilt)):
        pose = np.r_[anchor.pose[:5], anchor.pose[5] if len(anchor.pose) == 6 else 0.]
        pose[3:5] += [yaw, pitch]
        pose = with_center(pose, base_center, model)

        def residual(offset):
            estimate = pose.copy()
            estimate[:3] += offset
            measured = support(clipped_silhouette(observation.view, estimate))
            return measured - target if measured is not None else np.full(len(target), 5000.)

        epsilon = np.eye(3) * .015

        def derivative(offset):
            return np.stack([(residual(offset + d) - residual(offset - d)) / .030 for d in epsilon], axis=1)

        fit = least_squares(lambda d: residual(d)[TRAIN], np.zeros(3), jac=lambda d: derivative(d)[TRAIN],
                            bounds=(-reach, reach), loss="soft_l1", f_scale=2.5, max_nfev=15, ftol=.002)
        delta = fit.x
        current = pose.copy()
        current[:3] += delta
        withheld = abs(residual(delta)[~TRAIN])
        error = float(np.median(withheld))
        shape = region_overlap(observation, clipped_silhouette(observation.view, current))
        if shape >= .68 and np.linalg.norm(delta) <= reach:
            singular = np.linalg.svd(derivative(delta)[TRAIN], compute_uv=False)
            precision = max(1., 1.48 * error) / max(singular[-1], 1e-6)
            proposals.append((error, -shape, precision, current))
    if not proposals:
        return None
    best = min(proposals, key=lambda p: (p[0], p[1]))
    error, shape, precision, pose = best
    ambiguity = max((np.linalg.norm(other[3][:3] - pose[:3])
                     for other in proposals if other[0] <= error + 1.5), default=0.)
    return pose, -shape, precision, error, ambiguity


class RegionRecovery:
    def __init__(self, model, fps):
        self.model = model
        self.fps = fps
        self.anchors = {}
        self.last_pose = None
        self.last_frame = -1
        self.missing = 0

    def step(self, frame, estimate, cameras):
        self.missing = self.missing + 1 if estimate.pose is None else 0
        instances = {**estimate.tracked_2d, **estimate.views}
        observations = {key: Observation(View.of(cameras[key], item, self.model))
                        for key, item in instances.items() if key in cameras}
        if estimate.status in {"multi", "mono", "mask_multi", "cad_points"}:
            for key in estimate.views.keys() & observations.keys():
                ob = observations[key]
                parts = clipped_silhouette(ob.view, estimate.pose)
                overlap = region_overlap(ob, parts)
                if overlap < .73 or min(ob.size) < 55:
                    continue
                bias = support([ob.view.instance.contour]) - support(parts)
                if np.median(abs(bias)) < .035 * max(ob.size):
                    self.anchors[key] = Region(frame, estimate.pose, bias)
        if self.missing >= 3 and self.last_pose is not None:
            proposals = []
            for key in observations.keys() & self.anchors.keys():
                ob, anchor = observations[key], self.anchors[key]
                age = frame - anchor.frame
                if age > round(2.5 * self.fps) or min(ob.size) < 55:
                    continue
                reach = max(1.0, self.model.speed / self.fps * (frame - self.last_frame))
                tilt = min(.055, .012 + age / self.fps * .018)
                result = fit_region(anchor, ob, self.last_pose, reach, tilt)
                if result is None:
                    continue
                pose, overlap, precision, error, ambiguity = result
                if precision <= 1.2 and ambiguity <= 1.3 and error <= 3.5:
                    proposals.append((precision + ambiguity, key, pose, overlap))
            if proposals:
                _, key, pose, overlap = min(proposals, key=lambda p: p[0])
                if all(np.linalg.norm(pose[:3] - p[:3]) <= 1.3 for _, _, p, _ in proposals):
                    estimate = Estimate(pose, "cad_region", {key: instances[key]}, overlap, estimate.tracked_2d)
        if estimate.pose is not None:
            self.last_pose = estimate.pose
            self.last_frame = frame
        return estimate
