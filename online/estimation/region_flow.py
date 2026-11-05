"""Propagate metric CAD pose using bidirectionally checked dense image motion."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from online.estimation.cad_points import fit_points
from online.estimation.region_pose import RegionRecovery, clipped_silhouette, region_overlap
from online.estimation.surface_pose import Surface, Tracks, spatial_partition
from online.geometry import Observation, Rot, View, center_of
from online.tracker import Estimate


@dataclass
class FlowAnchor:
    start: int
    frame: int
    pose: np.ndarray
    observation: Observation
    gray: np.ndarray


def mask_on_crop(ob, low, shape):
    result = np.zeros(shape, np.uint8)
    points = np.rint(ob.view.instance.contour - low).astype(np.int32)
    cv2.fillPoly(result, [points], 1)
    return cv2.erode(result, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)


def fit_translation(pose, ob, tracks, reach):
    holdout = tracks.holdout
    if min(holdout.sum(), (~holdout).sum()) < 6:
        return None
    world = tracks.points @ Rot(*pose[3:]).T + pose[:3]

    def residual(delta, subset):
        pixels, depth = ob.view.project(world[subset] + delta)
        error = pixels - tracks.pixels[subset]
        error[depth < .05] = 1000.
        return error.ravel()

    fit = least_squares(residual, np.zeros(3), args=(~holdout,), bounds=(-reach, reach),
                        loss="soft_l1", f_scale=1.75, max_nfev=14)
    error = np.linalg.norm(residual(fit.x, holdout).reshape(-1, 2), axis=1)
    if np.median(error) > 2.1 or np.mean(error < 3.5) < .8:
        return None
    smallest = np.linalg.svd(fit.jac, compute_uv=False)[-1]
    precision = max(1., 1.48 * np.median(abs(fit.fun))) / max(1e-6, smallest)
    if precision > 1.05:
        return None
    candidate = pose.copy()
    candidate[:3] += fit.x
    return candidate, ob.overlap(candidate), precision


class RegionFlowRecovery:
    def __init__(self, model, fps):
        self.model = model
        self.fps = fps
        self.surface = Surface(model)
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.anchors = {}
        self.keyframes = {}
        self.missing = 0
        self.last_pose = None
        self.last_frame = -1
        self.previous_status = None

    def propagate(self, anchor, ob, gray, elapsed):
        if gray.shape != anchor.gray.shape:
            return None
        bounds = np.vstack([ob.box.reshape(2, 2), anchor.observation.box.reshape(2, 2)])
        low = np.maximum(np.floor(bounds.min(axis=0)).astype(int) - 22, 0)
        high = np.minimum(np.ceil(bounds.max(axis=0)).astype(int) + 23, gray.shape[::-1])
        x, y = low
        u, v = high
        if min(u - x, v - y) < 40:
            return None
        old = np.ascontiguousarray(anchor.gray[y:v, x:u])
        new = np.ascontiguousarray(gray[y:v, x:u])
        before = mask_on_crop(anchor.observation, low, old.shape)
        after = mask_on_crop(ob, low, new.shape)
        points = cv2.goodFeaturesToTrack(old, 140, .004, max(3., min(old.shape) / 26), mask=before)
        if points is None or len(points) < 16:
            return None
        xy = points[:, 0]
        forward = self.flow.calc(old, new, None)
        reverse = self.flow.calc(new, old, None)
        ix, iy = np.rint(xy).astype(int).T
        shifted = xy + forward[iy, ix]
        ix, iy = np.rint(shifted).astype(int).T
        in_bounds = (ix >= 0) & (iy >= 0) & (ix < old.shape[1]) & (iy < old.shape[0])
        ids = np.flatnonzero(in_bounds)
        keep = np.linalg.norm(shifted[ids] + reverse[iy[ids], ix[ids]] - xy[ids], axis=1) < 1.0
        ids = ids[keep & (after[iy[ids], ix[ids]] > 0)]
        if len(ids) < 14:
            return None
        surface, visible = self.surface.intersect(anchor.observation.view, anchor.pose, xy[ids] + low)
        pixels = shifted[ids[visible]] + low
        tracks = Tracks(surface, pixels, spatial_partition(pixels, ob.box))
        reach = max(.5, self.model.speed / self.fps) * elapsed
        solution = fit_translation(anchor.pose, ob, tracks, reach)
        if solution is None:
            solution = fit_points(anchor.pose, ob, tracks, self.model)
        if solution is None:
            return None
        pose, _, precision = solution
        overlap = region_overlap(ob, clipped_silhouette(ob.view, pose))
        displacement = np.linalg.norm(center_of(pose, self.model) - center_of(anchor.pose, self.model))
        if overlap < .67 or displacement > reach:
            return None
        return pose, overlap, precision

    def step(self, frame, estimate, cameras, images):
        self.missing = self.missing + 1 if estimate.pose is None else 0
        continuing_cad = self.previous_status == "cad_points" and self.last_frame == frame - 1
        self.previous_status = estimate.status
        instances = {**estimate.tracked_2d, **estimate.views}
        anchors = {}
        keyframes = {}
        proposed = []
        for key in cameras.keys() & instances.keys():
            current = Observation(View.of(cameras[key], instances[key], self.model))
            if min(current.size) < 46:
                continue
            anchor = self.anchors.get(key)
            keyframe = self.keyframes.get(key)
            fresh = key in estimate.views and estimate.status in {"multi", "mono", "mask_multi", "cad_points"}
            if not fresh and anchor is None and keyframe is None:
                continue
            gray = cv2.cvtColor(images[key], cv2.COLOR_BGR2GRAY)
            if fresh:
                overlap = region_overlap(current, clipped_silhouette(current.view, estimate.pose))
                if overlap >= .76:
                    anchors[key] = keyframes[key] = FlowAnchor(frame, frame, estimate.pose, current, gray)
                    continue
            sources = [candidate for candidate in (anchor, keyframe) if candidate is not None
                       and frame - candidate.frame <= 8 and frame - candidate.start <= round(1.8 * self.fps)]
            if keyframe is not None:
                keyframes[key] = keyframe
            for source in sources:
                result = self.propagate(source, current, gray, frame - source.frame)
                if result is not None:
                    break
            else:
                if sources and anchor is not None:
                    anchors[key] = anchor
                continue
            pose, overlap, precision = result
            anchors[key] = FlowAnchor(source.start, frame, pose, current, gray)
            if keyframe is None or frame - keyframe.frame >= 4:
                keyframes[key] = anchors[key]
            if self.missing >= 3 or continuing_cad and source.frame == frame - 1:
                proposed.append((precision, key, pose, overlap))
        self.anchors = anchors
        self.keyframes = keyframes
        if proposed and self.last_pose is not None:
            _, key, pose, overlap = min(proposed, key=lambda p: p[0])
            reach = max(1.0, 2 * self.model.speed / self.fps) * (frame - self.last_frame)
            if (np.linalg.norm(pose[:3] - self.last_pose[:3]) <= reach
                    and all(np.linalg.norm(other[:3] - pose[:3]) <= 1.5 for _,_,other,_ in proposed)):
                estimate = Estimate(pose, "cad_flow", {key:instances[key]}, overlap, estimate.tracked_2d)
        if estimate.pose is not None:
            self.last_pose = estimate.pose
            self.last_frame = frame
        return estimate


class CadRegionRecovery:
    """Prefer verified image motion and use contour geometry when that motion is unavailable."""

    def __init__(self, model, fps):
        self.flow = RegionFlowRecovery(model, fps)
        self.region = RegionRecovery(model, fps)
        self.last_pose = None
        self.last_frame = -1
        self.step_limit = min(2.8, max(1.0, model.speed / fps))

    def step(self, frame, estimate, cameras, images):
        for method in (self.flow, self.region):
            method.last_pose = self.last_pose
            method.last_frame = self.last_frame
        motion = self.flow.step(frame, estimate, cameras, images)
        shape = self.region.step(frame, estimate, cameras)
        if estimate.pose is None and self.last_pose is not None:
            reach = self.step_limit * (frame - self.last_frame)
            for candidate in (motion, shape):
                if candidate.pose is not None and np.linalg.norm(candidate.pose[:3] - self.last_pose[:3]) <= reach:
                    estimate = candidate
                    break
        if estimate.pose is not None:
            self.last_pose = estimate.pose
            self.last_frame = frame
        return estimate
