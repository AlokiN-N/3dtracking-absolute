"""Recover a missing pose from visible points attached to a metric CAD mesh."""
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from online.estimation.surface_pose import Surface, Tracks, follow, seed_tracks, spatial_partition
from online.geometry import Observation, Rot, View, center_of, fit_free, pose_angles
from online.tracker import Estimate


@dataclass
class Anchor:
    frame: int
    gray: np.ndarray
    tracks: Tracks
    pose: np.ndarray


def fit_points(prior, observation, tracks, model):
    """Check held-out image regions and uncertainty of the CAD reference point."""
    if min(tracks.holdout.sum(), (~tracks.holdout).sum()) < 6:
        return None
    rotation = Rot(*prior[3:])
    center = center_of(prior, model)
    view = observation.view
    scale = model.size

    def pose_at(delta):
        matrix = Rotation.from_rotvec(delta[3:]).as_matrix() @ rotation
        origin = center + scale * delta[:3] - matrix @ model.center
        return matrix, origin

    def residual(delta, selected):
        matrix, origin = pose_at(delta)
        pixels, depth = view.project(tracks.points[selected] @ matrix.T + origin)
        error = pixels - tracks.pixels[selected]
        error[depth < 0.05] = 1000
        return error.ravel()

    train = ~tracks.holdout
    fit = least_squares(residual, np.zeros(6), args=(train,), loss="soft_l1", f_scale=1.75, max_nfev=26)
    if not np.isfinite(fit.x).all() or np.linalg.norm(fit.x[3:]) > 0.30:
        return None
    error = np.linalg.norm(residual(fit.x, tracks.holdout).reshape(-1, 2), axis=1)
    if np.median(error) > 2.5 or np.mean(error < 4.0) < 0.75:
        return None
    _, singular, basis = np.linalg.svd(fit.jac, full_matrices=False)
    if singular[-1] < 1e-6:
        return None
    matrix, origin = pose_at(fit.x)
    pose = np.r_[origin, pose_angles(matrix)]
    # Finite differences include the offset between the CAD origin and its center.
    jacobian = np.column_stack([(pose_at(fit.x + step)[1] - origin) / 1e-5 for step in np.eye(6) * 1e-5])
    noise = max(1.0, np.median(np.abs(fit.fun)) * 1.48)
    precision = np.linalg.norm(jacobian @ (basis.T / singular), ord=2) * noise
    if precision > min(1.0, 0.09 * scale):
        return None
    overlap = observation.overlap(pose)
    if overlap < 0.52:
        return None
    return pose, overlap, precision


class CadPointRecovery:
    """Keep independent camera tracks and never feed a recovered pose to the main tracker."""

    def __init__(self, model, fps, matcher=None):
        self.model = model
        self.surface = Surface(model)
        self.fps = fps
        self.horizon = max(1, round(3.5 * fps))
        self.min_missing = max(2, round(fps / 8))
        self.anchors = {}
        self.previous_frame = -1
        self.last_output = None
        self.last_output_frame = -1
        self.missing = 0
        self.precision_m = None
        self.matcher = matcher
        self.references = {}
        self.last_measured = None
        self.previous_matches = {}

    def anchor_pose(self, observation, others, pose):
        overlap = observation.overlap(pose)
        if overlap >= 0.62:
            return pose
        # A weak stereo silhouette can bias its pose for the camera we want to seed.
        # Refine only the anchor and retain the independent views as a check.
        witnesses = [(other, other.overlap(pose)) for other in others if other is not observation]
        if not witnesses:
            return None
        proposals = [fit_free(others, pose, 0)[0], fit_free([observation], pose, 0)[0]]
        best = None
        for candidate in proposals:
            if np.linalg.norm(candidate[:3] - pose[:3]) > 0.12 * self.model.size:
                continue
            score = observation.overlap(candidate)
            if score < max(0.62, overlap + 0.04):
                continue
            if any(other.overlap(candidate) < max(0.38, baseline - 0.08) for other, baseline in witnesses):
                continue
            if best is None or score > best[0]:
                best = score, candidate
        return best[1] if best is not None else None

    def stereo_geometry(self, pose, observations):
        center = center_of(pose, self.model)
        information = []
        for observation in observations:
            camera = observation.view.camera
            offset = center - camera.loc / 100
            distance = np.linalg.norm(offset)
            if distance < 0.05:
                continue
            direction = offset / distance
            noise_px = max(1.5, 0.003 * max(observation.size))
            noise_m = distance * noise_px / camera.fpx
            information.append((np.eye(3) - np.outer(direction, direction)) / noise_m**2)
        limit = max(0.25, min(1.2, 0.09 * self.model.size))
        return any(np.linalg.eigvalsh(a + b)[0] >= 1 / limit**2 for a, b in combinations(information, 2))

    def match_points(self, frame, key, observation, image):
        if self.last_measured is None:
            return None
        last_frame, last_pose = self.last_measured
        gap = frame - last_frame
        if gap > self.horizon or min(observation.size) < 32:
            return None
        camera_reference = self.references.get(key)
        priors = [last_pose]
        if camera_reference is not None and frame - camera_reference[0] <= self.horizon:
            priors.append(camera_reference[1])
        seeds = [fit_free([observation], prior, 0)[0] for prior in priors]
        recent = self.previous_matches.get(key)
        if recent is None and self.last_output_frame == frame - 1:
            recent = self.last_output
        if recent is not None:
            seeds.append(recent)
        solutions = []
        for seed in seeds:
            if observation.overlap(seed) < 0.57:
                continue
            groups = self.matcher.correspondences(seed, {key: image}, [observation.view])
            if not groups:
                continue
            group = groups[0]
            indices = np.round((group.pixels - observation.origin) * observation.ratio).astype(int)
            inside = ((indices >= 0) & (indices < observation.mask.shape[::-1])).all(axis=1)
            x, y = indices[inside].T
            keep = np.flatnonzero(inside)[observation.mask[y, x] > 0]
            if len(keep) > 160:
                keep = keep[np.linspace(0, len(keep) - 1, 160, dtype=int)]
            pixels = group.pixels[keep]
            tracks = Tracks(group.points[keep], pixels, spatial_partition(pixels, observation.box))
            result = fit_points(seed, observation, tracks, self.model)
            if result is None:
                continue
            pose, overlap, precision = result
            reach = max(1.0, gap * self.model.speed / self.fps)
            if np.linalg.norm(center_of(pose, self.model) - center_of(last_pose, self.model)) <= reach:
                solutions.append((precision, key, pose, overlap))
        if not solutions:
            return None
        best = min(solutions, key=lambda candidate: candidate[0])
        if any(np.linalg.norm(candidate[2][:3] - best[2][:3]) > 1.5 for candidate in solutions):
            return None
        return best

    def step(self, frame, estimate, cameras, images):
        instances = {**estimate.tracked_2d, **estimate.views}
        observations = {key: Observation(View.of(cameras[key], item, self.model))
                        for key, item in instances.items() if key in cameras}
        self.missing = self.missing + 1 if estimate.pose is None else 0
        self.precision_m = None
        if estimate.status in {"multi", "mono", "mask_multi", "mask_large"}:
            self.last_measured = frame, estimate.pose
        anchors = {}
        candidates = []
        accepted = [observations[key] for key in estimate.views if key in observations]
        stereo = (estimate.status in {"multi", "mask_multi"} and estimate.confidence >= 0.42
                  and self.stereo_geometry(estimate.pose, accepted))
        strong_stereo = estimate.status == "multi" and estimate.confidence >= 0.62
        for key, observation in observations.items():
            anchor = self.anchors.get(key)
            if (anchor is None or frame - anchor.frame > self.horizon
                    or frame != self.previous_frame + 1):
                anchor = None
            if anchor is None and (not stereo or key not in estimate.views):
                continue
            gray = cv2.cvtColor(images[key], cv2.COLOR_BGR2GRAY)
            if anchor is not None:
                tracks = follow(anchor.tracks, anchor.gray, gray, observation)
                if tracks is not None and min(tracks.holdout.sum(), (~tracks.holdout).sum()) >= 6:
                    anchors[key] = Anchor(anchor.frame, gray, tracks, anchor.pose)
                    if self.missing >= self.min_missing:
                        candidate = fit_points(anchor.pose, observation, tracks, self.model)
                        if candidate is not None:
                            pose, score, precision = candidate
                            candidates.append((precision, key, pose, score))
            if (stereo and key in estimate.views and (not observation.clipped or strong_stereo)
                    and (key not in anchors or frame - anchor.frame >= max(1, round(self.fps / 4)))):
                pose = self.anchor_pose(observation, accepted, estimate.pose)
                if pose is not None:
                    tracks = seed_tracks(self.surface, observation.view, pose, gray)
                    if tracks is not None:
                        anchors[key] = Anchor(frame, gray, tracks, pose)
                        if self.matcher is not None:
                            self.references[key] = frame, pose
        matches = {}
        if self.matcher is not None and self.missing >= self.min_missing:
            matched = {key for _, key, _, _ in candidates}
            for key, observation in observations.items():
                if key in matched:
                    continue
                candidate = self.match_points(frame, key, observation, images[key])
                if candidate is None:
                    continue
                matches[key] = candidate[2]
                if key not in anchors:
                    gray = cv2.cvtColor(images[key], cv2.COLOR_BGR2GRAY)
                    tracks = seed_tracks(self.surface, observation.view, candidate[2], gray)
                    if tracks is not None:
                        anchors[key] = Anchor(self.last_measured[0], gray, tracks, candidate[2])
                previous = self.previous_matches.get(key)
                if previous is not None and np.linalg.norm(candidate[2][:3] - previous[:3]) <= max(
                        0.75, 1.5 * self.model.speed / self.fps):
                    candidates.append(candidate)
        self.previous_matches = matches
        self.anchors = anchors
        self.previous_frame = frame
        if candidates and self.last_output is not None:
            precision, key, pose, score = min(candidates, key=lambda candidate: candidate[0])
            limit = max(1.5, 2 * self.model.speed / self.fps) * (frame - self.last_output_frame)
            if (np.linalg.norm(pose[:3] - self.last_output[:3]) <= limit
                    and np.linalg.norm(center_of(pose, self.model) - center_of(self.last_output, self.model)) <= limit
                    and all(np.linalg.norm(other[:3] - pose[:3]) <= max(0.75, 2 * (precision + other_precision))
                            for other_precision, _, other, _ in candidates)):
                estimate = Estimate(pose, "cad_points", {key: instances[key]}, score, estimate.tracked_2d)
                self.precision_m = precision
                for _, camera, candidate_pose, _ in candidates:
                    if camera in self.anchors:
                        self.anchors[camera].pose = candidate_pose
        if estimate.pose is not None:
            self.last_output = estimate.pose
            self.last_output_frame = frame
        return estimate
