"""Causal pose recovery from image tracks attached to a metric CAD surface."""
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from online.geometry import Observation, Rot, center_of, pose_angles


def spatial_partition(pixels, box):
    """Keep image regions together to limit correlation between the splits."""
    cells = np.floor(4 * (pixels - box[:2]) / np.maximum(box[2:] - box[:2], 1))
    cells = np.clip(cells, 0, 3).astype(int)
    return (cells[:, 0] + cells[:, 1]) % 2 == 0


class Surface:
    def __init__(self, model):
        triangles = model.triangles / model.size
        self.size = model.size
        self.vertices = triangles[:, 0]
        self.edge_a = triangles[:, 1] - self.vertices
        self.edge_b = triangles[:, 2] - self.vertices

    def intersect(self, view, pose, pixels):
        """Return the closest visible point for each pixel, in CAD coordinates."""
        rotation = Rot(*pose[3:])
        origin = (view.camera.loc / 100 - pose[:3]) @ rotation / self.size
        cam = view.camera
        directions = np.c_[(pixels[:, 0] - cam.cx) / cam.fx,
                           (pixels[:, 1] - cam.cy) / cam.fy, np.ones(len(pixels))]
        directions = directions @ view.matrix @ rotation
        nearest = np.full(len(pixels), np.inf)
        for start in range(0, len(self.vertices), 512):
            subset = slice(start, start + 512)
            a, b = self.edge_a[subset], self.edge_b[subset]
            offset = origin - self.vertices[subset]
            cross = np.cross(directions[:, None], b)
            det = np.einsum("ptk,tk->pt", cross, a)
            valid = np.abs(det) > 1e-10
            inverse = np.divide(1, det, out=np.zeros_like(det), where=valid)
            u = np.einsum("ptk,tk->pt", cross, offset) * inverse
            cross_offset = np.cross(offset, a)
            v = (directions @ cross_offset.T) * inverse
            depth = np.einsum("tk,tk->t", b, cross_offset) * inverse
            valid &= (u >= 0) & (v >= 0) & (u + v <= 1) & (depth > 0.05 / self.size)
            nearest = np.minimum(nearest, np.min(np.where(valid, depth, np.inf), axis=1))
        keep = np.isfinite(nearest)
        return (origin + nearest[keep, None] * directions[keep]) * self.size, keep


@dataclass
class Tracks:
    points: np.ndarray
    pixels: np.ndarray
    holdout: np.ndarray


def seed_tracks(surface, view, pose, gray):
    x, y, u, v = np.round(view.instance.box).astype(int)
    crop = gray[y:v+1, x:u+1]
    if not crop.size:
        return None
    offset = np.array([x, y])
    mask = np.zeros_like(crop)
    cv2.fillPoly(mask, [np.round(view.instance.contour - offset).astype(np.int32)], 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    corners = cv2.goodFeaturesToTrack(crop, 80, 0.015, max(3, min(crop.shape) / 18), mask=mask)
    if corners is None or len(corners) < 16:
        return None
    pixels = corners[:, 0] + offset
    points, keep = surface.intersect(view, pose, pixels)
    pixels = pixels[keep]
    holdout = spatial_partition(pixels, view.instance.box)
    return Tracks(points, pixels, holdout) if min(holdout.sum(), (~holdout).sum()) >= 6 else None


def follow(tracks, previous, gray, observation):
    if previous.shape != gray.shape:
        return None
    bounds = np.stack([tracks.pixels.min(axis=0), tracks.pixels.max(axis=0), *observation.box.reshape(2, 2)])
    low = np.maximum(np.floor(bounds.min(axis=0)).astype(int) - 32, 0)
    high = np.minimum(np.ceil(bounds.max(axis=0)).astype(int) + 33, gray.shape[::-1])
    x, y = low
    u, v = high
    source = (tracks.pixels - low).astype(np.float32).reshape(-1, 1, 2)
    previous, gray = previous[y:v, x:u], gray[y:v, x:u]
    options = dict(winSize=(19, 19), maxLevel=3)
    current, forward, _ = cv2.calcOpticalFlowPyrLK(previous, gray, source, None, **options)
    if current is None:
        return None
    back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, previous, current, None, **options)
    if back is None:
        return None
    pixels = current[:, 0] + low
    keep = (forward[:, 0] != 0) & (backward[:, 0] != 0)
    keep &= np.isfinite(pixels).all(axis=1)
    keep &= np.linalg.norm(back[:, 0] - source[:, 0], axis=1) <= 0.75
    mask_xy = np.round((pixels[keep] - observation.origin) * observation.ratio).astype(int)
    inside = ((mask_xy >= 0) & (mask_xy < observation.mask.shape[::-1])).all(axis=1)
    x, y = mask_xy[inside].T
    ids = np.flatnonzero(keep)[inside][observation.mask[y, x] > 0]
    return Tracks(tracks.points[ids], pixels[ids], tracks.holdout[ids])


def surface_candidate(pose, observations, model):
    """Fit one pose, then check its image evidence in spatially separate regions."""
    rotation = Rot(*pose[3:])
    center = center_of(pose, model)
    size = model.size
    step_limit = 0.25

    def transform(delta):
        matrix = Rotation.from_rotvec(delta[3:]).as_matrix() @ rotation
        origin = center + size * delta[:3] - matrix @ model.center
        return matrix, origin

    def errors(delta, heldout):
        matrix, origin = transform(delta)
        result = []
        for ob, tracks in observations:
            use = tracks.holdout == heldout
            pixels, depth = ob.view.project(tracks.points[use] @ matrix.T + origin)
            residual = pixels - tracks.pixels[use]
            residual[depth <= 0.05] = 1000
            result.append(residual)
        return result

    def loss(delta):
        return np.concatenate([r.ravel() / np.sqrt(len(r)) * 4 for r in errors(delta, False)])

    zero = np.zeros(6)
    fit = least_squares(loss, zero, loss="soft_l1", f_scale=2, max_nfev=22)
    # Estimate measurement precision without counting pose regularization as evidence.
    # Fit only the observable modes when the camera geometry cannot constrain all six.
    _, singular, vectors = np.linalg.svd(fit.jac, full_matrices=False)
    noise = max(1.0, np.median(np.abs(loss(fit.x))) * 1.48)
    observable = singular >= noise * np.maximum(
        np.linalg.norm(vectors[:, :3], axis=1), np.linalg.norm(vectors[:, 3:], axis=1)
    ) / (step_limit / 2)
    if observable.sum() < 3:
        return pose
    delta = fit.x
    if not observable.all():
        basis = vectors[observable].T
        restricted = least_squares(lambda coefficients: loss(basis @ coefficients),
                                   np.zeros(basis.shape[1]), loss="soft_l1", f_scale=2, max_nfev=22)
        delta = basis @ restricted.x
    if np.linalg.norm(delta[:3]) > step_limit or np.linalg.norm(delta[3:]) > 0.25:
        return pose
    before = [np.median(np.linalg.norm(r, axis=1)) for r in errors(zero, True)]
    after = [np.median(np.linalg.norm(r, axis=1)) for r in errors(delta, True)]
    gains = np.subtract(before, after)
    if gains.min() < -0.35 or gains.max() < 0.75 or max(after) > 2.5:
        return pose
    matrix, origin = transform(delta)
    candidate = np.r_[origin, pose_angles(matrix)]
    for observation, _ in observations:
        baseline = observation.overlap(pose)
        if observation.overlap(candidate) < max(0.48, baseline - 0.025):
            return pose
    return candidate


class SurfacePose:
    def __init__(self, model, fps):
        self.model = model
        self.surface = Surface(model)
        self.horizon = max(2, round(0.45 * fps))
        self.anchor = None
        self.tracks = {}

    def step(self, frame, pose, status, confidence, accepted, views, memories):
        if pose is None:
            self.tracks.clear()
            self.anchor = None
            return pose, ()
        observations = {v.camera.index: Observation(v) for v in views}
        clean = {key: ob for key, ob in observations.items()
                 if not ob.clipped and min(ob.size) >= 24 and not ob.view.instance.flow}
        rays = [clean[key].view.ray(clean[key].center)[1] for key in accepted if key in clean]
        anchored = status == "multi" and confidence >= 0.68 and any(
            np.linalg.norm(np.cross(a, b)) >= 0.22 for a, b in combinations(rays, 2)
        )
        carried = {}
        supported = []
        if self.anchor is not None and frame - self.anchor <= self.horizon:
            for key in sorted(self.tracks.keys() & observations.keys()):
                ob = observations[key]
                history = memories[key].history
                if ob.clipped or len(history) != 2 or history[0][0] != frame - 1 or history[1][0] != frame:
                    continue
                tracks = follow(self.tracks[key], history[0][1], history[1][1], ob)
                if tracks is None or min(tracks.holdout.sum(), (~tracks.holdout).sum()) < 6:
                    continue
                carried[key] = tracks
                supported.append((ob, tracks))
        self.tracks = carried
        result = pose
        if status in {"mono", "multi", "motion", "flow"} and supported:
            if len(supported) >= 2 or status == "mono" and supported[0][0].view.camera.index in accepted:
                result = surface_candidate(pose, supported, self.model)
        if anchored:
            self.anchor = frame
            self.tracks = {}
            for key, ob in clean.items():
                if ob.overlap(pose) >= 0.58 and (key in accepted or ob.overlap(pose) >= 0.72):
                    tracks = seed_tracks(self.surface, ob.view, pose, memories[key].history[-1][1])
                    if tracks is not None:
                        self.tracks[key] = tracks
        cameras = tuple(sorted(ob.view.camera.index for ob, _ in supported)) if result is not pose else ()
        return result, cameras
