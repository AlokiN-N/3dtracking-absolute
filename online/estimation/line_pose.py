"""Independent metric pose from straight edges of a box-shaped model part."""
import cv2
import numpy as np
from scipy.optimize import least_squares

from online.estimation.edges import box_geometry
from online.geometry import Rot, center_of, with_center


class BoxLines:
    def __init__(self, gray, view, box):
        self.view = view
        part = view.model.edge_part
        self.low, self.high = part.min(axis=0), part.max(axis=0)
        self.corners, self.edges = box_geometry(tuple(self.low), tuple(self.high))
        padding = np.maximum((box[2:] - box[:2]) * 0.19, 12)
        low = np.maximum(np.floor(box[:2] - padding).astype(int), 0)
        high = np.minimum(np.ceil(box[2:] + padding).astype(int), gray.shape[::-1])
        x, y = low
        x2, y2 = high
        self.segments = np.empty((0, 2, 2))
        if min(x2-x, y2-y) < 24:
            return
        detected = cv2.createLineSegmentDetector().detect(gray[y:y2, x:x2])[0]
        if detected is None:
            return
        endpoints = detected.reshape(-1, 2, 2) + low
        delta = endpoints[:, 1] - endpoints[:, 0]
        length = np.linalg.norm(delta, axis=1)
        valid = length >= max(8, min(box[2:] - box[:2]) * 0.105)
        self.segments = endpoints[valid]
        self.length = length[valid]
        self.direction = delta[valid] / self.length[:, None]
        self.normals = np.column_stack([-self.direction[:, 1], self.direction[:, 0]])

    def project(self, pose):
        return self.view.project(self.corners @ Rot(*pose[3:]).T + pose[:3])[0]

    def match(self, pose, radius):
        if not len(self.segments):
            return []
        uv = self.project(pose)
        camera = (self.view.camera.loc / 100 - pose[:3]) @ Rot(*pose[3:])
        faces_hi, faces_lo = camera > self.high, camera < self.low
        candidates = []
        for a, b, axes, sides in self.edges:
            if not np.any(np.where(sides, faces_hi[axes], faces_lo[axes])):
                continue
            delta = uv[b] - uv[a]
            length = np.linalg.norm(delta)
            if length < 12:
                continue
            tangent = delta / length
            similarity = np.abs(self.direction @ tangent)
            intervals = (self.segments - uv[a]) @ tangent
            overlap = np.maximum(0, np.minimum(length, intervals.max(axis=1)) - np.maximum(0, intervals.min(axis=1)))
            coverage = overlap / length
            offsets = np.einsum("nkd,nd->nk", uv[[a, b]][None] - self.segments[:, :1], self.normals)
            distance = np.sqrt(np.mean(offsets**2, axis=1))
            keep = (similarity > 0.974) & (coverage > 0.33) & (distance < radius)
            axis = int(np.argmax(np.abs(self.corners[b] - self.corners[a])))
            for index in np.flatnonzero(keep):
                score = (0.45 + 0.55 * min(coverage[index], 1)) * (1 - 0.67 * distance[index] / radius)
                candidates.append((score, index, a, b, axis, distance[index]))
        matches = []
        edges, segments = set(), set()
        for candidate in sorted(candidates, reverse=True):
            _, index, a, b, _, _ = candidate
            if index not in segments and (a, b) not in edges:
                matches.append(candidate)
                segments.add(index)
                edges.add((a, b))
        return matches

    @staticmethod
    def sufficient(matches):
        return len(matches) >= 4 and len({m[4] for m in matches}) >= 2

    def refine(self, pose, box):
        if len(self.segments) < 4 or min(box[2:] - box[:2]) < 40:
            return None
        center = center_of(pose, self.view.model)
        origin = self.view.camera.loc / 100
        radius = np.clip(0.14 * min(box[2:] - box[:2]), 10, 28)
        tight = max(2.5, 0.033 * min(box[2:] - box[:2]))
        initial = self.match(pose, tight)
        initial_score = sum(m[0] for m in initial)
        best = None
        for scale in (1, 0.90, 1.10):
            seed = with_center(pose, origin + scale * (center - origin), self.view.model)
            matches = self.match(seed, radius)
            if not self.sufficient(matches):
                continue
            corners = np.array([[m[2], m[3]] for m in matches])
            indices = np.array([m[1] for m in matches])
            normals = self.normals[indices]
            anchors = self.segments[indices, :1]

            def residual(delta):
                candidate = with_center(pose + np.r_[np.zeros(3), delta[3:]], center_of(seed, self.view.model) + delta[:3], self.view.model)
                projected = self.project(candidate)[corners]
                distance = np.einsum("nkd,nd->nk", projected - anchors, normals)
                return np.r_[distance.ravel(), delta[3:] * [10, 21], delta[:3] * [0.16, 0.16, 0.4]]

            upper = np.array([12., 12., 5., 0.32, 0.14])
            solved = least_squares(residual, np.zeros(5), bounds=(-upper, upper), loss="soft_l1",
                                   f_scale=2, max_nfev=22, x_scale=[3, 3, 1, .1, .06])
            delta = solved.x
            candidate = with_center(pose + np.r_[np.zeros(3), delta[3:]], center_of(seed, self.view.model) + delta[:3], self.view.model)
            agreement = self.match(candidate, tight)
            if not self.sufficient(agreement):
                continue
            displacement = np.linalg.norm(center_of(candidate, self.view.model) - center)
            score = sum(m[0] for m in agreement) - 0.045 * displacement
            if displacement < 15 and score > initial_score + 0.65 and (best is None or score > best[0]):
                best = score, candidate
        return None if best is None else best[1]


def refine_range(pose, gray, view, box):
    """Use the independent line fit only when the cargo box is resolved."""
    if view.model.edge_part is None:
        return pose
    camera = view.camera
    if min((box[2:] - box[:2]) / [camera.fx, camera.fy]) <= 0.12:
        return pose
    center = center_of(pose, view.model)
    ray = center - camera.loc / 100
    ray /= max(np.linalg.norm(ray), 1)
    if abs(ray @ Rot(*pose[3:])[:, 0]) >= 0.85:
        return pose
    candidate = BoxLines(gray, view, box).refine(pose, box)
    if candidate is None:
        return pose
    depth = (center_of(candidate, view.model) - center) @ ray
    return with_center(pose, center + 0.58 * depth * ray, view.model)
