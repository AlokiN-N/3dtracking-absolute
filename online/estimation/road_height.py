"""Carry a confirmed object reference point along a level surface."""
from collections import deque
from dataclasses import dataclass, field
from math import ceil

import cv2
import numpy as np

from online.geometry import Rot
from online.estimation.kinematics import overlap
from online.motion import camera_warp
from online.perception.vision import contour_motion


@dataclass
class Anchor:
    point: np.ndarray
    seeded: int
    last_flow: int
    motions: deque = field(default_factory=lambda: deque(maxlen=7))


    def has_consistent_motion(self):
        if len(self.motions) < 4:
            return False
        motions = np.asarray(self.motions)
        directions = motions / np.linalg.norm(motions, axis=1)[:, None]
        return np.linalg.norm(directions.mean(axis=0)) >= 0.91


class RoadHeight:
    def __init__(self, fps, reference_vertices, center_offset):
        self.reference_vertices = np.asarray(reference_vertices, dtype=float)
        self.center_offset = np.asarray(center_offset, dtype=float)
        self.samples = deque(maxlen=max(12, ceil(1.05 * fps)))
        self.min_samples = max(8, ceil(0.4 * fps))
        self.max_age = 3.5 * fps
        self.max_gap = max(2, ceil(fps / 3))
        self.fps = fps
        self.anchors = {}

    def level(self, frame):
        if len(self.samples) < self.min_samples or frame - self.samples[-1][0] >= self.max_age:
            return None
        _, height, pitch = np.asarray(self.samples).T
        median = np.median(height)
        half = len(height) // 2
        change = abs(np.median(height[half:]) - np.median(height[:half]))
        if np.median(np.abs(height - median)) < 0.21 and change < 0.30 and abs(np.median(pitch)) < 0.13:
            return median
        return None

    @staticmethod
    def outline(view):
        contour = view.instance.contour
        return contour[::max(len(contour) // 100, 1)]

    @staticmethod
    def box(contour):
        return np.r_[contour.min(axis=0), contour.max(axis=0)]

    def displacement(self, pose, history):
        if len(history) != 2:
            return None
        previous_frame, previous, old_view = history[0]
        frame, current, view = history[1]
        if frame != previous_frame + 1 or view.instance is None:
            return None
        if np.linalg.norm(view.camera.loc - old_view.camera.loc) > 5:
            return None
        contour = self.outline(view)
        if min(np.ptp(contour, axis=0)) / view.camera.fpx >= 0.10:
            return None
        center = pose[:3] + Rot(*pose[3:]) @ self.center_offset
        uv, depth = view.project(center[None])
        origin, ray = view.ray(uv[0])
        if depth[0] <= 1 or ray[2] >= -0.12:
            return None
        warp = camera_warp(view, old_view) if not np.allclose(view.matrix, old_view.matrix) else None
        movement = contour_motion(previous, current, contour, warp)
        if movement is None or np.linalg.norm(movement) <= 1:
            return None
        _, back = view.ray(uv[0] - movement)
        if back[2] >= -0.10:
            return None
        delta = (center[2] - origin[2]) * (ray[:2] / ray[2] - back[:2] / back[2])
        if 0.06 < np.linalg.norm(delta) * self.fps / 30 < 1.5:
            return delta
        return None

    @staticmethod
    def visibility(view, vertices, mask):
        uv, depth = view.project(vertices)
        if depth.min() <= 1:
            return 0.0
        hull = cv2.convexHull(uv.astype(np.float32))
        return cv2.intersectConvexConvex(hull, mask)[0] / max(cv2.contourArea(hull), 1)

    def correct_at_image_edge(self, frame, pose, measured, level, observation):
        """Check a second range estimate when only part of the image is visible."""
        view, box, _, visible, agreement = observation
        camera = view.camera
        anchor = self.anchors.get(camera.index)
        if anchor is None or frame - anchor.last_flow > ceil(self.fps / 10):
            return pose
        if not anchor.has_consistent_motion() or agreement < 0.45 or visible < 0.65:
            return pose

        short_side = min(box[2:] - box[:2])
        edge = min(*box[:2], camera.width - 1 - box[2], camera.height - 1 - box[3])
        if short_side / camera.fpx >= 0.08 or edge >= max(4, 0.08 * short_side):
            return pose
        if not 0.65 < pose[2] - level < 4.5 or measured[2] >= level + 0.45:
            return pose

        ray = pose[:3] - camera.loc / 100
        ray /= max(np.linalg.norm(ray), 1)
        if ray[2] >= -0.14:
            return pose
        delta = ray * ((level - pose[2]) / ray[2])
        motion_delta = anchor.point - pose[:3]
        distance = np.linalg.norm(delta)
        motion_distance = np.linalg.norm(motion_delta)
        if delta @ motion_delta <= 0.75 * distance * motion_distance or distance >= 1.55 * motion_distance:
            return pose

        candidate = pose.copy()
        candidate[:3] += 0.85 * delta
        limits = [camera.width - 1, camera.height - 1] * 2
        current_box = view.box(pose)
        candidate_box = view.box(candidate)
        if current_box is None or candidate_box is None:
            return pose
        visible_box = np.clip(box, 0, limits)
        before = overlap(np.clip(current_box, 0, limits), visible_box)
        after = overlap(np.clip(candidate_box, 0, limits), visible_box)
        return candidate if after >= before - 0.045 else pose

    def step(self, frame, pose, measured, accepted, memories):
        if pose is None:
            self.samples.clear()
            self.anchors.clear()
            return None

        level = self.level(frame)
        for key, anchor in list(self.anchors.items()):
            if key not in memories or frame - anchor.seeded > self.max_age or frame - anchor.last_flow > self.max_gap:
                del self.anchors[key]
                continue
            history = memories[key].history
            delta = self.displacement(pose, history) if history and history[-1][0] == frame else None
            previous = np.median(anchor.motions, axis=0) if anchor.motions else None
            if delta is not None and previous is not None:
                if delta @ previous < 0.68 * np.linalg.norm(delta) * np.linalg.norm(previous):
                    delta = None
            if delta is not None:
                anchor.motions.append(delta)
                anchor.last_flow = frame
            elif previous is not None:
                delta = previous
            if delta is not None:
                anchor.point[:2] += delta

        observations = []
        reference = self.reference_vertices @ Rot(*measured[3:]).T + measured[:3]
        for view in accepted:
            outline = self.outline(view)
            mask = cv2.convexHull(outline.astype(np.float32))
            visible = self.visibility(view, reference, mask)
            box = self.box(outline)
            agreement = overlap(view.box(measured), box)
            observations.append((view, box, mask, visible, agreement))

        credible = [
            view for view, _, _, visible, agreement in observations
            if visible >= 0.75 and agreement >= 0.58
        ]
        reliable = len(credible) >= 2 and abs(measured[4]) < 0.18
        if reliable and level is not None and abs(measured[2] - level) <= 0.35:
            if np.linalg.norm(measured[:3] - pose[:3]) < 2.5:
                for view in credible:
                    if min(np.ptp(self.outline(view), axis=0)) / view.camera.fpx >= 0.10:
                        continue
                    origin = view.camera.loc / 100
                    ray = measured[:3] - origin
                    ray /= max(np.linalg.norm(ray), 1)
                    if ray[2] < -0.14:
                        point = origin + ray * ((level - origin[2]) / ray[2])
                        key = view.camera.index
                        previous = self.anchors.get(key)
                        anchor = Anchor(point, frame, frame)
                        if previous is not None:
                            anchor.motions = previous.motions
                        self.anchors[key] = anchor

        result = pose
        if level is not None and (not reliable or abs(measured[2] - level) > 0.75):
            if 0.65 < level - pose[2] < 6.0:
                best = 0.0
                fallback_score = 0.0
                fallback = None
                for view, box, mask, visible, agreement in observations:
                    anchor = self.anchors.get(view.camera.index)
                    if anchor is None or frame - anchor.last_flow > 1:
                        continue
                    if not anchor.has_consistent_motion():
                        continue
                    if agreement < 0.45:
                        continue
                    if min(box[2:] - box[:2]) / view.camera.fpx >= 0.08:
                        continue
                    delta = anchor.point - pose[:3]
                    if np.linalg.norm(delta) > 18:
                        continue
                    candidate = pose.copy()
                    candidate[:3] += 0.62 * delta
                    before = overlap(view.box(pose), box)
                    after = overlap(view.box(candidate), box)
                    if visible >= 0.45 and after >= before - 0.055:
                        if agreement * visible > best:
                            best, result = agreement * visible, candidate
                    elif visible < 0.45 and after >= before + 0.04:
                        # The measured pose may put a visible reference outside
                        # the mask. Check the flow proposal against the image too.
                        reference = self.reference_vertices @ Rot(*candidate[3:]).T + candidate[:3]
                        candidate_visible = self.visibility(view, reference, mask)
                        if candidate_visible >= 0.65 and candidate_visible - visible >= 0.15:
                            score = after * candidate_visible
                            if score > fallback_score:
                                fallback_score, fallback = score, candidate
                if best == 0 and fallback is not None:
                    result = fallback

        if level is not None and len(accepted) == 1 and len(observations) == 1 and result is pose:
            result = self.correct_at_image_edge(frame, pose, measured, level, observations[0])

        if reliable:
            self.samples.append((frame, measured[2], measured[4]))
        return result
