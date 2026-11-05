"""Carry a recent metric anchor into a short single-camera interval."""
from collections import deque
from itertools import combinations

import numpy as np

from online.geometry import Observation, Rot, center_of, with_center
from online.estimation.kinematics import overlap
from online.perception.vision import transport


def motion_errors(before, after, old_view, view, affine):
    vertices = view.model.vertices
    old, depth = old_view.project(vertices @ Rot(*before[3:]).T + before[:3])
    box = old_view.instance.box
    visible = (depth > 0.05) & (old >= box[:2]).all(axis=1) & (old <= box[2:]).all(axis=1)
    if visible.sum() < min(12, max(4, len(vertices) // 2)):
        return None
    projected, depth = view.project(vertices[visible] @ Rot(*after[3:]).T + after[:3])
    if np.any(depth <= 0.05):
        return None
    tracked = old[visible] @ affine[:, :2].T + affine[:, 2]
    return np.median(np.linalg.norm(projected - tracked, axis=1))


class MonoDepth:
    def __init__(self, model, fps):
        self.model = model
        self.fps = fps
        self.samples = deque(maxlen=max(4, round(0.4 * fps)))
        self.previous = None

    def remember(self, frame, pose, views, confidence):
        if len(views) < 2 or confidence < 0.68:
            return
        observations = [Observation(view) for view in views]
        directions = [obs.view.ray(obs.center)[1] for obs in observations]
        if not any(np.linalg.norm(np.cross(a, b)) >= 0.22 for a, b in combinations(directions, 2)):
            return
        if min(obs.overlap(pose) for obs in observations) >= 0.56:
            self.samples.append((frame, center_of(pose, self.model)))

    def predict(self, frame):
        nearby = [(f, c) for f, c in self.samples if frame - f <= round(0.8 * self.fps)]
        if len(nearby) < 4 or frame - nearby[-1][0] > round(self.fps / 3):
            return None
        times, positions = map(np.asarray, zip(*nearby))
        gaps = times[:, None] - times[None, :]
        i, j = np.where(gaps > 0)
        velocity = np.median((positions[i] - positions[j]) / gaps[i, j, None], axis=0)
        if np.linalg.norm(velocity) * self.fps > self.model.speed:
            return None
        projected = positions + (frame - times)[:, None] * velocity
        center = np.median(projected, axis=0)
        if np.median(np.linalg.norm(projected - center, axis=1)) > 0.035 * self.model.size:
            return None
        return center, velocity

    def correct(self, frame, pose, view, memory, previous):
        if previous is None or len(memory.history) != 2:
            return pose
        old_frame, old_gray, old_view = memory.history[0]
        new_frame, gray, _ = memory.history[1]
        if old_frame != frame - 1 or new_frame != frame or old_view.instance is None:
            return pose
        observation = Observation(view)
        prediction = self.predict(frame)
        if observation.clipped or prediction is None:
            return pose
        predicted, velocity = prediction
        origin = view.camera.loc / 100
        center = center_of(pose, self.model)
        ray = center - origin
        distance = np.linalg.norm(ray)
        if distance <= 0.05:
            return pose
        ray /= distance
        depth = (predicted - origin) @ ray
        size = self.model.size
        if depth <= 0.05 or not 0.035 * size < abs(depth - distance) < 0.31 * size:
            return pose
        corrected = origin + ray * depth
        if np.linalg.norm(corrected - predicted) >= 0.09 * size:
            return pose
        candidate = with_center(pose, corrected, self.model)
        if observation.overlap(candidate) < max(0.55, observation.overlap(pose) - 0.035):
            return pose
        tracked = transport(old_gray, gray, old_view.instance)
        if tracked is None or overlap(tracked.box, view.instance.box) < 0.85:
            return pose
        old_pose = with_center(previous, predicted - velocity, self.model)
        before = motion_errors(old_pose, pose, old_view, view, tracked.affine)
        after = motion_errors(old_pose, candidate, old_view, view, tracked.affine)
        sigma = max(1, 0.009 * min(observation.size))
        if before is None or after is None or after > 2 * sigma or before - after < 0.25 * sigma:
            return pose
        return candidate

    def step(self, frame, pose, status, confidence, views, memories, metric_measurement=False):
        previous, self.previous = self.previous, None if pose is None else pose.copy()
        if pose is None:
            self.samples.clear()
        elif status == "multi":
            self.remember(frame, pose, views, confidence)
        elif status == "mono" and len(views) == 1 and not metric_measurement:
            view = views[0]
            return self.correct(frame, pose, view, memories[view.camera.index], previous)
        return pose
