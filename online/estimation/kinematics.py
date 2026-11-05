"""Causal filtering of the output pose in model-center coordinates."""
from collections import deque
from itertools import combinations

import numpy as np

from online.geometry import Observation, angle_diff, center_of, with_center


def limited(vector, maximum):
    length = np.linalg.norm(vector)
    return vector * min(1, maximum / max(length, 1e-9))


class FlowJumpGuard:
    """Reject an implausible jump supported only by propagated silhouettes."""

    def __init__(self, model, fps):
        self.model = model
        self.step_limit = model.speed / fps
        self.history = deque(maxlen=2)

    def step(self, pose, status, confidence, accepted, views):
        if pose is None:
            self.history.clear()
            return pose
        center = center_of(pose, self.model)
        valid = status in {"mono", "multi"} and confidence >= 0.55
        cameras = frozenset(accepted) if valid else frozenset()
        if len(self.history) == 2 and status == "multi":
            pose = self.correct(pose, center, cameras, views)
            center = center_of(pose, self.model)
        self.history.append((center, cameras))
        return pose

    def correct(self, pose, center, cameras, views):
        (earlier, seen_earlier), (previous, seen_previous) = self.history
        velocity = previous - earlier
        if (len(cameras) < 2 or np.linalg.norm(velocity) > 1.5 * self.step_limit
                or np.linalg.norm(center - previous) < 2.5 * self.step_limit):
            return pose
        used = [view for view in views if view.camera.index in cameras]
        if len(used) != len(cameras) or any(not view.instance.flow for view in used):
            return pose
        stable = seen_earlier & seen_previous
        observations = [Observation(view) for view in used if view.camera.index in stable]
        if not observations:
            return pose
        best = max(observations, key=lambda observation: observation.overlap(pose))
        if best.clipped:
            return pose
        baseline = best.overlap(pose)
        if baseline < 0.70:
            return pose
        forecast = previous + limited(velocity, self.step_limit)
        candidate = with_center(pose, forecast, self.model)
        if best.overlap(candidate) < max(0.70, baseline - 0.03):
            return pose
        return candidate


def overlap(a, b):
    if a is None:
        return 0.0
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
    return intersection / max(union, 1)


def confirmed_translation(pose, predicted, views):
    """Look for independent bearings that favor the current measurement."""
    old_pose = with_center(pose, predicted)
    center = center_of(pose)
    evidence = []
    for view in views:
        box = view.instance.box
        current = overlap(view.box(pose), box)
        if current < 0.37:
            continue
        prior = overlap(view.box(old_pose), box)
        ray = center - view.camera.loc / 100
        ray /= max(np.linalg.norm(ray), 1)
        evidence.append((ray, current, current - prior))
    return any(
        np.linalg.norm(np.cross(a[0], b[0])) > 0.35
        and min(a[2], b[2]) > -0.06
        and max(a[2], b[2]) > 0.18
        and a[2] + b[2] > 0.22
        for a, b in combinations(evidence, 2)
    )


class CenterKalman:
    """Constant-velocity observer with wider uncertainty along a single ray."""

    def __init__(self, fps):
        dt = 1 / fps
        axes = np.eye(3)
        self.transition = np.eye(6)
        self.transition[:3, 3:] = dt * axes
        acceleration = np.vstack([0.5 * dt**2 * axes, dt * axes]) * [10, 10, 3.6]
        self.process_noise = acceleration @ acceleration.T
        self.state = None

    def step(self, center, cameras, observed):
        if center is None:
            self.state = None
            return None
        if self.state is None:
            self.state = np.r_[center, np.zeros(3)]
            self.covariance = np.diag([2., 2., 1., 50., 50., 12.])
            return self.state[:3]

        self.state = self.transition @ self.state
        covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        if observed:
            if len(cameras) == 1:
                ray = center - cameras[0].loc / 100
                ray /= max(np.linalg.norm(ray), 1)
                noise = np.eye(3) + (3.5**2 - 1) * np.outer(ray, ray)
            else:
                noise = np.diag([1.3**2, 1.3**2, 0.55**2])
            residual = center - self.state[:3]
            whitened = np.sqrt(residual @ np.linalg.solve(covariance[:3, :3] + noise, residual))
            noise *= max(1, whitened / 2.5) ** 2
            gain = np.linalg.solve(covariance[:3, :3] + noise, covariance[:, :3].T).T
            self.state += gain @ residual
            covariance -= gain @ covariance[:3, :]
        self.covariance = 0.5 * (covariance + covariance.T)
        return self.state[:3]


class PoseFilter:
    def __init__(self, fps):
        self.dt = 1 / fps
        self.center = None
        self.velocity = np.zeros(3)
        self.yaw = None
        self.pitch = None
        self.disagreement = 0
        self.reverse_votes = 0
        self.gaussian = CenterKalman(fps)

    def step(self, pose, status, cameras, views=()):
        if pose is None:
            self.center = None
            self.gaussian.step(None, cameras, False)
            return None

        measured = center_of(pose)
        gaussian_center = self.gaussian.step(measured, cameras, status != "forecast")
        if self.center is None:
            self.center = measured
            self.velocity[:] = 0
            self.yaw, self.pitch = pose[3:]
            self.disagreement = 0
            self.reverse_votes = 0
            return pose.copy()

        predicted = self.center + self.velocity
        innovation = measured - predicted
        size = np.linalg.norm(innovation)
        reliable_pair = status == "multi" and len(cameras) > 1
        self.disagreement = self.disagreement + 1 if size > 5 else 0
        corroborated = 4 < size < 25 and confirmed_translation(pose, predicted, views)

        # At a single viewpoint, a similar projection can occur at a different range.
        # Apply a smaller influence along the ray for an abrupt monocular correction.
        if len(cameras) == 1 and not corroborated:
            ray = measured - cameras[0].loc / 100
            ray /= max(np.linalg.norm(ray), 1)
            depth = innovation @ ray
            cap = (1.1 if abs(depth) < 5 else 0.38) * self.dt * 30
            if self.disagreement > 18:
                cap += 0.40
            innovation += (np.clip(depth, -cap, cap) - depth) * ray

        if status == "forecast":
            correction = np.zeros(3)
        elif corroborated:
            correction = 0.82 * limited(innovation, 5.0 * self.dt * 30)
        else:
            gain = 0.66 if reliable_pair else 0.56
            limit = 2.8 if reliable_pair else 1.55
            if size > 5:
                gain *= min(1, self.disagreement / 12)
                if self.disagreement > 18:
                    limit += 1.0
            correction = gain * limited(innovation, limit * self.dt * 30)
        correction[2] = np.clip(correction[2], -8 * self.dt, 8 * self.dt)
        self.center = predicted + correction
        self.velocity = limited(self.velocity + (0.045 if corroborated else 0.12) * correction, 42 * self.dt)
        self.velocity[2] = np.clip(self.velocity[2], -7 * self.dt, 7 * self.dt)

        raw_yaw = pose[3]
        yaw = raw_yaw
        if abs(angle_diff(yaw, self.yaw)) > np.pi / 2:
            yaw += np.pi
        speed = np.linalg.norm(self.velocity[:2])
        if speed > 5 * self.dt:
            course = np.arctan2(self.velocity[1], self.velocity[0])
            course_delta = angle_diff(course, yaw)
            if abs(course_delta) < np.pi / 2:
                yaw += 0.18 * course_delta
            # Change which end is the cab only when translation and the raw fit
            # agree on the opposite heading for several consecutive frames.
            reverse = (abs(angle_diff(raw_yaw, self.yaw)) > 2.1 and
                       abs(angle_diff(raw_yaw, course)) < 0.9 and size < 5)
            self.reverse_votes = self.reverse_votes + 1 if reverse else 0
            if self.reverse_votes >= 6:
                self.yaw = raw_yaw
                self.reverse_votes = 0
        else:
            self.reverse_votes = 0
        turn = np.clip(angle_diff(yaw, self.yaw), -3.5 * self.dt, 3.5 * self.dt)
        self.yaw += turn
        self.pitch += np.clip(pose[4] - self.pitch, -2.4 * self.dt, 2.4 * self.dt)
        return with_center(np.r_[pose[:3], self.yaw, self.pitch], 0.5 * (self.center + gaussian_center))
