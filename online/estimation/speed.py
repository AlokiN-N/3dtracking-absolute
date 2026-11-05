"""Estimate observed world speed from recent CAD centres."""
from collections import deque
import numpy as np


class SpeedMeter:
    def __init__(self, fps, speed_limit):
        self.fps = fps
        self.speed_limit = speed_limit
        self.samples = deque(maxlen=max(3, round(0.35 * fps) + 1))
        self.min_span = max(2, round(0.10 * fps))

    def step(self, frame, center):
        if center is None:
            self.samples.clear()
            return None
        if self.samples:
            previous_frame, previous_center = self.samples[-1]
            step = np.linalg.norm(center - previous_center)
            if frame != previous_frame + 1 or step > 1.5 * self.speed_limit / self.fps:
                self.samples.clear()
        self.samples.append((frame, np.asarray(center).copy()))
        if self.samples[-1][0] - self.samples[0][0] < self.min_span:
            return None
        times, positions = zip(*self.samples)
        times, positions = np.asarray(times), np.asarray(positions)
        first, second = np.triu_indices(len(times), k=self.min_span)
        velocities = (positions[second] - positions[first]) * (
            self.fps / (times[second] - times[first]))[:, None]
        return float(np.linalg.norm(np.median(velocities, axis=0)))
