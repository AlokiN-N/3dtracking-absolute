"""Bound position corrections after missing or weak tracking evidence."""
import numpy as np

from online.estimation.gap_fill import FILLED
from online.tracker import Estimate


PREDICTED = frozenset((*FILLED, "forecast", "startup_prediction"))


class GapHandoff:
    def __init__(self, fps, speed):
        self.limit = max(1.5, 2 * speed / fps)
        self.last_position = None
        self.previous_gap = False
        self.previous_predicted = False
        self.rejoining_gap = False
        self.previous_limited = False

    def step(self, estimate, primary_missing):
        gap = primary_missing and estimate.status != "stereo_recovery"
        if estimate.pose is None:
            self.last_position = None
            self.previous_gap = gap
            self.previous_predicted = False
            self.rejoining_gap = False
            self.previous_limited = False
            return estimate

        from_gap = gap or self.previous_gap or self.rejoining_gap
        weak_return = estimate.status == "mono" and (self.previous_predicted or self.previous_limited)
        point = estimate.pose[:3]
        limited = False
        result = estimate
        if self.last_position is not None and (from_gap or weak_return):
            shift = point - self.last_position
            distance = np.linalg.norm(shift)
            if distance > self.limit:
                pose = estimate.pose.copy()
                pose[:3] = self.last_position + shift * (self.limit / distance)
                result = Estimate(pose, "rejoining", {}, 0.0, estimate.tracked_2d)
                limited = True
        self.last_position = result.pose[:3].copy()
        self.previous_gap = gap
        self.previous_predicted = estimate.status in PREDICTED
        self.rejoining_gap = limited and from_gap
        self.previous_limited = limited
        return result
