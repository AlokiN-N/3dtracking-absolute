"""Recheck weak tracker output against a separate, directed stereo estimate."""
import numpy as np

from online.estimation.stereo_recovery import StereoRecovery
from online.tracker import Estimate


class WeakTrackRecovery:
    def __init__(self, detector, prompt, model, fps):
        self.stereo = StereoRecovery(detector, model, fps, prompt, min_output_iou=0.77)
        self.max_coast = max(1, round(0.25 * fps))
        self.position = None
        self.velocity = np.zeros(3)
        self.last_measurement = None

    def clear(self):
        self.stereo.clear()
        self.position = None
        self.velocity.fill(0)
        self.last_measurement = None

    def step(self, frame, primary, cameras, images):
        weak = primary.pose is None or primary.status in {"forecast", "mono", "flow"}
        if not weak and self.position is None:
            self.stereo.clear()
            return primary

        if self.position is not None and primary.pose is not None and primary.status == "multi":
            if np.linalg.norm(primary.pose[:3] - self.position[:3] - self.velocity) <= self.stereo.step_limit:
                self.clear()
                return primary

        absent = Estimate(None, "lost", {}, 0.0, primary.tracked_2d)
        checked = self.stereo.step(frame, absent, cameras, images)
        if checked.pose is not None:
            if self.position is None:
                if primary.pose is None or np.linalg.norm(primary.pose[:3] - checked.pose[:3]) < 5.0:
                    return primary
            else:
                elapsed = frame - self.last_measurement
                motion = (checked.pose[:3] - self.position[:3]) / elapsed
                self.velocity += 0.4 * (motion - self.velocity)
            self.position = checked.pose.copy()
            self.last_measurement = frame
            return Estimate(checked.pose, "stereo_recheck", checked.views, checked.confidence, primary.tracked_2d)

        if self.position is not None:
            elapsed = frame - self.last_measurement
            if elapsed <= self.max_coast:
                pose = self.position.copy()
                pose[:3] += elapsed * self.velocity
                return Estimate(pose, "stereo_recheck_prediction", {}, 0.0, primary.tracked_2d)
            self.clear()
        return primary
