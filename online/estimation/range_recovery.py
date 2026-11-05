"""Recover a disputed range from recent metric geometry and current image flow."""
from dataclasses import dataclass

import numpy as np

from online.geometry import Rot, center_of, with_center
from online.estimation.kinematics import overlap
from online.perception.vision import transport


@dataclass
class Anchor:
    frame: int
    camera: int
    pose: np.ndarray
    shift: np.ndarray


def angular_span(view):
    return min((view.instance.box[2:] - view.instance.box[:2]) / [view.camera.fx, view.camera.fy])


def transported_pose(pose, previous, history, max_step):
    """Move the old world center to the tracked bearing at the current camera pose."""
    if len(history) != 2:
        return None
    old_frame, old_gray, old_view = history[0]
    frame, gray, view = history[1]
    if frame != old_frame + 1 or old_view.instance is None or view.instance is None:
        return None
    carried = transport(old_gray, gray, old_view.instance)
    if carried is None or overlap(carried.box, view.instance.box) < 0.84:
        return None

    old_center = center_of(previous, view.model)
    uv, old_depth = old_view.project(old_center[None])
    pixel = carried.affine @ np.r_[uv[0], 1.0]
    origin, bearing = view.ray(pixel)
    forward = view.matrix[2]
    depth = (old_center - origin) @ forward
    if old_depth[0] < 2 or depth < 2 or bearing @ forward < 0.2:
        return None
    predicted = origin + bearing * (depth / (bearing @ forward))
    if np.linalg.norm(predicted - old_center) > 1.25 * max_step:
        return None
    return with_center(pose, predicted, view.model), carried.affine


def disputed_range(pose, candidate, shift, view, max_step):
    offset = center_of(pose, view.model) - center_of(candidate, view.model)
    distance = np.linalg.norm(offset)
    ray = center_of(pose, view.model) - view.camera.loc / 100
    ray /= max(np.linalg.norm(ray), 1)
    return (distance >= 2.5 * max_step and abs(offset @ ray) >= 0.94 * distance
            and shift @ offset < -max_step * distance)


def agrees_with_flow(pose, candidate, previous, history, affine):
    """Compare how the same model vertices move under each range hypothesis."""
    old_view = history[0][2]
    view = history[1][2]

    def project(view, estimate):
        return view.project(view.model.vertices @ Rot(*estimate[3:]).T + estimate[:3])[0]

    previous_uv = project(old_view, previous)
    mapped = previous_uv @ affine[:, :2].T + affine[:, 2]
    before = np.median(np.linalg.norm(project(view, pose) - mapped, axis=1))
    after = np.median(np.linalg.norm(project(view, candidate) - mapped, axis=1))
    return before > 3 * max(after, 1.0)


class RangeRecovery:
    def __init__(self, fps, speed_limit):
        self.max_step = speed_limit / fps
        self.max_age = max(1, round(0.2 * fps))
        self.road_age = max(1, round(0.1 * fps))
        self.line = None
        self.road = {}

    def check_line(self, frame, pose, measured, accepted, memories):
        anchor = self.line
        if anchor is None or frame - anchor.frame > self.max_age or anchor.camera not in accepted:
            return pose, None
        view = accepted[anchor.camera]
        history = memories[anchor.camera].history
        if angular_span(view) <= 0.12:
            return pose, None
        tracked = transported_pose(pose, anchor.pose, history, self.max_step)
        if tracked is None:
            return pose, None
        candidate, _ = tracked
        before = overlap(view.box(pose), view.instance.box)
        after = overlap(view.box(candidate), view.instance.box)
        if after < 0.70:
            return pose, None
        propagated = Anchor(anchor.frame, anchor.camera, candidate, anchor.shift)
        if not disputed_range(pose, candidate, anchor.shift, view, self.max_step):
            return pose, propagated
        if len(accepted) == 1 and frame - anchor.frame == 1:
            residual = np.linalg.norm(candidate[:3] - measured[:3])
            if (after >= before - 0.10 and residual <= 2 * self.max_step
                    and residual < 0.6 * np.linalg.norm(pose[:3] - measured[:3])):
                return candidate, propagated
        elif len(accepted) > 1:
            # A tiny secondary silhouette can pull the joint solution down the
            # dominant viewing ray. Keep the temporal range only if the resolved
            # silhouette improves and all other detections still overlap it.
            weak = [other for key, other in accepted.items() if key != anchor.camera]
            if (after >= max(0.80, before + 0.05)
                    and all(4 * angular_span(other) <= angular_span(view)
                            and overlap(other.box(candidate), other.instance.box) >= 0.30
                            for other in weak)):
                return candidate, propagated
        return pose, propagated

    def check_road(self, frame, pose, measured, accepted, observed, memories):
        tracked = {}
        for key, anchor in self.road.items():
            if key not in observed or frame - anchor.frame > self.road_age:
                continue
            history = memories[key].history
            transported = transported_pose(pose, anchor.pose, history, self.max_step)
            if transported is not None:
                candidate, affine = transported
                tracked[key] = anchor, candidate, affine
        self.road = {key: Anchor(anchor.frame, key, candidate, anchor.shift)
                     for key, (anchor, candidate, _) in tracked.items()}
        if len(accepted) != 1:
            return pose
        key, view = next(iter(accepted.items()))
        if key not in tracked:
            return pose
        anchor, candidate, affine = tracked[key]
        history = memories[key].history
        box = view.instance.box
        if (frame - anchor.frame == 1 and angular_span(view) < 0.08
                and disputed_range(pose, candidate, anchor.shift, view, self.max_step)
                and overlap(view.box(candidate), box) >= max(0.75, overlap(view.box(pose), box) - 0.075)
                and agrees_with_flow(pose, candidate, anchor.pose, history, affine)):
            return candidate

        # The current pose can discard a real camera observation when the
        # selected single view fits a different range. Reconsider that rejected
        # observation only when the two independent feature tracks agree.
        if not agrees_with_flow(pose, candidate, anchor.pose, history, affine):
            return pose
        for other_key, (other_anchor, alternate, _) in tracked.items():
            if other_key == key:
                continue
            other = observed[other_key]
            other_box = other.instance.box
            if (angular_span(other) >= 0.08 or overlap(other.box(measured), other_box) > 0.2
                    or overlap(other.box(alternate), other_box) < 0.70
                    or overlap(view.box(alternate), box) < 0.25
                    or np.linalg.norm(center_of(alternate, view.model) - center_of(candidate, view.model)) > self.max_step):
                continue
            center = center_of(alternate, view.model)
            rays = [center - v.camera.loc / 100 for v in (view, other)]
            a, b = [ray / max(np.linalg.norm(ray), 1) for ray in rays]
            if (np.linalg.norm(np.cross(a, b)) >= 0.4
                    and disputed_range(pose, alternate, other_anchor.shift, other, self.max_step)):
                return alternate
        return pose

    def step(self, frame, pose, measured, road_pose, line_anchor, accepted, observed, memories):
        if pose is None:
            self.line = None
            self.road.clear()
            return None
        views = {view.camera.index: view for view in accepted}
        if road_pose is not pose:
            self.road = {key: Anchor(frame, key, road_pose, road_pose[:3] - pose[:3]) for key in views}
            self.line = None
            return road_pose

        result, self.line = self.check_line(frame, pose, measured, views, memories)
        if result is pose:
            visible = {view.camera.index: view for view in observed}
            result = self.check_road(frame, pose, measured, views, visible, memories)
        else:
            self.road.clear()
        if line_anchor is not None and result is pose:
            key, shift = line_anchor
            view = views[key]
            if np.linalg.norm(shift) >= 2 * self.max_step and overlap(view.box(pose), view.instance.box) >= 0.72:
                self.line = Anchor(frame, key, pose, shift)
        return result
