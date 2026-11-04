"""Strictly forward state machine. Each call consumes one synchronized frame."""
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

from online.estimation.edges import BoxEdges
from online.geometry import (DEFAULT_TARGET, Observation, Rot, View, angle_diff,
                       center_of, detailed, fit, intersects, optimize, with_center)
from online.estimation.kinematics import PoseFilter
from online.estimation.line_pose import refine_range
from online.estimation.range_recovery import RangeRecovery
from online.perception.resilience import plausible_box, prepare_image, stabilize
from online.estimation.road_height import RoadHeight
from online.motion import camera_warp, motion_boxes
from online.perception.vision import Instance, contour_motion, transport


@dataclass
class CameraMemory:
    history: deque = field(default_factory=lambda: deque(maxlen=2))
    mask: object = None
    frame: int = -2
    detected: int = -3
    prompted: int = -50
    exposure: np.ndarray | None = None


@dataclass
class Estimate:
    pose: np.ndarray | None
    status: str
    views: dict
    confidence: float


def box_overlap(a, b):
    common = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    return common / max((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - common, 1)


class Tracker:
    def __init__(self, segmenter, fps, cadence=3, prompt=None, model=DEFAULT_TARGET, seed_boxes=None,
                 robust=False, adaptive=False, video_masks=None):
        self.model = model
        self.seed_boxes = seed_boxes or {}
        self.acquired = False
        self.segmenter = segmenter
        self.prompt = prompt
        self.fps = fps
        self.cadence = cadence
        self.robust = robust
        self.adaptive = adaptive
        self.video_masks = video_masks
        self.frame_guard = robust
        self.guard_through = -1
        self.memory = {}
        self.camera_ids = set()
        self.current_views = {}
        self.camera_health = {}
        self.pose = None
        self.velocity = np.zeros(3)
        self.confidence = 0.0
        self.gap = 0
        self.frame = -1
        self.altitude = None
        self.altitude_age = 0
        self.output = PoseFilter(fps) if model.ground_vehicle else None
        self.range_recovery = RangeRecovery(fps, speed_limit=model.speed) if model.edge_part is not None or model.level_surface else None
        reference = model.reference_part if model.reference_part is not None else model.vertices
        self.road_height = RoadHeight(fps, reference, model.center) if model.level_surface else None

    def candidates(self, key, image, gray, view, expected, carried):
        mem = self.memory[key]
        height, width = image.shape[:2]
        motion = None
        old_gray, old_view = None, None
        if mem.history:
            _, old_gray, old_view = mem.history[0]
            motion = motion_boxes(gray, old_gray, view, old_view, compensate_exposure=self.frame_guard)
        expected = expected if intersects(expected, width, height) else None
        if expected is not None:
            expected = np.clip(expected, [0, 0, 0, 0], [width-1, height-1, width-1, height-1])
        motion_box = None
        if motion:
            reference = carried.box if carried is not None else expected
            motion_box = max(motion[:4], key=lambda item: item[1] ** 0.75 *
                             (1 + 0.35 * box_overlap(item[0], reference) if reference is not None else 1))[0]
        if not self.model.ground_vehicle and not self.acquired and self.seed_boxes:
            seed = self.seed_boxes.get(key)
            if seed is None:
                return [], motion_box
            seed = np.clip(seed, [0, 0, 0, 0], [width - 1, height - 1] * 2)
            if min(seed[2:] - seed[:2]) < 10:
                return [], motion_box
            detected = self.prompt.search(image, seed, precise=True) if self.prompt is not None else None
            if detected is not None and box_overlap(detected.box, seed) >= 0.30:
                return [detected], motion_box
            matches = [d for d in self.segmenter.search(image, seed) if box_overlap(d.box, seed) >= 0.35]
            return matches, motion_box
        age = self.frame - mem.detected
        agreement = box_overlap(carried.box, motion_box) if carried is not None and motion_box is not None else 0
        if carried is not None and age < self.cadence and (motion_box is None or agreement > 0.48):
            return [carried], motion_box
        found = []
        if carried is not None and age < 15 and (motion_box is None or agreement > 0.40):
            found.append((0.66 - 0.021 * age, carried))
        target = motion_box if motion_box is not None else expected
        if motion_box is not None and carried is not None and agreement >= 0.55:
            target = np.r_[np.minimum(motion_box[:2], carried.box[:2]),
                           np.maximum(motion_box[2:], carried.box[2:])]
        if not self.model.ground_vehicle and expected is not None:
            target = expected
        if target is not None and self.prompt is not None:
            elapsed = self.frame - mem.prompted
            should_prompt = motion_box is not None and (elapsed >= 7 or (not found and elapsed >= 4))
            if not self.model.ground_vehicle and expected is not None and not found and elapsed >= self.cadence:
                should_prompt = True
            if should_prompt:
                mem.prompted = self.frame
                detected = self.prompt.search(image, target, precise=not self.model.ground_vehicle)
                if detected is not None and box_overlap(detected.box, target) >= 0.42:
                    movement = None
                    if self.model.ground_vehicle and old_view is not None:
                        transform = camera_warp(view, old_view) if not np.allclose(view.matrix, old_view.matrix) else None
                        movement = contour_motion(old_gray, gray, detected.contour, transform)
                    if movement is None or np.linalg.norm(movement) >= 0.50:
                        found.append((0.32 + 0.55 * detected.score, detected))
        if not found or (age >= self.cadence and self.frame % self.cadence == key % self.cadence
                         and not any(d.source == "sam" and not d.flow for _, d in found)):
            if target is not None and (motion is None or motion_box is not None or carried is not None):
                for detection in self.segmenter.search(image, target):
                    if not self.model.ground_vehicle and self.model.labels and detection.label not in self.model.labels:
                        continue
                    match = box_overlap(detection.box, target)
                    if match < 0.28:
                        continue
                    score = detection.score * (1 if (not self.model.labels or detection.label in self.model.labels) else 0.74)
                    if score >= (0.14 if motion_box is not None else 0.26):
                        found.append((0.16 + 0.50 * score + 0.20 * match, detection))
            elif (self.pose is None or self.gap > 5) and self.frame % 4 == 0:
                for detection in self.segmenter.search(image):
                    size = detection.box[2:] - detection.box[:2]
                    if (not self.model.labels or detection.label in self.model.labels) and detection.score >= 0.24 and max(size) < 0.7 * min(width, height):
                        if self.frame == 0 and self.prompt is not None:
                            refined = self.prompt.search(image, detection.box, precise=not self.model.ground_vehicle)
                            if refined is not None and box_overlap(refined.box, detection.box) > 0.52:
                                detection = refined
                                mem.prompted = self.frame
                        found.append((detection.score, detection))
        return [d for _, d in sorted(found, key=lambda item: item[0], reverse=True)], motion_box

    def flow_pose(self, flows, prior, previous):
        results = []
        for key, (view, old, instance) in flows.items():
            if self.frame - self.memory[key].detected > 22:
                continue
            center = self.model.center @ Rot(*previous[3:]).T
            prev_uv, old_depth = old.project((previous[:3] + center)[None])
            box = instance.box
            span = box[2:] - box[:2]
            mapped = instance.affine @ np.r_[prev_uv[0], 1.0]
            near = 2 if self.model.ground_vehicle else 0.06
            if old_depth[0] < near or np.any(mapped < box[:2] - 0.20 * span) or np.any(mapped > box[2:] + 0.20 * span):
                continue
            origin, ray = view.ray(mapped)
            expected = prior[:3] + center
            projected_depth = (expected - origin) @ ray
            if self.altitude is not None and ray[2] < -0.12:
                plane_depth = (self.altitude + center[2] - origin[2]) / ray[2]
                if abs(plane_depth - projected_depth) < 10:
                    projected_depth = 0.80 * plane_depth + 0.20 * projected_depth
            point = origin + ray * projected_depth - center
            if projected_depth > near and np.linalg.norm(point - prior[:3]) < 3.0:
                results.append((instance.score, key, instance, point))
        if not results:
            return None
        _, key, instance, point = max(results, key=lambda row: row[0])
        pose = prior.copy()
        pose[:3] += 0.70 * (point - prior[:3])
        return pose, key, instance

    def motion_pose(self, proposals, prior):
        """Recover from a motion proposal by aligning current cargo edges at the recent altitude."""
        if self.altitude is None or self.model.edge_part is None:
            return None
        best = None
        for view, gray, box in proposals:
            observed = box[2:] - box[:2]
            if min(observed) < 26:
                continue
            target = (box[:2] + box[2:]) / 2
            origin, ray = view.ray(target)
            if ray[2] >= -0.13:
                continue
            for change in (0.0, -0.55, 0.55):
                seed = prior.copy()
                seed[3] += change
                offset = self.model.center @ Rot(*seed[3:]).T
                depth = (self.altitude + offset[2] - origin[2]) / ray[2]
                if depth < 8:
                    continue
                seed[:3] = origin + ray * depth - offset
                seed_box = view.box(seed)
                if seed_box is None or box_overlap(box, seed_box) < 0.34:
                    continue
                edge = BoxEdges(gray, view, box, seed)
                if len(edge.a) < 4:
                    continue
                pose, _ = optimize([], seed, seed, prior[3], prior[4], [5, 5, 1],
                                      self.altitude_prior(), [edge], phase="refine", model=self.model)
                error = np.sort(edge.residual(pose))
                projected = view.box(pose)
                distance = np.linalg.norm(center_of(pose, self.model) - center_of(prior, self.model))
                limit = self.jump_limit() if self.frame_guard else 21 + 0.35 * min(self.gap, 20)
                if projected is None or distance > limit:
                    continue
                overlap = box_overlap(box, projected)
                cost = np.mean(error[:round(0.72 * len(error))]) + 0.05 * distance + 3 * (1 - overlap)
                if overlap > 0.35 and cost < 3.2 and (best is None or cost < best[0]):
                    best = cost, pose, view.camera.index, box
        if best is None:
            return None
        _, pose, key, box = best
        x1, y1, x2, y2 = box
        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return pose, key, Instance(contour, 0.35, -1, source="motion")

    def altitude_prior(self):
        if self.altitude is None:
            return None
        return self.altitude, 0.18 + min(self.altitude_age, 160) * 0.0016

    def jump_limit(self):
        travel = self.model.speed / self.fps
        return max(3 * travel, 0.30 * self.model.size) + min(self.gap, 30) * travel

    def accepts_jump(self, step, observations, overlaps):
        """Require independent bearings to accept a large motion correction."""
        limit = self.jump_limit()
        if step <= limit:
            return True
        if step > 2.5 * limit:
            return False
        bearings = [ob.view.ray(ob.center)[1] for ob, quality in zip(observations, overlaps) if quality >= 0.65]
        return any(np.linalg.norm(np.cross(a, b)) >= 0.20 for a, b in combinations(bearings, 2))

    def step(self, frame, cameras, images):
        estimate = self.measure(frame, cameras, images)
        views = [view if self.frame_guard else View.of(view.camera, self.memory[key].mask, self.model)
                 for key, view in self.current_views.items()
                 if self.memory[key].frame == frame and self.memory[key].mask is not None]
        pose = estimate.pose
        if self.output is not None:
            pose = self.output.step(pose, estimate.status, [cameras[key] for key in estimate.views], views)
        line_anchor = None
        if pose is not None and len(estimate.views) == 1:
            key = next(iter(estimate.views))
            memory = self.memory[key]
            if memory.frame == frame and memory.mask is not None:
                gray = memory.history[-1][1]
                refined = refine_range(pose, gray, self.current_views[key], memory.mask.box)
                if estimate.status == "mono" and refined is not pose:
                    line_anchor = key, refined[:3] - pose[:3]
                pose = refined
        accepted = [view for view in views if view.camera.index in estimate.views]
        road_pose = pose
        if self.road_height is not None:
            road_pose = self.road_height.step(frame, pose, estimate.pose, accepted, self.memory)
        corrected = road_pose
        if self.range_recovery is not None:
            corrected = self.range_recovery.step(frame, pose, estimate.pose, road_pose, line_anchor,
                                                 accepted, views, self.memory)
        return Estimate(corrected, estimate.status, estimate.views, estimate.confidence)

    def accept_measurement(self, pose, velocity_mps):
        """Update the next-frame prior using the configured model's pose dimensions."""
        self.pose = pose[:5].copy() if self.model.ground_vehicle else pose.copy()
        self.velocity = np.asarray(velocity_mps, dtype=float) / self.fps

    def prepare_cameras(self, frame, cameras, images, previous):
        """Collect camera health before using any observation for the frame."""
        protected = self.robust or self.adaptive
        self.current_views = {}
        self.camera_ids.update(cameras)
        self.camera_ids.update(images)
        self.camera_health = {}
        for key, mem in list(self.memory.items()):
            if protected and (key not in cameras or key not in images or images[key] is None):
                del self.memory[key]
                continue
            while mem.history and mem.history[0][0] < frame - 2:
                mem.history.popleft()
        for key in self.camera_ids - cameras.keys():
            self.camera_health[key] = "missing" if key not in images or images[key] is None else "uncalibrated"
        prepared = {}
        for key, camera in cameras.items():
            image = images.get(key)
            if image is None or not image.size:
                self.camera_health[key] = "missing"
                self.memory.pop(key, None)
                continue
            if protected and (camera.index != key or image.shape != (camera.height, camera.width, 3)
                                or min(camera.fx, camera.fy) <= 0
                                or not np.isfinite(np.r_[camera.loc, camera.rot, camera.fx, camera.fy,
                                                         camera.cx, camera.cy]).all()):
                self.camera_health[key] = "uncalibrated"
                self.memory.pop(key, None)
                continue
            view = View.of(camera, model=self.model)
            mem = self.memory.setdefault(key, CameraMemory())
            if protected:
                image, gray, mem.exposure, exposure = prepare_image(image, mem.exposure)
                if image is None:
                    self.camera_health[key] = "unusable"
                    self.memory.pop(key, None)
                    continue
                if mem.history:
                    _, old_gray, old_view = mem.history[-1]
                    if old_gray.shape != gray.shape:
                        mem = self.memory[key] = CameraMemory()
                    else:
                        region = old_view.box(previous) if previous is not None else None
                        view = stabilize(view, old_view, gray, old_gray, region)
                if view.calibration_adjusted:
                    self.camera_health[key] = "stabilized"
                elif exposure:
                    self.camera_health[key] = "exposure_corrected"
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.current_views[key] = view
            prepared[key] = image, gray
        if self.adaptive and self.camera_health:
            # Flush detector history contaminated around an observed camera fault.
            self.guard_through = frame + max(2, round(0.2 * self.fps))
        self.frame_guard = self.robust or frame <= self.guard_through
        return prepared

    def measure(self, frame, cameras, images):
        if frame != self.frame + 1:
            raise ValueError("Frame numbers must be consecutive and start at 0")
        self.frame = frame
        previous = self.pose
        prior = None if previous is None else previous.copy()
        if prior is not None:
            prior[:3] += self.velocity
        heading = 0.0 if prior is None else prior[3]
        grade = 0.0 if prior is None else prior[4]
        observations, grays, flows, proposals = [], {}, {}, []
        prepared = self.prepare_cameras(frame, cameras, images, previous)
        if self.video_masks is not None:
            self.video_masks.discard_missing(frame, prepared)
        for key, (image, gray) in prepared.items():
            view = self.current_views[key]
            camera = view.camera
            mem = self.memory[key]
            expected = view.box(prior) if prior is not None else None
            grays[key] = gray
            carried = None
            if mem.mask is not None and mem.frame == frame - 1 and mem.history:
                carried = transport(mem.history[-1][1], gray, mem.mask)
                if carried is not None:
                    flows[key] = (view, mem.history[-1][2], carried)
            candidates, proposal = self.candidates(key, image, gray, view, expected, carried)
            if self.video_masks is not None:
                seed = candidates[0] if candidates else None
                tracked_mask = self.video_masks.step(frame, key, image, seed)
                if tracked_mask is not None:
                    candidates = [tracked_mask, *candidates]
            if self.frame_guard and expected is not None and (self.confidence >= 0.45 or self.gap <= 4):
                original = candidates
                candidates = [item for item in candidates if plausible_box(item.box, expected, self.gap)]
                if original and not candidates:
                    self.camera_health[key] = "outlier"
                if proposal is not None and not plausible_box(proposal, expected, self.gap):
                    proposal = None
                if carried is not None and not plausible_box(carried.box, expected, self.gap):
                    flows.pop(key, None)
            if proposal is not None:
                proposals.append((view, gray, proposal))
            if candidates:
                view.instance = candidates[0]
                movement = None
                if mem.history:
                    _, old_gray, old_view = mem.history[0]
                    if np.linalg.norm(camera.loc - old_view.camera.loc) <= 5:
                        warp = camera_warp(view, old_view) if not np.allclose(view.matrix, old_view.matrix) else None
                        movement = contour_motion(old_gray, gray, view.instance.contour, warp)
                observations.append(Observation(view, movement))
                if not self.frame_guard:
                    mem.mask, mem.frame = view.instance, frame
                    if not view.instance.flow:
                        mem.detected = frame
            mem.history.append((frame, gray, view))
            if mem.frame < frame - 2:
                mem.mask = None

        used = []
        quality = 0.0
        strong = max(observations, key=lambda ob: min(ob.size) / ob.view.camera.fpx, default=None)
        detailed_fit = None
        if self.model.ground_vehicle and len(observations) > 1 and strong is not None and detailed(strong):
            mono_pose, mono_overlap = fit([strong], prior, heading, grade, self.gap,
                                            self.confidence, None, grays, self.velocity)
            if mono_overlap[0] > 0.73:
                detailed_fit = mono_pose, mono_overlap
                strong_angle = min(strong.size) / strong.view.camera.fpx
                complete = [strong]
                for observation in observations:
                    if observation is strong:
                        continue
                    angle = min(observation.size) / observation.view.camera.fpx
                    projected = observation.view.box(mono_pose)
                    partial = (projected is not None and
                               min(observation.size / np.maximum(projected[2:] - projected[:2], 1)) < 0.60)
                    if strong_angle < 2.4 * angle or not partial:
                        complete.append(observation)
                observations = complete

        while observations:
            pose, overlaps = fit(observations, prior, heading, grade, self.gap, self.confidence, self.altitude_prior(), grays, self.velocity)
            if detailed_fit is not None and len(observations) > 1:
                mono_pose, mono_overlap = detailed_fit
                delta = np.linalg.norm(center_of(pose, self.model) - center_of(mono_pose, self.model))
                strong_loss = mono_overlap[0] - strong.overlap(pose)
                competing_support = max(min(ob.size) / ob.view.camera.fpx
                                        for ob in observations if ob is not strong)
                dominant = min(strong.size) / strong.view.camera.fpx > 2.4 * competing_support
                if delta > 3 and strong_loss > (0.035 if dominant else 0.075):
                    pose, overlaps = mono_pose, mono_overlap
                    observations = [strong]
            step = 0 if prior is None else np.linalg.norm(center_of(pose, self.model) - center_of(prior, self.model))
            anchored = (len(observations) == 1 and self.altitude is not None
                        and observations[0].view.ray(observations[0].center)[1][2] < -0.12)
            supported_jump = (len(observations) > 1 and min(overlaps) > 0.65
                              or (anchored or detailed(observations[0])) and min(overlaps) > 0.65 and step < 37)
            direction = np.mean([ob.direction(pose) for ob in observations]) if self.model.ground_vehicle else 0.0
            turn_fix = (self.model.ground_vehicle and prior is not None and abs(angle_diff(pose[3], prior[3])) > 1.9
                        and direction > 0.22 and min(overlaps) > 0.55 and step < 35)
            reasonable = step < 6.0 + min(self.gap, 30) * 1.05 or supported_jump or turn_fix
            if self.frame_guard:
                reasonable = reasonable and self.accepts_jump(step, observations, overlaps)
            if reasonable and np.mean(overlaps) >= 0.51 and np.min(overlaps) >= 0.40:
                used = observations
                quality = float(np.mean(overlaps))
                break
            if len(observations) == 1:
                break
            observations.pop(int(np.argmin(overlaps)))

        if self.frame_guard:
            accepted_keys = {ob.view.camera.index for ob in used}
            for key, view in self.current_views.items():
                if key in accepted_keys:
                    mem = self.memory[key]
                    mem.mask, mem.frame = view.instance, frame
                    if not view.instance.flow:
                        mem.detected = frame
                elif view.instance is not None:
                    view.instance = None
                    self.camera_health[key] = "outlier"
        if not used and prior is not None and proposals:
            fallback = self.motion_pose(proposals, prior)
            if fallback is not None:
                self.pose, key, instance = fallback
                change = center_of(self.pose, self.model) - center_of(previous, self.model)
                if np.linalg.norm(change) < 2:
                    self.velocity = 0.90 * self.velocity + 0.10 * change
                else:
                    self.velocity *= 0.7
                self.gap = 0
                self.confidence = 0.35
                return Estimate(self.pose.copy(), "motion", {key: instance}, self.confidence)
        if not used and prior is not None and flows:
            fallback = self.flow_pose(flows, prior, previous)
            if fallback is not None:
                self.pose, key, instance = fallback
                self.velocity = 0.84 * self.velocity + 0.16 * (center_of(self.pose, self.model) - center_of(previous, self.model))
                mem = self.memory[key]
                mem.mask, mem.frame = instance, frame
                if self.frame_guard:
                    self.current_views[key].instance = instance
                self.gap = 0
                self.confidence = min(self.confidence, 0.55)
                return Estimate(self.pose.copy(), "flow", {key: instance}, self.confidence)
        if not used:
            self.gap += 1
            self.confidence *= 0.93
            if prior is None or self.gap > round(2.0 * self.fps):
                self.pose = None
                self.velocity[:] = 0
                self.altitude = None
                return Estimate(None, "lost", {}, 0.0)
            self.pose = prior
            if self.gap > 20:
                self.velocity *= 0.98
            return Estimate(self.pose.copy(), "forecast", {}, self.confidence)

        if previous is not None:
            correction = center_of(pose, self.model) - center_of(prior, self.model)
            reacquired = np.linalg.norm(correction) > 6.5
            if not reacquired:
                pose = with_center(pose, center_of(prior, self.model) + correction * (0.86 if len(used) > 1 else 0.82), self.model)
            velocity = self.velocity + correction * (0.075 if len(used) > 1 else 0.035)
            max_speed = self.model.speed / self.fps
            if np.linalg.norm(velocity) > max_speed:
                velocity *= max_speed / np.linalg.norm(velocity)
            self.velocity = velocity
            if reacquired:
                self.velocity *= 0.25
            if self.model.ground_vehicle and np.linalg.norm(velocity[:2]) > 5 / self.fps:
                course = np.arctan2(velocity[1], velocity[0])
                change = angle_diff(course, pose[3])
                if abs(change) < 1.2:
                    center = center_of(pose, self.model)
                    pose[3] += (0.065 if len(used) == 1 else 0.05) * change
                    pose = with_center(pose, center, self.model)
            if self.model.ground_vehicle and len(used) == 1:
                self.velocity[2] *= 0.86

        self.gap = 0
        if self.road_height is not None:
            if self.altitude is None:
                self.altitude = pose[2]
                self.altitude_age = 0
            elif quality >= 0.74 and (
                (len(used) >= 2 and np.min(overlaps) >= 0.64)
                or (detailed(used[0]) and abs(pose[2] - self.altitude) < 1.2)
            ):
                self.altitude = 0.74 * self.altitude + 0.26 * pose[2]
                self.altitude_age = 0
            else:
                self.altitude_age += 1
        self.pose = pose
        self.acquired = True
        self.confidence = quality
        views = {ob.view.camera.index: ob.view.instance for ob in used}
        return Estimate(pose.copy(), "multi" if len(used) > 1 else "mono", views, quality)
