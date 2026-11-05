"""Verify a second detector before filling a missing pose."""
from collections import deque
from dataclasses import replace
from itertools import combinations

import numpy as np

from online.estimation.cad_points import well_conditioned_stereo
from online.geometry import (Observation, View, center_of, fit, intersects, linear_seed,
                             optimize, with_center)
from online.tracker import Estimate, box_overlap


class VisibleObservation(Observation):
    def overlap(self, pose):
        return super().overlap(pose, visible_only=True)


class StereoRecovery:
    def __init__(self, segmenter, model, fps, prompt=None, min_output_iou=0.75):
        self.segmenter = segmenter
        self.prompt = prompt
        self.model = model
        self.min_output_iou = min_output_iou
        self.step_limit = max(0.6, 1.6 * model.speed / fps)
        self.last_output = None
        self.last_pose = None
        self.history = deque()
        self.window = max(6, round(fps / 2))
        self.max_evidence_gap = max(2, round(fps / 7))
        self.streak = 0

    def clear(self):
        self.last_output = None
        self.last_pose = None
        self.history.clear()
        self.streak = 0

    def groups(self, primary, cameras, images):
        windows = [(key, images[key], None) for key in cameras]
        detections = self.segmenter.detect(windows, 960, 0.07)
        prior = self.last_pose
        if prior is not None:
            boxes = {}
            for key, camera in cameras.items():
                box = View.of(camera, model=self.model).box(prior)
                if (intersects(box, camera.width, camera.height) and min(box[2:] - box[:2]) >= 12
                        and not any(box_overlap(item.box, box) >= 0.40 for item in detections.get(key, ()))):
                    boxes[key] = box
            for key, instances in self.segmenter.regions(images, boxes).items():
                detections[key] = [*instances, *detections.get(key, ())]
        elif detections:
            steep = {key: images[key] for key, camera in cameras.items()
                     if camera.forward[2] < -0.65 and not any(
                         item.score >= 0.30 for item in detections.get(key, ()))}
            for key, instances in self.segmenter.overhead(steep).items():
                detections[key] = [*detections.get(key, ()), *instances]
        groups = {}
        for key, camera in cameras.items():
            items = sorted(detections.get(key, ()), key=lambda instance: instance.score, reverse=True)
            tracked = primary.tracked_2d.get(key)
            if (tracked is not None and tracked.source != "sam-video-local"
                    and any(box_overlap(tracked.box, item.box) >= 0.55 for item in items)):
                items = [*items, tracked]
            observations = [VisibleObservation(View.of(camera, item, self.model)) for item in items[:3]
                            if min(item.box[2:] - item.box[:2]) >= 12]
            if observations:
                groups[key] = observations
        return groups

    def velocity(self):
        if len(self.history) < 4:
            return None
        slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in combinations(self.history, 2)
                  if b[0] - a[0] >= 4]
        if len(slopes) < 3:
            return None
        velocity = np.median(slopes, axis=0)
        speed = np.linalg.norm(velocity[:2])
        age = self.history[-1][0] - self.history[0][0]
        if speed * age < 1.5 or speed > self.step_limit:
            return None
        consensus = np.asarray(slopes)[:, :2] @ velocity[:2] >= 0.35 * speed**2
        return velocity if np.mean(consensus) >= 0.75 else None

    @staticmethod
    def support(pose, groups):
        support = []
        for observations in groups.values():
            best = None
            for observation in observations:
                view = observation.view
                box = view.box(pose)
                if box is None:
                    continue
                visible = np.clip(box, [0, 0, 0, 0],
                                  [view.camera.width - 1, view.camera.height - 1] * 2)
                if box_overlap(visible, observation.box) < 0.40:
                    continue
                overlap = observation.overlap(pose)
                if overlap >= 0.60 and (best is None or overlap > best[1]):
                    best = observation, overlap
            if best is not None:
                support.append(best)
        return support

    def fit(self, frame, groups):
        best = None
        velocity = self.velocity()
        for a, b in combinations(groups.values(), 2):
            for first in a:
                for second in b:
                    pair = [first, second]
                    scores = [ob.view.instance.score for ob in pair]
                    if (max(scores) < 0.35 or first.clipped and second.clipped
                            or linear_seed(pair) is None):
                        continue
                    pose, _ = fit(pair, None, 0.0, 0.0, 0, 0.8, velocity=velocity)
                    support = self.support(pose, groups)
                    if len(support) < 2 or not well_conditioned_stereo(
                            pose, [ob for ob, _ in support], self.model):
                        continue
                    center = center_of(pose, self.model)
                    if self.history:
                        previous_frame, previous_center = self.history[-1]
                        if np.linalg.norm(center - previous_center) > self.step_limit * (frame - previous_frame):
                            continue
                    merit = sum(overlap * np.sqrt(ob.view.instance.score) for ob, overlap in support)
                    if best is None or merit > best[0]:
                        best = merit, pose, support
        return (None, []) if best is None else best[1:]

    def align_with_course(self, pose, support, velocity, groups):
        if not self.model.ground_vehicle:
            return pose, support
        if velocity is None:
            return None, []
        heading = float(np.arctan2(velocity[1], velocity[0]))
        if np.cos(heading - pose[3]) >= 0.60:
            return pose, support
        center = center_of(pose, self.model)
        seed = with_center(np.r_[pose[:3], heading, pose[4]], center, self.model)
        oriented, _ = optimize([ob for ob, _ in support], seed, None, heading, pose[4], None,
                               yaw_window=0.9, model=self.model)
        checked = self.support(oriented, groups)
        if (len(checked) < 2 or np.linalg.norm(center_of(oriented, self.model) - center) > 1.5
                or not well_conditioned_stereo(oriented, [ob for ob, _ in checked], self.model)):
            return None, []
        return oriented, checked

    def refine_prompt(self, pose, support, images):
        if self.prompt is None or not self.model.ground_vehicle:
            return pose, support
        revised = []
        for ob, _ in support:
            camera = ob.view.camera
            box = np.clip(ob.view.box(pose), [0, 0, 0, 0], [camera.width-1, camera.height-1] * 2)
            mask = self.prompt.search(images[camera.index], box, precise=True)
            if mask is not None and mask.score >= 0.75 and box_overlap(mask.box, ob.box) >= 0.50:
                revised.append(VisibleObservation(View.of(camera, replace(
                    ob.view.instance, contour=mask.contour), self.model)))
            else:
                revised.append(ob)
        if all(a is b for a, (b, _) in zip(revised, support)):
            return pose, support
        candidate, _ = optimize(revised, pose, pose, pose[3], pose[4],
                                [1.5, 1.5, 0.8], yaw_window=0.25, model=self.model)
        quality = [ob.overlap(candidate) for ob in revised]
        if (np.linalg.norm(candidate[:3] - pose[:3]) > 1.25 or min(quality) < 0.75
                or np.mean(quality) < np.mean([ob.overlap(pose) for ob in revised]) + 0.02
                or any(ob.overlap(candidate) < before - 0.08 for ob, before in support)):
            return pose, support
        return candidate, list(zip(revised, quality))

    def step(self, frame, primary, cameras, images):
        if primary.pose is not None:
            self.last_output = frame, primary.pose[:3].copy()
            self.last_pose = None
            self.history.clear()
            self.streak = 0
            return primary

        while self.history and frame - self.history[0][0] > self.window:
            self.history.popleft()
        if self.history and frame - self.history[-1][0] > self.max_evidence_gap:
            self.history.clear()
            self.last_pose = None
        groups = self.groups(primary, cameras, images)
        pose, support = self.fit(frame, groups)
        if pose is None:
            self.streak = 0
            return primary

        self.history.append((frame, center_of(pose, self.model)))
        self.streak += 1
        self.last_pose = pose
        if self.streak < 2:
            return primary
        pose, support = self.align_with_course(pose, support, self.velocity(), groups)
        if pose is None:
            return primary
        self.last_pose = pose
        support = [(observation, score) for observation, score in support if score >= self.min_output_iou]
        if len(support) < 2 or not well_conditioned_stereo(
                pose, [observation for observation, _ in support], self.model):
            return primary
        pose, support = self.refine_prompt(pose, support, images)
        if self.last_output is not None:
            last_frame, xyz = self.last_output
            if np.linalg.norm(pose[:3] - xyz) > self.step_limit * (frame - last_frame):
                return primary

        self.last_pose = pose
        quality = min(overlap for _, overlap in support)
        self.last_output = frame, pose[:3].copy()
        cameras_used = {ob.view.camera.index: ob.view.instance for ob, _ in support}
        return Estimate(pose, "stereo_recovery", cameras_used, quality, primary.tracked_2d)
