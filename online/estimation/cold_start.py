"""Acquire a directed pose from the target's visible CAD reference part."""
from dataclasses import replace

import numpy as np

from online.estimation.cad_points import well_conditioned_stereo
from online.estimation.stereo_recovery import StereoRecovery, VisibleObservation
from online.geometry import View, angle_diff, center_of, optimize, with_center
from online.tracker import Estimate


class ColdStartRecovery:
    def __init__(self, detector, prompt, model, fps):
        self.stereo = StereoRecovery(detector, model, fps, prompt)
        self.prompt = prompt
        self.model = model
        self.part = replace(model, parts=(model.reference_part,))
        self.limit = max(0.5, 2 * model.speed / fps)
        self.max_age = max(1, round(fps / 2))
        self.deadline = round(4 * fps)
        self.pose = None
        self.velocity = np.zeros(3)
        self.cameras = ()
        self.last_measurement = None
        self.finished = False

    def part_observations(self, pose, cameras, images, keys, threshold):
        result = []
        for key in keys:
            if key not in cameras:
                continue
            camera = cameras[key]
            view = View.of(camera, model=self.part)
            box = view.box(pose)
            if (box is None or min(box[2:] - box[:2]) < 15 or np.any(box[:2] < 2)
                    or np.any(box[2:] > [camera.width - 3, camera.height - 3])):
                continue
            mask = self.prompt.search(images[key], box, precise=True)
            if mask is None or mask.score < 0.75:
                continue
            observation = VisibleObservation(View.of(camera, mask, self.part))
            overlap = observation.overlap(pose)
            if overlap >= threshold:
                result.append((observation, overlap))
        return result

    def refine_part(self, pose, support, shift_limit):
        observations = [observation for observation, _ in support]
        candidate, _ = optimize(observations, pose, pose, pose[3], pose[4], [1.2, 1.2, 1.0],
                                yaw_window=0.30, model=self.part)
        overlaps = [observation.overlap(candidate) for observation in observations]
        if (np.linalg.norm(candidate[:3] - pose[:3]) > shift_limit or min(overlaps) < 0.65
                or not well_conditioned_stereo(candidate, observations, self.part)):
            return None, []
        return candidate, list(zip(observations, overlaps))

    def acquire(self, frame, primary, cameras, images):
        groups = self.stereo.groups(primary, cameras, images)
        pose, support = self.stereo.fit(frame, groups)
        if pose is None:
            return None, []
        center = center_of(pose, self.model)
        yaw = pose[3] + np.pi
        mirror = with_center(np.r_[pose[:3], yaw, -pose[4]], center, self.model)
        mirror, _ = optimize([ob for ob, _ in support], mirror, None, yaw, -pose[4], None,
                             yaw_window=0.7, model=self.model)
        hypotheses = []
        for candidate in (pose, mirror):
            whole = self.stereo.support(candidate, groups)
            keys = [ob.view.camera.index for ob, _ in whole]
            part = self.part_observations(candidate, cameras, images, keys, 0.70)
            if len(part) >= 2 and well_conditioned_stereo(candidate, [ob for ob, _ in part], self.part):
                quality = sum(overlap for _, overlap in part)
                hypotheses.append((quality, candidate, part))
        hypotheses.sort(key=lambda item: item[0], reverse=True)
        if not hypotheses or len(hypotheses) > 1 and hypotheses[0][0] - hypotheses[1][0] < 0.15:
            return None, []
        _, pose, support = hypotheses[0]
        return self.refine_part(pose, support, 1.5)

    def follow(self, frame, primary, predicted, cameras, images):
        self.stereo.last_pose = predicted
        groups = self.stereo.groups(primary, cameras, images)
        for key in self.cameras:
            if key not in cameras:
                continue
            camera = cameras[key]
            view = View.of(camera, model=self.model)
            box = view.box(predicted)
            if box is None:
                continue
            box = np.clip(box, [0, 0, 0, 0], [camera.width - 1, camera.height - 1] * 2)
            if min(box[2:] - box[:2]) < 15:
                continue
            mask = self.prompt.search(images[key], box, precise=True)
            if mask is None or mask.score < 0.75:
                continue
            observation = VisibleObservation(View.of(camera, mask, self.model))
            if observation.overlap(predicted) >= 0.55:
                groups.setdefault(key, []).append(observation)
        support = self.stereo.support(predicted, groups)
        if len(support) < 2:
            return None, []
        observations = [observation for observation, _ in support]
        pose, _ = optimize(observations, predicted, predicted, predicted[3], predicted[4],
                           [1.6, 1.6, 1.0], yaw_window=0.5, model=self.model)
        checked = self.stereo.support(pose, groups)
        if (len(checked) < 2 or np.linalg.norm(pose[:3] - predicted[:3]) > self.limit
                or not well_conditioned_stereo(pose, [ob for ob, _ in checked], self.model)):
            return None, []
        return pose, checked

    def step(self, frame, primary, cameras, images):
        if self.finished:
            return primary
        if frame >= self.deadline:
            self.finished = True
            return primary
        if self.pose is None:
            if self.last_measurement is None and primary.pose is not None:
                self.finished = True
                return primary
            self.stereo.last_pose = None
            pose, support = self.acquire(frame, primary, cameras, images)
        else:
            predicted = self.pose.copy()
            predicted[:3] += self.velocity
            if (primary.pose is not None and primary.status == "multi"
                    and np.linalg.norm(primary.pose[:3] - predicted[:3]) <= self.limit
                    and abs(angle_diff(primary.pose[3], predicted[3])) < 0.75):
                self.finished = True
                return primary
            keys = set(self.cameras) | set(primary.views)
            support = self.part_observations(predicted, cameras, images, sorted(keys), 0.50)
            pose, support = self.refine_part(predicted, support, self.limit) if len(support) >= 2 else (None, [])
            if pose is None:
                pose, support = self.follow(frame, primary, predicted, cameras, images)
        if pose is None:
            if self.pose is None or frame - self.last_measurement > self.max_age:
                self.pose = None
                self.velocity.fill(0)
                return primary
            self.pose = predicted
            return Estimate(predicted.copy(), "startup_prediction", {}, 0.0, primary.tracked_2d)

        if self.pose is not None:
            move = pose[:3] - self.pose[:3]
            move *= min(1., self.limit / max(np.linalg.norm(move), 1e-9))
            pose[:3] = self.pose[:3] + move
            self.velocity += 0.4 * (move - self.velocity)
        self.pose = pose.copy()
        self.cameras = tuple(ob.view.camera.index for ob, _ in support)
        self.last_measurement = frame
        views = {ob.view.camera.index: ob.view.instance for ob, _ in support}
        return Estimate(pose, "stereo_recovery", views, min(overlap for _, overlap in support), primary.tracked_2d)
