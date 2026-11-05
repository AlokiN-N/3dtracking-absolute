"""Refine a world pose from simultaneous CAD-to-pixel correspondences."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from online.geometry import Rot, center_of, pose_angles
from online.estimation.kinematics import overlap
from online.perception.vision import transport


@dataclass
class CameraMatches:
    observation: object
    sigma: float
    choices: list

    @property
    def view(self):
        return self.observation.view


def point_split(points, pixels, box=None):
    if box is None:
        holdout = np.arange(len(points)) % 4 == 0
    else:
        from online.estimation.surface_pose import spatial_partition

        holdout = spatial_partition(pixels, box)
    data = np.c_[points, pixels]

    def sample(values, limit):
        return values[np.linspace(0, len(values) - 1, limit, dtype=int)] if len(values) > limit else values

    return sample(data[~holdout], 200), sample(data[holdout], 100)


def score(pose, camera, points):
    uv, depth = camera.view.project(points[:, :3] @ Rot(*pose[3:]).T + pose[:3])
    errors = np.linalg.norm(uv - points[:, 3:], axis=1) / camera.sigma
    errors[depth <= 0.05] = np.inf
    return np.mean(np.minimum(errors, 5)), np.median(errors), np.mean(errors < 2)


def refine_joint(baseline, observations, spatial_check=False):
    """Return a correction only when independent views support it on held-out points."""
    if len(observations) < 2:
        return baseline
    model = observations[0][0].view.model
    baseline_rotation = Rot(*baseline[3:])
    cameras = []
    seeds = [baseline]
    for observation, hypotheses in observations:
        sigma = max(1.5, 0.010 * observation.size.max())
        choices = []
        for match in hypotheses:
            if match is None or match["heldout_fraction"] < 0.46 or match["heldout_median"] > 3 * sigma:
                continue
            pose = match["pose"]
            if np.linalg.norm(pose[:3] - baseline[:3]) > 0.60 * model.size:
                continue
            if np.trace(Rot(*pose[3:]).T @ baseline_rotation) < 1 + 2 * np.cos(0.65):
                continue
            box = observation.box if spatial_check else None
            train, holdout = point_split(match["model_points"], match["image_points"], box)
            if min(len(train), len(holdout)) >= 12:
                choices.append((train, holdout))
                seeds.append(pose)
        if choices:
            cameras.append(CameraMatches(observation, sigma, choices))
    bearings = [camera.view.ray(camera.observation.center)[1] for camera in cameras]
    if not any(np.linalg.norm(np.cross(a, b)) >= 0.15 for a, b in combinations(bearings, 2)):
        return baseline

    def assignment(seed):
        return [min(camera.choices, key=lambda pair: score(seed, camera, pair[0])[0]) for camera in cameras]

    def fit_cost(seed):
        return np.mean([score(seed, camera, train)[0] for camera, (train, _) in zip(cameras, assignment(seed))])

    seed = min(seeds, key=fit_cost)
    selected = list(zip(cameras, assignment(seed)))
    rotation = Rot(*seed[3:])
    pivot = seed[:3] + rotation @ model.center

    def transform(delta):
        matrix = Rotation.from_rotvec(delta[3:]).as_matrix() @ rotation
        return matrix, pivot + delta[:3] - matrix @ model.center

    def residual(delta, partition, prior=True):
        matrix, origin = transform(delta)
        errors = []
        for camera, pair in selected:
            points = pair[partition]
            uv, depth = camera.view.project(points[:, :3] @ matrix.T + origin)
            error = (uv - points[:, 3:]) * (8 / (camera.sigma * np.sqrt(len(points))))
            error[depth <= 0.05] = 25
            errors.extend(error.ravel())
        if prior:
            errors.extend(delta[:3] / (0.30 * model.size))
            errors.extend(delta[3:] / 0.25)
        return np.asarray(errors)

    def fit_partition(partition):
        fit = least_squares(residual, np.zeros(6), args=(partition,), loss="soft_l1", f_scale=1.6,
                            max_nfev=30, x_scale=[max(0.05, model.size * 0.1)] * 3 + [0.1] * 3)
        return fit, *transform(fit.x)

    before = np.asarray([score(baseline, camera, holdout) for camera, (_, holdout) in selected])
    if np.mean(before[:, 1]) < 1.4:
        return baseline
    fit, matrix, origin = fit_partition(0)
    pose = np.r_[origin, pose_angles(matrix)]
    if np.linalg.norm(origin - baseline[:3]) > 0.45 * model.size:
        return baseline

    after = np.asarray([score(pose, camera, holdout) for camera, (_, holdout) in selected])
    gain = before[:, 0] - after[:, 0]
    if gain.min() < 0.35 or gain.mean() < 0.60 or after[:, 2].min() < 0.65:
        return baseline
    if any(camera.observation.overlap(pose) < camera.observation.overlap(baseline) - 0.035
           for camera in cameras):
        return baseline
    if spatial_check:
        matched = {camera.view.camera.index for camera in cameras}
        for observation, _ in observations:
            if observation.view.camera.index not in matched and not observation.view.instance.flow:
                overlap_before = observation.overlap(baseline)
                if overlap_before >= 0.6 and observation.overlap(pose) < overlap_before - 0.035:
                    return baseline
    # Agreement in pixels can hide unstable depth. Fit the disjoint point
    # subset independently and check that it gives the same world pose.
    check, check_matrix, check_origin = fit_partition(1)
    if np.linalg.norm(origin - check_origin) > 0.035 * model.size:
        return baseline
    if np.trace(matrix.T @ check_matrix) < 1 + 2 * np.cos(0.12):
        return baseline
    if spatial_check:
        # The pose prior is not image evidence. Propagate pixel residuals
        # through each spatial split without the prior's rows in the Jacobian.
        for partition, solution in enumerate((fit, check)):
            jac = solution.jac[:-6]
            _, singular, vectors = np.linalg.svd(jac, full_matrices=False)
            if singular[-1] < 1e-7:
                return baseline
            position_sigma = np.linalg.norm(vectors[:, :3] / singular[:, None], axis=0)
            noise = max(1.0, 1.48 * np.median(np.abs(residual(solution.x, partition, prior=False))))
            if np.max(position_sigma) * noise > 0.10 * model.size:
                return baseline
    return pose


@dataclass
class MotionEvidence:
    view: object
    predicted: np.ndarray
    before: float
    sigma: float


def motion_evidence(frame, previous, baseline, views, memories):
    """Compute the current frame's independent motion evidence once."""
    if previous is None:
        return []
    evidence = []
    for view in views:
        history = memories[view.camera.index].history
        if len(history) != 2 or view.instance is None:
            continue
        old_frame, old_gray, old_view = history[0]
        current_frame, gray, _ = history[1]
        if old_frame != frame - 1 or current_frame != frame or old_view.instance is None:
            continue
        old_box = old_view.instance.box
        pixel, depth = old_view.project(center_of(previous, view.model)[None])
        if depth[0] <= 0.05 or np.any(pixel[0] < old_box[:2]) or np.any(pixel[0] > old_box[2:]):
            continue
        box = view.instance.box
        pixels, depths = view.project(center_of(baseline, view.model)[None])
        if depths[0] <= 0.05:
            continue
        tracked = transport(old_gray, gray, old_view.instance)
        if tracked is None or overlap(tracked.box, box) < 0.82:
            continue
        predicted = tracked.affine @ np.r_[pixel[0], 1.0]
        before = float(np.linalg.norm(pixels[0] - predicted))
        sigma = max(1, 0.009 * min(box[2:] - box[:2]))
        evidence.append(MotionEvidence(view, predicted, before, sigma))
    return evidence


def motion_supports(frame, previous, baseline, candidate, views, memories, evidence=None):
    """Check the CAD center's displacement against independent camera flow."""
    if evidence is None:
        evidence = motion_evidence(frame, previous, baseline, views, memories)
    gains = []
    for item in evidence:
        view = item.view
        pixels, depths = view.project(center_of(candidate, view.model)[None])
        box = view.instance.box
        if depths[0] <= 0.05 or np.any(pixels[0] < box[:2]) or np.any(pixels[0] > box[2:]):
            continue
        after = np.linalg.norm(pixels[0] - item.predicted)
        gains.append((item.before - after) / item.sigma)
    return sum(gain >= 0.5 for gain in gains) >= 2 and min(gains) >= -0.5
