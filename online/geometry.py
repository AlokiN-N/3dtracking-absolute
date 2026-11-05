"""Fit a metric CAD silhouette against simultaneous observations."""
from dataclasses import dataclass
from itertools import combinations, permutations, product

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from core.geometry import triangulate
from online.target import way1_target

DEFAULT_TARGET = way1_target()
ANGLES = np.arange(12) * (2 * np.pi / 12)
DIRECTIONS = np.array([np.cos(ANGLES), np.sin(ANGLES)]).T


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def Rot(yaw, pitch, roll=0.0):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    forward = np.array([cp * cy, cp * sy, sp])
    right = np.array([-sy, cy, 0.0])
    up = np.array([-sp * cy, -sp * sy, cp])
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        right, up = c * right + s * up, c * up - s * right
    return np.stack([forward, right, up], axis=1)


def center_of(pose, model=DEFAULT_TARGET):
    return pose[:3] + Rot(*pose[3:]) @ model.center


def with_center(pose, center, model=DEFAULT_TARGET):
    result = pose.copy()
    result[:3] = center - Rot(*result[3:]) @ model.center
    return result


def angle_diff(a, b):
    return (a - b + np.pi) % (2*np.pi) - np.pi


@dataclass
class View:
    camera: object
    matrix: np.ndarray
    instance: object
    model: object = DEFAULT_TARGET
    calibration_adjusted: bool = False

    @classmethod
    def of(cls, camera, instance=None, model=DEFAULT_TARGET):
        forward, right, down = camera.basis
        return cls(camera, np.stack([right, down, forward]), instance, model)

    def project(self, points):
        xyz = (points - self.camera.loc / 100) @ self.matrix.T
        z = xyz[:, 2]
        uv = xyz[:, :2] / np.maximum(z[:, None], 0.05)
        uv *= [self.camera.fx, self.camera.fy]
        uv += [self.camera.cx, self.camera.cy]
        return uv, z

    def box(self, pose):
        uv, depth = self.project(self.model.vertices @ Rot(*pose[3:]).T + pose[:3])
        if depth.min() < (1 if self.model.ground_vehicle else 0.06):
            return None
        return np.r_[uv.min(axis=0), uv.max(axis=0)]

    def ray(self, uv):
        if self.calibration_adjusted:
            cam = self.camera
            local = np.array([(uv[0] - cam.cx) / cam.fx, (uv[1] - cam.cy) / cam.fy, 1.0])
            direction = self.matrix.T @ local
            return cam.loc / 100, direction / np.linalg.norm(direction)
        origin, direction = self.camera.ray(uv)
        return origin / 100, direction


@dataclass
class Observation:
    view: View
    movement: np.ndarray | None = None

    def __post_init__(self):
        item = self.view.instance
        self.box = item.box
        self.center = (self.box[:2] + self.box[2:]) / 2
        self.size = self.box[2:] - self.box[:2]
        self.target = (item.contour @ DIRECTIONS.T).max(axis=0)
        self.sigma = max(0.8, 0.013 * self.size.max()) * (1.3 if item.flow else 1.0)
        shape = [self.view.camera.width - 1, self.view.camera.height - 1]
        self.clipped = np.any(self.box[:2] < 4) or np.any(self.box[2:] > np.array(shape) - 4)
        self.weight = 0.70 if self.clipped else 1.0
        margin = 8
        self.origin = np.floor(self.box[:2]).astype(int) - margin
        self.ratio = min(1.0, 150 / max(self.size))
        width, height = np.ceil((self.size + 2 * margin + 1) * self.ratio).astype(int)
        self.mask = np.zeros((height, width), np.uint8)
        poly = np.round((item.contour - self.origin) * self.ratio).astype(np.int32)
        cv2.fillPoly(self.mask, [poly], 1)

    def direction(self, pose):
        if self.movement is None:
            return 0.0
        magnitude = np.linalg.norm(self.movement)
        if magnitude < 1.3:
            return 0.0
        rotation = Rot(*pose[3:])
        center = pose[:3] + rotation @ self.view.model.center
        uv, depth = self.view.project(np.stack([center, center + 4 * rotation[:, 0]]))
        forward = uv[1] - uv[0]
        span = np.linalg.norm(forward)
        if depth.min() < 1 or span < 2:
            return 0.0
        confidence = min(1, magnitude / 4) * min(1, span / 12)
        return float(confidence * (forward @ self.movement) / (span * magnitude))

    def overlap(self, pose, *, visible_only=False):
        matrix = Rot(*pose[3:]).T
        rendered = np.zeros_like(self.mask)
        pixels, depth = self.view.project(self.view.model.vertices @ matrix + pose[:3])
        if depth.min() <= (1 if self.view.model.ground_vehicle else 0.06):
            return 0.0
        i = 0
        for vertices in self.view.model.parts:
            q = (pixels[i:i + len(vertices)] - self.origin) * self.ratio
            i += len(vertices)
            hull = cv2.convexHull(np.clip(np.round(q), -10000, 10000).astype(np.int32))
            cv2.fillConvexPoly(rendered, hull, 1)
        if visible_only and self.clipped:
            camera = self.view.camera
            xs = np.arange(rendered.shape[1]) / self.ratio + self.origin[0]
            ys = np.arange(rendered.shape[0]) / self.ratio + self.origin[1]
            rendered[:, (xs < 0) | (xs > camera.width - 1)] = 0
            rendered[(ys < 0) | (ys > camera.height - 1), :] = 0
        match = np.count_nonzero(rendered & self.mask)
        return match / max(np.count_nonzero(rendered | self.mask), 1)


def linear_seed(obs):
    rays = [o.view.ray(o.center) for o in obs]
    model = obs[0].view.model
    near = 3 if model.ground_vehicle else 0.06
    candidates = []
    for a, b in combinations(rays, 2):
        if np.linalg.norm(np.cross(a[1], b[1])) < 0.10:
            continue
        center, _ = triangulate([a[0], b[0]], [a[1], b[1]], [1, 1])
        if any(np.dot(center - o, d) < near for o, d in (a, b)):
            continue
        distances = [np.linalg.norm(np.cross(center - o, d)) for o, d in rays]
        candidates.append((np.sum(np.minimum(distances, 15)), center))
    return min(candidates, key=lambda entry: entry[0])[1] if candidates else None


def mono_seed(obs, yaw, pitch):
    matrix = Rot(yaw, pitch).T
    model = obs.view.model
    vertices = model.vertices @ matrix
    span = np.ptp(vertices @ obs.view.matrix[:2].T, axis=0)
    focal = np.array([obs.view.camera.fx, obs.view.camera.fy])
    depth = (span * focal @ obs.size) / (obs.size @ obs.size)
    depth = np.clip(depth, 8, 1800)
    origin, direction = obs.view.ray(obs.center)
    xyz = origin + direction * (depth / (direction @ obs.view.matrix[2]))
    return np.r_[xyz - model.center @ matrix, yaw, pitch]


def optimize(observations, seed, prior, yaw_prior, pitch_prior, speed_sigma, anchor=None, edges=(), phase="full", yaw_window=None, model=DEFAULT_TARGET):
    prior_center = None if prior is None else center_of(prior, model)

    def residual(pose):
        matrix = Rot(*pose[3:]).T
        world = model.vertices @ matrix + pose[:3]
        values = []
        for ob in observations:
            uv, depth = ob.view.project(world)
            if ob.clipped:
                uv = np.clip(uv, [0, 0], [ob.view.camera.width - 1, ob.view.camera.height - 1])
            error = ((uv @ DIRECTIONS.T).max(axis=0) - ob.target) / ob.sigma
            values.extend(error * ob.weight)
            values.append(min(depth.min() - 1, 0) * 30)
        for edge in edges:
            values.extend(np.minimum(edge.residual(pose), 22) / 3.6)
        if yaw_prior is not None:
            values.append(angle_diff(pose[3], yaw_prior) / (0.42 if len(observations) == 1 else 0.34))
        if anchor is not None:
            values.append((pose[2] - anchor[0]) / anchor[1])
        values.append((pose[4] - pitch_prior) / (0.075 if len(observations) == 1 else 0.12))
        if prior is not None:
            values.extend((pose[:3] + model.center @ matrix - prior_center) / speed_sigma)
        return np.asarray(values)

    low = np.array([-np.inf] * 4 + [-0.52])
    high = np.array([np.inf] * 4 + [0.52])
    if yaw_window is not None and yaw_prior is not None:
        low[3], high[3] = yaw_prior - yaw_window, yaw_prior + yaw_window
        seed = seed.copy()
        seed[3] = yaw_prior + angle_diff(seed[3], yaw_prior)
    if anchor is not None:
        low[2] = anchor[0] - 1.35 * anchor[1]
        high[2] = anchor[0] + 1.35 * anchor[1]
    res = least_squares(
        residual, np.clip(seed, low, high), loss="soft_l1", f_scale=2.5,
        x_scale=[5, 5, 2, 0.3, 0.1],
        bounds=(low, high),
        max_nfev=25 if phase == "full" else 13, ftol=0.003,
    )
    return res.x, res.cost


def detailed(observation):
    """Angular support large enough to use metric size without a tight road-height prior."""
    camera = observation.view.camera
    return not observation.clipped and min(observation.size / [camera.fx, camera.fy]) > 0.065


def pose_angles(rotation):
    yaw, pitch, roll = Rotation.from_matrix(rotation).as_euler("ZYX")
    return np.array([yaw, -pitch, roll])


def fit_free(observations, prior, gap):
    """Fit a freely rotating rigid body from the current camera observations."""
    model = observations[0].view.model
    offsets = model.vertices - model.center
    main = max(observations, key=lambda ob: min(ob.size) / ob.sigma)
    view = main.view
    origin, ray = view.ray(main.center)
    multi = linear_seed(observations) if len(observations) > 1 else None
    prior_center = None if prior is None else center_of(prior, model)
    prior_rotation = None if prior is None else Rot(*prior[3:])
    scale = max(0.1, model.size)
    weights = [ob.weight for ob in observations]

    def center_seed(rotation):
        if multi is not None:
            return multi
        span = np.ptp(offsets @ rotation.T @ view.matrix[:2].T, axis=0)
        focal = np.array([view.camera.fx, view.camera.fy])
        depth = max(0.08, (span * focal @ main.size) / (main.size @ main.size))
        return origin + ray * (depth / (ray @ view.matrix[2]))

    def pixels(center, rotation):
        world = offsets @ rotation.T + center
        values = []
        for ob, weight in zip(observations, weights):
            uv, depth = ob.view.project(world)
            if ob.clipped:
                uv = np.clip(uv, [0, 0], [ob.view.camera.width - 1, ob.view.camera.height - 1])
            values.extend(weight * ((uv @ DIRECTIONS.T).max(axis=0) - ob.target) / ob.sigma)
            values.append(min(depth.min() - 0.06, 0) * 30)
        return values

    def cost(center, rotation):
        error = np.asarray(pixels(center, rotation))
        return np.sum(np.log1p((error / 2.5)**2))

    seeds = []
    if prior_rotation is not None:
        seeds.extend([(prior_center, prior_rotation), (center_seed(prior_rotation), prior_rotation)])
    if prior is None or gap >= 4:
        rotations = []
        for axes in permutations(range(3)):
            for signs in product((-1, 1), repeat=3):
                rotation = np.eye(3)[:, axes] * signs
                if np.linalg.det(rotation) > 0:
                    rotations.append(rotation)
        coarse = [(center_seed(rotation), rotation) for rotation in rotations]
        seeds.extend(sorted(coarse, key=lambda seed: cost(*seed))[:8])

    def refine(center, rotation):
        def residual(delta):
            result = pixels(center + delta[:3], Rotation.from_rotvec(delta[3:]).as_matrix() @ rotation)
            if prior_center is not None:
                result.extend((center + delta[:3] - prior_center) / (scale * (0.7 + 0.15 * gap)))
                result.extend(delta[3:] / (0.5 + 0.08 * gap))
            return result

        fit = least_squares(residual, np.zeros(6), loss="soft_l1", f_scale=2.5,
                            max_nfev=24, x_scale=[scale] * 3 + [0.5] * 3, ftol=0.003)
        rotation = Rotation.from_rotvec(fit.x[3:]).as_matrix() @ rotation
        result = np.r_[center + fit.x[:3] - rotation @ model.center, pose_angles(rotation)]
        overlaps = np.array([ob.overlap(result) for ob in observations])
        return overlaps.mean() - 0.005 * fit.cost / len(observations), result, overlaps

    _, pose, overlaps = max((refine(*seed) for seed in seeds), key=lambda item: item[0])
    return pose, overlaps


def fit(observations, prior, heading, grade, gap, prior_quality, altitude=None, grays=None, velocity=None):
    model = observations[0].view.model
    if not model.ground_vehicle:
        return fit_free(observations, prior, gap)
    course = None
    if velocity is not None:
        speed = np.linalg.norm(velocity[:2])
        if speed > 0.10:
            course = velocity[:2] / speed

    def direction_score(candidate):
        optical = np.mean([ob.direction(candidate) for ob in observations])
        if course is None:
            return optical
        yaw = candidate[3]
        forward = course @ [np.cos(yaw), np.sin(yaw)]
        return 0.46 * optical + 1.30 * forward
    mono = len(observations) == 1
    if mono and detailed(observations[0]):
        altitude = None
    center = linear_seed(observations) if not mono else None
    if prior is None:
        headings = np.arange(12) * np.pi / 6
        sigma = None
    else:
        headings = [heading]
        if gap >= 5:
            headings += [heading + 0.5, heading - 0.5]
        if mono:
            # A weak depth/road prior stabilizes poorly conditioned single views.
            sigma = np.array([15.0, 15.0, 2.5]) * (1 + min(gap, 25) / 6)
            if prior_quality < 0.5:
                sigma *= 1.6
        else:
            sigma = np.array([3.0, 3.0, 1.25]) * (1 + min(gap, 25) / 7)
    best = None
    yaw_prior = heading if prior is not None else None
    yaw_window = 0.21 + min(gap, 10) * 0.12
    main = max(observations, key=lambda o: o.weight * o.view.instance.score * min(o.size.max(), 130))
    for trial, yaw in enumerate(headings):
        if trial and prior is not None and best[2].min() >= 0.78:
            break
        if center is not None:
            seed = np.r_[center - model.center @ Rot(yaw, grade).T, yaw, grade]
        elif prior is not None:
            seed = mono_seed(main, yaw, grade) if mono and detailed(main) else np.r_[prior[:3], yaw, grade]
            if mono and altitude is not None:
                origin, ray = main.view.ray(main.center)
                center_offset = model.center @ Rot(yaw, grade).T
                if ray[2] < -0.12:
                    depth = (altitude[0] + center_offset[2] - origin[2]) / ray[2]
                    if depth > 8:
                        seed[:3] = origin + ray * depth - center_offset
        else:
            seed = mono_seed(main, yaw, grade)
        pose, cost = optimize(observations, seed, prior, heading if prior is not None else None,
                                   grade, sigma, altitude if mono else None, yaw_window=yaw_window, model=model)
        overlaps = np.array([ob.overlap(pose) for ob in observations])
        directional = direction_score(pose)
        merit = np.mean(overlaps) - 0.005 * cost / len(observations) + 0.11 * directional
        if best is None or merit > best[0]:
            best = merit, pose, overlaps
    merit, pose, overlaps = best
    signs = np.array([ob.direction(pose) for ob in observations])
    strongest = np.argmax(np.abs(signs))
    optical_reverse = signs[strongest] < -0.38 and signs.mean() < -0.15
    course_reverse = course is not None and direction_score(pose) < -0.48
    if optical_reverse or course_reverse:
        flipped = pose.copy()
        flipped[3] += np.pi
        flipped[:3] += (Rot(*pose[3:]) - Rot(*flipped[3:])) @ model.center
        alternative, cost = optimize(
            observations, flipped, flipped, flipped[3], grade, [10, 10, 3],
            altitude if mono else None, model=model,
        )
        alternative_overlaps = np.array([ob.overlap(alternative) for ob in observations])
        signs_alt = np.array([ob.direction(alternative) for ob in observations])
        flip_direction = direction_score(alternative)
        flip_merit = alternative_overlaps.mean() - 0.005 * cost / len(observations) + 0.11 * flip_direction
        if flip_merit > merit - 0.025 and (signs_alt[strongest] > 0.25 or flip_direction > 0.45):
            pose, overlaps = alternative, alternative_overlaps
            yaw_prior = flipped[3]
            prior = flipped
            sigma = np.array([10, 10, 3])
    if model.edge_part is not None and grays is not None and min(overlaps) >= 0.50:
        from online.estimation.edges import BoxEdges

        edges = [BoxEdges(grays[ob.view.camera.index], ob.view, ob.box, pose)
                 for ob in observations if not ob.clipped and min(ob.size) >= 20]
        if edges:
            improved, _ = optimize(
                observations, pose, prior, yaw_prior,
                grade, sigma, altitude if mono else None, edges, phase="refine", yaw_window=yaw_window, model=model,
            )
            new_overlaps = np.array([ob.overlap(improved) for ob in observations])
            if np.mean(new_overlaps) > np.mean(overlaps) - 0.055:
                pose, overlaps = improved, new_overlaps
    return pose, overlaps


def intersects(box, width, height, margin=0):
    return (box is not None and box[2] >= -margin and box[3] >= -margin
            and box[0] < width + margin and box[1] < height + margin)
