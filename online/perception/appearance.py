"""Optional current-frame RGB-to-CAD pose refinement using PicoPose."""

from itertools import combinations
from pathlib import Path
import sys

import cv2
import numpy as np

from online.geometry import Observation, Rot, center_of, pose_angles

# Unreal coordinates are left handed; OpenCV's camera coordinates are right handed.
REFLECT_Y = np.diag([1.0, -1.0, 1.0])


def render_template(angles, triangles, colors):
    az, el = angles
    center = (triangles.max((0, 1)) + triangles.min((0, 1))) / 2
    extent = np.ptp(triangles, axis=(0, 1))
    out = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    origin = center + np.linalg.norm(extent) * 1.45 * out
    forward = -out
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    translation = -rotation @ origin
    points = triangles @ rotation.T + translation
    size = 384
    focal = 450.0
    mid = size / 2
    K = np.array([[focal, 0, mid], [0, focal, mid], [0, 0, 1.0]])
    uv = points[..., :2] * focal / points[..., 2, None] + mid
    color = np.zeros((size, size, 3), np.uint8)
    zmap = np.zeros((size, size), np.float32) + np.inf
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    shades = 0.55 + 0.45 * np.abs(
        normals @ np.array([-0.22, -0.35, -0.91])
    ) / np.maximum(norms, 1e-9)
    for pts, xy, rgb, shade in zip(points, uv, colors, shades):
        lo = np.maximum(np.floor(xy.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(xy.max(axis=0)).astype(int) + 1, size)
        if np.any(hi <= lo):
            continue
        x0, y0 = lo
        x1, y1 = hi
        yy, xx = np.ogrid[y0:y1, x0:x1]
        a, b = xy[1:] - xy[:1]
        det = a[0] * b[1] - a[1] * b[0]
        if abs(det) < 1e-6:
            continue
        x = xx - xy[0, 0]
        y = yy - xy[0, 1]
        u = (x * b[1] - y * b[0]) / det
        v = (a[0] * y - a[1] * x) / det
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        z = 1 / np.maximum(
            (1 - u - v) / pts[0, 2] + u / pts[1, 2] + v / pts[2, 2], 1e-6
        )
        keep = inside & (z < zmap[y0:y1, x0:x1])
        zmap[y0:y1, x0:x1][keep] = z[keep]
        color[y0:y1, x0:x1][keep] = np.clip(rgb * shade, 0, 255).astype(np.uint8)
    mask = np.isfinite(zmap)
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    xyz = np.zeros((size, size, 3), np.float32)
    d = zmap[y, x]
    xyz[y, x] = np.c_[(x - mid) * d / focal, (y - mid) * d / focal, d]
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    lo = [x.min(), y.min()]
    hi = [x.max() + 1, y.max() + 1]
    return cropped(color, mask, lo, hi, K, pose, xyz)


def cropped(image, mask, lo, hi, K, pose=None, xyz=None):
    x, y = lo
    u, v = hi
    side = max(u - x, v - y)
    if side < 8:
        return None
    x = (x + u - side) // 2
    y = (y + v - side) // 2
    resize = 224 / side
    M = np.array([[resize, 0, -x * resize], [0, resize, -y * resize], [0, 0, 1.0]])
    im = cv2.warpAffine(image, M[:2], (224, 224)).astype(np.float32) / 255
    m = cv2.warpAffine(
        mask.astype("uint8"), M[:2], (224, 224), flags=cv2.INTER_NEAREST
    ).astype(np.float32)
    im = (im - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array(
        [0.26862954, 0.26130258, 0.27577711]
    )
    data = {"rgb": im.transpose(2, 0, 1), "mask": m, "K": K, "M": M}
    xx, yy = np.meshgrid(
        (np.arange(64) + 0.5) * side / 64 + x, (np.arange(64) + 0.5) * side / 64 + y
    )
    if xyz is not None:
        data["pts3d"] = cv2.remap(
            xyz, xx.astype(np.float32), yy.astype(np.float32), cv2.INTER_NEAREST
        )
        data["pose"] = pose
    else:
        data["pts2d"] = np.stack([xx, yy], axis=-1)
        data["pose"] = np.eye(4)
    return data


def model_points(output, continuous=None):
    # The published inference outputs indices into square template/query grids.
    if continuous is not None:
        flow, certainty = continuous
        flow = flow[0].permute(1, 2, 0).float().cpu().numpy()
        certainty = certainty[0, 0].sigmoid().float().cpu().numpy()
        height, width = flow.shape[:2]
        yy, xx = np.indices((height, width))
        pixels = flow + np.stack([xx, yy], axis=-1)
        valid = (certainty > 0.5) & np.isfinite(pixels).all(axis=-1)
        valid &= (pixels > 0).all(axis=-1) & (pixels < [width - 1, height - 1]).all(axis=-1)
        source = output["src_pts_3d"][0].permute(1, 2, 0).float().cpu().numpy()
        valid &= source[..., 2] > 0
        query_grid = output["tar_pts_2d"][0].float().cpu().numpy()
        origin = query_grid[:, 0, 0]
        step_x = query_grid[:, 1, 0] - origin
        step_y = query_grid[:, 0, 1] - origin
        # Training targets are crop coordinates divided by patch size.
        # The lookup grid is centered at half a cell.
        uv = origin + (pixels[valid, 0, None] - 0.5) * step_x + (pixels[valid, 1, None] - 0.5) * step_y
        pose = output["tem_pose"][0].float().cpu().numpy()
        return (source[valid] - pose[:3, 3]) @ pose[:3, :3], uv
    u = output["pred_tar_pts"][0].cpu().numpy()
    q = output["pred_src_pts"][0].cpu().numpy()
    take = (u >= 0).all(1) & (q >= 0).all(1) & (u < 64).all(1) & (q < 64).all(1)
    u, q = u[take], q[take]
    tar = output["tar_pts_2d"][0].cpu().numpy()
    src = output["src_pts_3d"][0].cpu().numpy()
    P = src[:, q[:, 1], q[:, 0]].T
    uv = tar[:, u[:, 0], u[:, 1]].T
    valid = P[:, 2] > 0
    pose = output["tem_pose"][0].cpu().numpy()
    return (P[valid] - pose[:3, 3]) @ pose[:3, :3], uv[valid]


def fit_correspondences(P, uv, K, view):
    if len(P) < 16 or not np.isfinite(P).all() or not np.isfinite(uv).all():
        return None
    index = np.arange(len(P))
    val = index % 4 == 0
    train = ~val
    P = np.ascontiguousarray(P, dtype=np.float64)
    uv = np.ascontiguousarray(uv, dtype=np.float64)
    try:
        ok, r, t, ii = cv2.solvePnPRansac(
            P[train],
            uv[train],
            K,
            None,
            iterationsCount=180,
            reprojectionError=4.0,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or ii is None or len(ii) < 8:
            return None
        idx = np.flatnonzero(train)[ii[:, 0]]
        r, t = cv2.solvePnPRefineLM(P[idx], uv[idx], K, None, r, t)
    except cv2.error:
        return None
    if not np.isfinite(r).all() or not np.isfinite(t).all():
        return None
    R = cv2.Rodrigues(r)[0]
    residual = np.linalg.norm(
        cv2.projectPoints(P[val], r, t, K, None)[0][:, 0] - uv[val], axis=1
    )
    world = view.matrix.T @ t[:, 0] + view.camera.loc / 100
    world_r = view.matrix.T @ R @ REFLECT_Y
    return dict(
        pose=np.r_[world, pose_angles(world_r)],
        heldout_median=float(np.median(residual)),
        heldout_fraction=float(np.mean(residual < 5)),
    )


def query_for(image, contour, view):
    contour = np.asarray(contour)
    lo = np.maximum(np.floor(contour.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(contour.max(0)).astype(int) + 1, image.shape[:2][::-1])
    m = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(contour).astype(np.int32)], 1)
    c = view.camera
    K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1.0]])
    return cropped(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), m, lo, hi, K), K


def load_network(source, checkpoint):
    import collections
    import typing

    import torch
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    root = Path(source).resolve()
    sys.path.insert(0, str(root))
    try:
        from model.picopose import Net

        config = OmegaConf.load(root / "config/base.yaml").model
        config.stage1.pretrained = False
        safe_types = [
            DictConfig,
            ListConfig,
            Metadata,
            ContainerMetadata,
            AnyNode,
            collections.defaultdict,
            typing.Any,
            list,
            dict,
            int,
        ]
        with torch.serialization.safe_globals(safe_types):
            state = torch.load(
                checkpoint, weights_only=True, map_location="cpu", mmap=True
            )["state_dict"]
        weights = {
            key.removeprefix("network."): value
            for key, value in state.items()
            if key.startswith("network.")
        }
        model = Net(config)
        model.load_state_dict(weights, strict=True, assign=True)
        return model.eval().to("cuda")
    finally:
        sys.path.pop(0)


class PicoPose:
    def __init__(self, source, checkpoint, appearance, subpixel=False):
        import torch

        if appearance.triangles is None:
            raise ValueError("PicoPose requires a triangular CAD mesh")
        if not torch.cuda.is_available():
            raise ValueError("PicoPose requires CUDA in this configuration")
        self.torch = torch
        self.network = load_network(source, checkpoint)
        self.subpixel = subpixel
        triangles = appearance.triangles @ REFLECT_Y
        with torch.inference_mode():
            bank = [
                render_template((azimuth, elevation), triangles, appearance.face_colors)
                for elevation in np.deg2rad([-37, 0, 32, 61])
                for azimuth in np.linspace(-np.pi, np.pi, 12, endpoint=False)
            ]
            bank = [item for item in bank if item is not None]
            if len(bank) < 3:
                raise ValueError("CAD mesh could not produce three usable PicoPose templates")
            self.bank = {
                "tem_" + key: self.tensor(np.stack([item[key] for item in bank]))[None]
                for key in bank[0]
            }
            features = [
                self.network.feature_extractor(image)[-1]
                for image in self.bank["tem_rgb"][0].split(8)
            ]
            self.bank["template_feature"] = torch.cat(features)[None]

    def tensor(self, value):
        return self.torch.as_tensor(value, dtype=self.torch.float32, device="cuda")

    def hypotheses(self, image, view, retain_points=False):
        query, intrinsic = query_for(image, view.instance.contour, view)
        if query is None:
            return []
        inputs = {
            **self.bank,
            **{"real_" + key: self.tensor(value)[None] for key, value in query.items()},
        }
        continuous = []
        hook = None
        if self.subpixel:
            hook = self.network.offset_regressor.register_forward_hook(
                lambda module, args, result: continuous.append((result[0][-1], result[1][-1])))
        try:
            with self.torch.inference_mode():
                output = self.network(inputs, hyp=3)
        finally:
            if hook is not None:
                hook.remove()
        if self.subpixel and len(continuous) != len(output):
            raise RuntimeError("PicoPose flow outputs do not match the selected hypotheses")
        matches = []
        for index, item in enumerate(output):
            points, pixels = model_points(item, continuous[index] if self.subpixel else None)
            match = fit_correspondences(points, pixels, intrinsic, view)
            if match is not None and retain_points:
                match["model_points"] = points @ REFLECT_Y
                match["image_points"] = pixels
            matches.append(match)
        return matches

    def refine(self, pose, images, views, accepted, joint=False, spatial_check=False):
        self.joint_applied = False
        self.single_applied = False
        self.single_pose = pose
        if pose is None or not accepted:
            return pose
        cache = {}

        def hypotheses(view):
            key = view.camera.index
            if key not in cache:
                cache[key] = self.hypotheses(images[key], view, retain_points=joint)
            return cache[key]

        single = self.refine_single(pose, views, accepted, hypotheses)
        self.single_applied = single is not pose
        self.single_pose = single
        pose = single
        if joint:
            from online.estimation.joint_pose import refine_joint

            observations = [Observation(view) for view in views]
            eligible = [obs for obs in observations if not obs.clipped and min(obs.size) >= 36
                        and obs.overlap(pose) >= 0.35]
            if len(eligible) >= 2:
                candidate = refine_joint(pose, [(obs, hypotheses(obs.view)) for obs in eligible], spatial_check)
                self.joint_applied = candidate is not pose
                pose = candidate
        return pose

    def recover_second_view(self, frame, previous, pose, images, views, primary_id, memories):
        """Add a second camera only if both reprojection and motion support it."""
        from online.estimation.joint_pose import motion_supports, refine_joint

        if previous is None or len(views) < 2:
            return pose, None
        observations = [Observation(view) for view in views]
        eligible = {obs.view.camera.index: obs for obs in observations
                    if not obs.clipped and min(obs.size) >= 36 and obs.overlap(pose) >= 0.35}
        if primary_id not in eligible or len(eligible) < 2:
            return pose, None
        primary = eligible.pop(primary_id)
        _, primary_ray = primary.view.ray(primary.center)
        primary_matches = self.hypotheses(images[primary_id], primary.view, retain_points=True)
        recovered = []
        for key, secondary in eligible.items():
            _, ray = secondary.view.ray(secondary.center)
            parallax = np.linalg.norm(np.cross(primary_ray, ray))
            if parallax < 0.15:
                continue
            secondary_matches = self.hypotheses(images[key], secondary.view, retain_points=True)
            candidate = refine_joint(pose, [(primary, primary_matches), (secondary, secondary_matches)],
                                     spatial_check=True)
            if candidate is not pose and motion_supports(
                    frame, previous, pose, candidate, (primary.view, secondary.view), memories):
                recovered.append((parallax, candidate, secondary.view))
        if not recovered:
            return pose, None
        _, candidate, secondary = max(recovered, key=lambda result: result[0])
        center = center_of(candidate, secondary.model)
        if any(np.linalg.norm(center - center_of(other, secondary.model)) > 0.05 * secondary.model.size
               for _, other, _ in recovered):
            return pose, None
        return candidate, secondary

    def refine_single(self, pose, views, accepted, hypotheses):
        options = []
        for view in views:
            observation = Observation(view)
            pixels = min(observation.size)
            angular = pixels / view.camera.fpx
            if observation.clipped or angular < 0.12:
                continue
            if any(
                other.camera.index in accepted
                and other.camera.index != view.camera.index
                and min(other.instance.box[2:] - other.instance.box[:2])
                / other.camera.fpx
                > angular / 3
                for other in views
            ):
                continue

            old_iou = observation.overlap(pose)
            size = view.model.size
            candidates = []
            for match in hypotheses(view):
                if (
                    match is None
                    or match["heldout_fraction"] < 0.5
                    or match["heldout_median"] > 0.04 * pixels
                ):
                    continue
                candidate = match["pose"]
                delta = np.linalg.norm(candidate[:3] - pose[:3])
                if not 0.15 * size < delta < 0.60 * size:
                    continue
                iou = observation.overlap(candidate)
                if iou >= max(0.70, old_iou - 0.02):
                    candidates.append((iou, match["heldout_fraction"], candidate))
            for a, b in combinations(candidates, 2):
                if np.linalg.norm(a[2][:3] - b[2][:3]) > 0.12 * size:
                    continue
                relative = Rot(*a[2][3:]).T @ Rot(*b[2][3:])
                if np.trace(relative) < 1 + 2 * np.cos(np.deg2rad(20)):
                    continue
                choice = max((a, b), key=lambda entry: entry[:2])
                if choice[0] >= old_iou + 0.015:
                    options.append(choice)
        return max(options, key=lambda entry: entry[:2])[2] if options else pose
