"""Causal checks for camera images and suspect object observations."""
from dataclasses import replace

import cv2
import numpy as np

from online.motion import camera_warp


def prepare_image(image, reference=None):
    """Normalize exposure jumps against recent good luminance, without moving pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sampled = gray[::8, ::8]
    levels = np.percentile(sampled, [5, 50, 95])
    low, median, high = levels
    spread = high - low
    if spread < 7 and (median < 24 or median > 232 or np.ptp(sampled) < 12):
        return None, gray, reference, False
    jump = reference is not None and abs(median - reference[1]) > max(18, 0.22 * reference[1])
    extreme = median < 35 or median > 220
    if jump or extreme:
        nominal = reference[1] if reference is not None else np.clip(median, 90, 165)
        nominal_spread = reference[2] - reference[0] if reference is not None else 90
        gain = np.clip(nominal_spread / max(spread, 1), 0.42, 3.0)
        lut = np.clip((np.arange(256) - median) * gain + nominal, 0, 255).astype(np.uint8)
        image = cv2.LUT(image, lut)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if reference is None:
            reference = np.percentile(gray[::8, ::8], [5, 50, 95])
        return image, gray, reference, True
    reference = levels if reference is None else 0.95 * reference + 0.05 * levels
    return image, gray, reference, False


def stabilize(view, old_view, current, past, target_box=None):
    """Estimate small unreported rotations from distributed static-background matches."""
    if current.shape != past.shape or np.linalg.norm(view.camera.loc - old_view.camera.loc) > 5:
        return view
    ratio = min(0.5, 640 / current.shape[1])
    a, b = (cv2.equalizeHist(cv2.resize(g, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA))
            for g in (past, current))
    height, width = a.shape
    if min(height, width) < 40:
        return view
    mask = np.zeros_like(a)
    mask[8:-8, 8:-8] = 255
    if target_box is not None:
        lo = np.maximum(np.floor(target_box[:2] * ratio).astype(int) - 12, 0)
        hi = np.minimum(np.ceil(target_box[2:] * ratio).astype(int) + 12, [width, height])
        mask[lo[1]:hi[1], lo[0]:hi[0]] = 0
    before = cv2.goodFeaturesToTrack(a, 160, 0.02, 9, mask=mask)
    if before is None or len(before) < 24:
        return view
    predicted = cv2.perspectiveTransform(before, camera_warp(view, old_view, ratio))
    projected = predicted[:, 0]
    inside = np.isfinite(projected).all(axis=1) & (projected >= 4).all(axis=1)
    inside &= (projected <= [width - 5, height - 5]).all(axis=1)
    before, predicted = before[inside], predicted[inside]
    if len(before) < 24:
        return view
    opts = dict(winSize=(21, 21), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    found, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, before, predicted.copy(), **opts)
    if found is None:
        return view
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, found, before.copy(), **opts)
    if back is None:
        return view
    predicted, found = predicted[:, 0], found[:, 0]
    valid = (ok[:, 0] == 1) & (back_ok[:, 0] == 1)
    valid &= np.linalg.norm(back[:, 0] - before[:, 0], axis=1) < 0.8
    valid &= (found >= 4).all(axis=1) & (found <= [width - 5, height - 5]).all(axis=1)
    if valid.sum() < 24:
        return view
    a, b = predicted[valid], found[valid]
    if np.median(np.linalg.norm(a - b, axis=1)) < 1.0:
        return view
    _, inliers = cv2.findHomography(a, b, cv2.RANSAC, 1.2)
    if inliers is None:
        return view
    good = inliers[:, 0] == 1
    if good.sum() < 20 or good.mean() < 0.65:
        return view
    a, b = a[good], b[good]
    occupied = set(zip((b[:, 0] > width / 2), (b[:, 1] > height / 2)))
    if len(occupied) < 3 or np.any(np.ptp(b, axis=0) < [0.30 * width, 0.30 * height]):
        return view

    cam = view.camera
    focal = ratio * np.array([cam.fx, cam.fy])
    principal = ratio * np.array([cam.cx, cam.cy])
    rays_a = np.c_[(a - principal) / focal, np.ones(len(a))]
    rays_b = np.c_[(b - principal) / focal, np.ones(len(b))]
    rays_a /= np.linalg.norm(rays_a, axis=1)[:, None]
    rays_b /= np.linalg.norm(rays_b, axis=1)[:, None]
    u, _, vt = np.linalg.svd(rays_b.T @ rays_a)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    if not np.deg2rad(0.1) <= angle <= np.deg2rad(4):
        return view
    adjusted = rays_a @ rotation.T
    pixels = adjusted[:, :2] / adjusted[:, 2, None] * focal + principal
    before_error = np.median(np.linalg.norm(a - b, axis=1))
    after_error = np.median(np.linalg.norm(pixels - b, axis=1))
    if after_error > min(0.85, 0.40 * before_error):
        return view
    return replace(view, matrix=rotation @ view.matrix, calibration_adjusted=True)


def plausible_box(box, expected, gap):
    """Reject distant lookalikes while allowing partial silhouettes."""
    size = np.maximum(expected[2:] - expected[:2], 12)
    delta = ((box[:2] + box[2:]) - (expected[:2] + expected[2:])) / (2 * size)
    relative_size = (box[2:] - box[:2]) / size
    allowance = 0.90 + 0.10 * min(gap, 10)
    return np.linalg.norm(delta) <= allowance and max(relative_size) <= 3.0 + 0.15 * min(gap, 10)
