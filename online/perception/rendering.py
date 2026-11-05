"""Render visible CAD samples in a calibrated crop."""
from dataclasses import dataclass

import cv2
import numpy as np

from online.geometry import Rot


@dataclass
class RenderedCrop:
    rgb: np.ndarray
    points: np.ndarray
    mask: np.ndarray
    transform: np.ndarray

    def original_pixels(self, pixels):
        inverse = np.linalg.inv(self.transform)
        return pixels @ inverse[:2, :2].T + inverse[:2, 2]


def crop_transform(box, size, padding=1.2):
    center = (box[:2] + box[2:]) / 2
    side = max(8., float(np.max(box[2:] - box[:2])) * padding)
    scale = size / side
    return np.array([[scale, 0, size / 2 - center[0] * scale],
                     [0, scale, size / 2 - center[1] * scale], [0, 0, 1.]])


def render_crop(model, pose, view, box, size=280):
    """Perspective-correct z-buffer with object-space points and face colors."""
    if model.triangles is None:
        raise ValueError("Rendering requires a triangular CAD mesh")
    transform = crop_transform(box, size)
    world = model.triangles @ Rot(*pose[3:]).T + pose[:3]
    uv, depths = view.project(world.reshape(-1, 3))
    uv = (uv @ transform[:2, :2].T + transform[:2, 2]).reshape(-1, 3, 2)
    depths = depths.reshape(-1, 3)
    rgb = np.zeros((size, size, 3), np.uint8)
    points = np.zeros((size, size, 3), np.float32)
    zbuffer = np.full((size, size), np.inf)
    for triangle, pixel, depth, color in zip(model.triangles, uv, depths, model.face_colors):
        if depth.min() <= 0.05:
            continue
        low = np.maximum(np.floor(pixel.min(0)).astype(int), 0)
        high = np.minimum(np.ceil(pixel.max(0)).astype(int) + 1, size)
        if np.any(high <= low):
            continue
        a, b = pixel[1:] - pixel[0]
        det = a[0] * b[1] - a[1] * b[0]
        if abs(det) < 1e-8:
            continue
        x0, y0 = low
        x1, y1 = high
        yy, xx = np.mgrid[y0:y1, x0:x1]
        delta_x, delta_y = xx - pixel[0, 0], yy - pixel[0, 1]
        u = (delta_x * b[1] - delta_y * b[0]) / det
        v = (a[0] * delta_y - a[1] * delta_x) / det
        bary = np.stack([1 - u - v, u, v], axis=-1) / depth
        reciprocal = bary.sum(-1)
        z = 1 / np.maximum(reciprocal, 1e-12)
        keep = (u >= 0) & (v >= 0) & (u + v <= 1) & (z < zbuffer[y0:y1, x0:x1])
        points[y0:y1, x0:x1][keep] = (bary[keep] @ triangle) * z[keep, None]
        rgb[y0:y1, x0:x1][keep] = color
        zbuffer[y0:y1, x0:x1][keep] = z[keep]
    return RenderedCrop(rgb, points, np.isfinite(zbuffer), transform)


def image_crop(image, transform, size):
    return cv2.warpAffine(image, transform[:2], (size, size))
