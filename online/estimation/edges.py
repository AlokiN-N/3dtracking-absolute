"""Continuous reprojection residual for the visible edges of the CAD cargo box."""
from functools import lru_cache
from itertools import product

import cv2
import numpy as np

from online.geometry import Rot


@lru_cache(maxsize=8)
def box_geometry(low, high):
    corners = np.array(list(product(*zip(low, high))))
    edges = []
    for a, i in enumerate(corners):
        for b, j in enumerate(corners[a+1:], a+1):
            fixed = np.flatnonzero(i == j)
            if len(fixed) == 2:
                edges.append((a, b, fixed, i[fixed] == np.asarray(high)[fixed]))
    return corners, edges


BINS = 8
ANGLES = np.arange(BINS) * np.pi / BINS
STEPS = np.linspace(0.11, 0.89, 7)


class BoxEdges:
    def __init__(self, gray, view, box, seed):
        self.view = view
        low, high = view.model.edge_part.min(axis=0), view.model.edge_part.max(axis=0)
        self.corners, model_edges = box_geometry(tuple(low), tuple(high))
        extent = box[2:] - box[:2]
        self.origin = np.maximum(np.floor(box[:2] - extent * .28).astype(int), 0)
        end = np.minimum(np.ceil(box[2:] + extent * .28).astype(int), gray.shape[::-1])
        x, y = self.origin
        x2, y2 = end
        crop = gray[y:y2, x:x2]
        blurred = cv2.GaussianBlur(crop, (3, 3), 0.65)
        edges = cv2.Canny(blurred, 18, 46, L2gradient=True) > 0
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
        direction = np.arctan2(gy, gx) % np.pi
        self.distance = []
        for angle in ANGLES:
            diff = np.abs((direction - angle + np.pi / 2) % np.pi - np.pi / 2)
            directional = (edges & (diff < np.pi / 7)).astype(np.uint8)
            self.distance.append(cv2.distanceTransform(1 - directional, cv2.DIST_L2, 3))
        self.distance = np.array(self.distance)
        self.limits = np.array(crop.shape[::-1]) - 2
        camera = (view.camera.loc / 100 - seed[:3]) @ Rot(*seed[3:])
        faces_hi = camera > high
        faces_lo = camera < low
        projected, _ = view.project(self.corners @ Rot(*seed[3:]).T + seed[:3])
        self.a, self.b, self.bins = [], [], []
        for a, b, axes, sides in model_edges:
            visible = np.where(sides, faces_hi[axes], faces_lo[axes])
            delta = projected[b] - projected[a]
            if visible.any() and np.linalg.norm(delta) >= 8:
                angle = (np.arctan2(delta[1], delta[0]) + np.pi / 2) % np.pi
                self.a.append(a)
                self.b.append(b)
                self.bins.append(round(angle * BINS / np.pi) % BINS)
        self.bins = np.asarray(self.bins, dtype=int)[:, None]

    def residual(self, pose):
        if not self.a:
            return np.empty(0)
        uv, _ = self.view.project(self.corners @ Rot(*pose[3:]).T + pose[:3])
        p = uv[self.a, None] * (1 - STEPS[None, :, None]) + uv[self.b, None] * STEPS[None, :, None]
        p -= self.origin
        clipped = np.clip(p, 0, self.limits)
        base = np.floor(clipped).astype(int)
        x, y = base[..., 0], base[..., 1]
        a, b = np.moveaxis(clipped - base, -1, 0)
        image = self.distance
        value = ((1-a) * (1-b) * image[self.bins, y, x]
                 + a * (1-b) * image[self.bins, y, x+1]
                 + (1-a) * b * image[self.bins, y+1, x]
                 + a * b * image[self.bins, y+1, x+1])
        return (value + np.linalg.norm(p-clipped, axis=-1)).ravel()
