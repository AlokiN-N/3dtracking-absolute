"""Calibrated image-pair matching with an optional RoMa v2 backend."""
import sys
from pathlib import Path

import cv2
import numpy as np

from online.estimation.measurements import Correspondences
from online.estimation.surface_pose import Surface
from online.perception.rendering import crop_transform, image_crop


class RoMaMatcher:
    def __init__(self, source, model, setting="fast"):
        import torch

        sys.path.insert(0, str(Path(source).resolve() / "src"))
        from romav2 import RoMaV2

        self.torch = torch
        self.network = RoMaV2(RoMaV2.Cfg(setting=setting))
        self.surface = Surface(model)

    def correspondences(self, pose, images, views):
        eligible = [v for v in views if min(v.instance.box[2:] - v.instance.box[:2]) >= 32]
        if len(eligible) < 2:
            return []
        primary = max(eligible, key=lambda v: min(v.instance.box[2:] - v.instance.box[:2]) / v.camera.fpx)
        transform = crop_transform(primary.instance.box, 512)
        first = image_crop(images[primary.camera.index], transform, 512)
        first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        from PIL import Image

        result = []
        for secondary in eligible:
            if secondary is primary:
                continue
            other_transform = crop_transform(secondary.instance.box, 512)
            second = image_crop(images[secondary.camera.index], other_transform, 512)
            second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
            with self.torch.inference_mode():
                predictions = self.network.match(Image.fromarray(first), Image.fromarray(second))
                with self.torch.random.fork_rng(devices=[self.torch.cuda.current_device()]):
                    self.torch.manual_seed(0)
                    matches, confidence, _, _ = self.network.sample(predictions, 512)
                a, b = self.network.to_pixel_coordinates(matches, 512, 512, 512, 512)
            a, b = a.cpu().numpy(), b.cpu().numpy()
            weight = confidence.cpu().numpy().reshape(-1)
            inverse_a, inverse_b = np.linalg.inv(transform), np.linalg.inv(other_transform)
            a = a @ inverse_a[:2, :2].T + inverse_a[:2, 2]
            b = b @ inverse_b[:2, :2].T + inverse_b[:2, 2]
            points, visible = self.surface.intersect(primary, pose, a)
            a, b, weight = a[visible], b[visible], weight[visible]
            if not len(points):
                continue
            # Known camera poses provide the epipolar constraint at this timestamp.
            origins_a, rays_a = zip(*(primary.ray(pixel) for pixel in a))
            origins_b, rays_b = zip(*(secondary.ray(pixel) for pixel in b))
            baseline = np.asarray(origins_b) - np.asarray(origins_a)
            normal = np.cross(baseline, rays_a)
            angular = np.abs(np.sum(normal * rays_b, axis=1)) / np.maximum(np.linalg.norm(normal, axis=1), 1e-9)
            good = (angular * secondary.camera.fpx < 2.0) & (weight >= 0.5)
            result.extend([Correspondences(primary, points[good], a[good], weight[good]),
                           Correspondences(secondary, points[good], b[good], weight[good])])
        return result
