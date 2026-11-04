"""Check learned pose corrections against independent visible image boundaries."""
import cv2
import numpy as np

from online.geometry import Observation
from online.perception.rendering import image_crop, render_crop


def boundaries_support(baseline, candidate, images, views, model):
    observations = [Observation(view) for view in views]
    eligible = [ob for ob in observations if not ob.clipped and min(ob.size) >= 36
                and ob.overlap(baseline) >= 0.70]
    if not eligible:
        return False
    observation = max(eligible, key=lambda ob: min(ob.size) / ob.view.camera.fpx)
    before = render_crop(model, baseline, observation.view, observation.box)
    after = render_crop(model, candidate, observation.view, observation.box)
    crop = image_crop(images[observation.view.camera.index], before.transform, before.mask.shape[0])
    edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 40, 100)
    distances = cv2.distanceTransform(1 - (edges > 0).astype(np.uint8), cv2.DIST_L2, 3)
    scores = []
    for rendered in (before, after):
        mask = rendered.mask.astype(np.uint8)
        boundary = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
        samples = distances[boundary > 0]
        if len(samples) < 16:
            return False
        scores.append((np.median(samples), np.mean(np.minimum(samples, 8))))
    return bool(scores[1][0] <= scores[0][0] and scores[1][1] < scores[0][1])
