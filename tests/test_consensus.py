"""Guard against promoting an isolated detection into the multi-camera prior."""
import unittest
from itertools import product

import cv2
import numpy as np

from core.camera import Camera
from online.estimation.consensus import ConsensusTracker
from online.geometry import DEFAULT_TARGET, Rot, View, center_of
from online.perception.vision import Instance
from online.target import Target


class Detector:
    def __init__(self):
        self.detections = {}

    def __call__(self, images):
        return {key: self.detections[key] for key in images if key in self.detections}

    def overhead(self, images):
        return {}

    def tiles(self, images):
        return {}

    def regions(self, images, boxes):
        return {}


class ConsensusTests(unittest.TestCase):
    def test_unverified_pose_is_not_carried_into_next_frame(self):
        model = DEFAULT_TARGET
        cameras = {
            1: Camera(1, np.array([0., -1200., 300.]), np.array([0., -4., 14.]), 35., (36., 27.), 640, 480),
            2: Camera(2, np.array([500., 1600., 600.]), np.array([0., -7., -20.]), 35., (36., 27.), 640, 480),
        }
        truth = np.array([55., 0., 0., 0., 0.])
        masks = {}
        for key, camera in cameras.items():
            pixels, _ = View.of(camera, model=model).project(model.vertices + truth[:3])
            hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
            masks[key] = [Instance(hull, 0.95, 7)]

        detector = Detector()
        images = {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in cameras}
        tracker = ConsensusTracker(detector, model, 30, len(cameras))
        detector.detections = {1: masks[1]}
        self.assertIsNone(tracker.step(0, cameras, images).pose)
        detector.detections = {}
        self.assertIsNone(tracker.step(1, cameras, images).pose)
        detector.detections = masks
        estimate = tracker.step(2, cameras, images)
        self.assertEqual(estimate.status, "multi")
        self.assertEqual(set(estimate.views), {1, 2})
        np.testing.assert_allclose(center_of(estimate.pose, model), center_of(truth, model), atol=1.0)
        detector.detections = {}
        self.assertEqual(tracker.step(3, cameras, images).status, "forecast")
        for frame in range(4, 34):
            tracker.step(frame, cameras, images)
        detector.detections = {1: masks[1]}
        self.assertIsNone(tracker.step(34, cameras, images).pose)

    def test_acquires_a_free_rigid_object_from_two_calibrated_views(self):
        body = np.array(list(product((-2.0, 2.0), (-0.8, 0.8), (-0.6, 0.6))))
        appendage = np.array(list(product((0.2, 1.4), (0.4, 1.2), (0.5, 1.0))))
        model = Target("inspection probe", (body, appendage), speed=8)
        truth = np.array([15., 0., 1., 0.26, 0.13, -0.12])
        cameras = {
            2: Camera(2, np.array([0., -450., 300.]), np.array([0., -5., 17.]), 35., (36., 27.), 640, 480),
            8: Camera(8, np.array([0., 650., 400.]), np.array([0., -10., -23.]), 35., (36., 27.), 640, 480),
        }
        detector = Detector()
        for key, camera in cameras.items():
            pixels, _ = View.of(camera, model=model).project(model.vertices @ Rot(*truth[3:]).T + truth[:3])
            hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
            detector.detections[key] = [Instance(hull, 0.95, 7)]
        images = {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in cameras}
        estimate = ConsensusTracker(detector, model, 30, len(cameras)).step(0, cameras, images)
        self.assertEqual(estimate.status, "multi")
        self.assertEqual(len(estimate.pose), 6)
        np.testing.assert_allclose(center_of(estimate.pose, model), center_of(truth, model), atol=0.3)


if __name__ == "__main__":
    unittest.main()
