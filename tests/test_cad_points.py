from itertools import product
import unittest

import cv2
import numpy as np

from core.camera import Camera
from online.estimation.cad_points import CadPointRecovery, fit_points
from online.estimation.surface_pose import Tracks
from online.geometry import Observation, Rot, View
from online.perception.vision import Instance
from online.target import Target
from online.tracker import Estimate


class CadPointTests(unittest.TestCase):
    def test_metric_pose_requires_agreement_on_held_out_points(self):
        points = np.array(list(product((-2., -0.4, 1.6), (-1., 0.6), (-0.8, 0.9))))
        model = Target("probe", (points,))
        camera = Camera(7, np.array([0., 0., 0.]), np.zeros(3), 40., (36., 27.), 960, 720)
        truth = np.array([12., 0.4, -0.3, 0.14, -0.12, 0.08])
        pixels, _ = View.of(camera, model=model).project(points @ Rot(*truth[3:]).T + truth[:3])
        hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
        observation = Observation(View.of(camera, Instance(hull, .9, 0), model))
        holdout = np.arange(len(points)) % 2 == 0
        prior = truth + [.4, -.15, .08, .02, -.01, .03]
        pose, overlap, precision = fit_points(prior, observation, Tracks(points, pixels, holdout), model)
        np.testing.assert_allclose(pose, truth, atol=.01)
        self.assertGreater(overlap, .95)
        self.assertLess(precision, .2)
        wrong_pixels = pixels.copy()
        wrong_pixels[holdout] += [12, -8]
        self.assertIsNone(fit_points(prior, observation, Tracks(points, wrong_pixels, holdout), model))

    def test_render_match_must_be_confirmed_before_emitting(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        model = Target("probe", (points,), triangles=points[None])
        solver = CadPointRecovery(model, 16, matcher=object())
        camera = Camera(3, np.zeros(3), np.zeros(3), 40., (36., 27.), 640, 480)
        shape = Instance(np.array([[20., 20.], [80., 20.], [80., 80.]]), .9, 0)
        pose = np.array([10., 0., 0., 0., 0., 0.])
        solver.match_points = lambda *args: (.2, 3, pose, .8)
        cameras = {3: camera}
        images = {3: np.zeros((480, 640, 3), dtype=np.uint8)}
        primary = Estimate(pose, "mono", {3: shape}, .9)
        self.assertIs(solver.step(0, primary, cameras, images), primary)
        missing = Estimate(None, "lost", {}, 0., {3: shape})
        self.assertIs(solver.step(1, missing, cameras, images), missing)
        self.assertIs(solver.step(2, missing, cameras, images), missing)
        self.assertEqual(solver.step(3, missing, cameras, images).status, "cad_points")
        self.assertIsNone(missing.pose)
