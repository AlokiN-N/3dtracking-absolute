from itertools import product
import unittest

import cv2
import numpy as np

from core.camera import Camera
from online.estimation.region_flow import fit_translation
from online.estimation.region_pose import Region, clipped_silhouette, fit_region, support
from online.estimation.surface_pose import Tracks
from online.geometry import Observation, Rot, View
from online.perception.vision import Instance
from online.target import Target


class RegionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array(list(product((-1.2, 0., 1.2), (-1.1, 1.1), (-.8, .8))))
        self.model = Target("inspection box", (self.points,))
        camera = Camera(7, np.zeros(3), np.zeros(3), 40., (36., 27.), 640, 480)
        self.view = View.of(camera, model=self.model)

    def observation(self, contour):
        return Observation(View.of(self.view.camera, Instance(contour, .9, 0), self.model))

    def test_recovers_from_visible_contour_at_image_border(self):
        anchor_pose = np.array([10., -3.6, 0., .10, -.08, 0.])
        truth = anchor_pose + [.2, -.1, .08, 0., 0., 0.]
        visible, = clipped_silhouette(self.view, truth)
        self.assertEqual(visible[:, 0].min(), 0)
        self.assertTrue(self.observation(visible).clipped)
        anchor = Region(0, anchor_pose, np.zeros_like(support([visible])))
        recovered, overlap, _, _, _ = fit_region(anchor, self.observation(visible), anchor_pose, 1., 0.)
        np.testing.assert_allclose(recovered, truth, atol=.03)
        self.assertGreater(overlap, .97)

    def test_dense_translation_checks_points_outside_fit(self):
        previous = np.array([10., 0., 0., .10, -.08, 0.])
        truth = previous + [.2, -.1, .08, 0., 0., 0.]
        pixels, _ = self.view.project(self.points @ Rot(*truth[3:]).T + truth[:3])
        contour = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
        observed = self.observation(contour)
        holdout = np.arange(len(pixels)) % 2 == 0
        recovered, _, _ = fit_translation(previous, observed, Tracks(self.points, pixels, holdout), 1.)
        np.testing.assert_allclose(recovered, truth, atol=.03)
        pixels[holdout] += [9, -7]
        self.assertIsNone(fit_translation(previous, observed, Tracks(self.points, pixels, holdout), 1.))
