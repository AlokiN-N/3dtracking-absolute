"""Regression checks for crop coordinates, CAD origins and legacy imports."""
import importlib
import unittest
import numpy as np
from core.camera import Camera
from online.geometry import View, Rot
from online.perception.rendering import crop_transform, render_crop
from online.target import Target
from online.tracker import Tracker
from online.geometry import DEFAULT_TARGET

class GeometryContracts(unittest.TestCase):
    def test_pose_feedback_preserves_model_contract(self):
        pose = np.array([10., 1., 2., .1, .2, .3])
        velocity = np.array([1., 2., 3.])
        for model, dimensions in ((DEFAULT_TARGET, 5),
                                  (Target("free", (DEFAULT_TARGET.vertices,)), 6)):
            tracker = Tracker(None, 25, 3, model=model)
            tracker.accept_measurement(pose, velocity)
            np.testing.assert_array_equal(tracker.pose, pose[:dimensions])
            np.testing.assert_allclose(tracker.velocity, velocity / 25)
            self.assertIsNot(tracker.pose, pose)

    def test_import_compatibility(self):
        for name, package in (("appearance", "perception"), ("vision", "perception"),
                              ("joint_pose", "estimation"), ("evaluate", "evaluation")):
            self.assertIs(importlib.import_module(f"online.{name}"),
                          importlib.import_module(f"online.{package}.{name}"))

    def test_crop_inverse(self):
        transform = crop_transform(np.array([31., 63., 231., 183.]), 280)
        pixels = np.array([[46.25, 97.75], [184.5, 133.125]])
        mapped = pixels @ transform[:2, :2].T + transform[:2, 2]
        inverse = np.linalg.inv(transform)
        np.testing.assert_allclose(mapped @ inverse[:2, :2].T + inverse[:2, 2], pixels)

    def test_surface_coordinates_reproject(self):
        triangle = np.array([[0., -1., -1.], [0., 1., -1.], [0., 0., 1.]])
        model = Target("triangle", (triangle,), triangles=triangle[None],
                       face_colors=np.array([[200, 100, 50]]))
        camera = Camera(7, np.zeros(3), np.zeros(3), 35, (36, 24), 640, 480)
        view = View.of(camera, model=model)
        pose = np.array([10., 0., 0., 0., 0., 0.])
        rendered = render_crop(model, pose, view, np.array([240, 140, 400, 340]), 120)
        yy, xx = np.nonzero(rendered.mask)
        self.assertGreater(len(xx), 50)
        xyz = rendered.points[yy, xx] @ Rot(*pose[3:]).T + pose[:3]
        pixels, _ = view.project(xyz)
        np.testing.assert_allclose(pixels, rendered.original_pixels(np.c_[xx, yy]), atol=2e-5)

if __name__ == "__main__":
    unittest.main()
