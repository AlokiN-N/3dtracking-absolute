"""Check the output-only startup correction on a generic CAD target."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.camera import Camera
from online.estimation.cab_ray import CabRay
from online.geometry import View
from online.target import load_target


class CabRayTest(unittest.TestCase):
    def test_secondary_view_corrects_depth_until_first_multi(self):
        part = [[0, 1, 0], [0, 3, 0], [0, 2, 2]]
        with TemporaryDirectory() as directory:
            spec = Path(directory) / "target.json"
            spec.write_text(json.dumps({"parts": [part], "reference_part": part, "labels": [16],
                                        "origin": [0, 1, 0], "meters_per_unit": 0.5}))
            model = load_target(spec)
        detector = Mock()
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=detector))}):
            refiner = CabRay("test.pt", model)
        self.assertEqual(refiner.labels, (16,))
        np.testing.assert_allclose(refiner.part_center, [0, 0.5, 0.5])
        cameras = {
            i: View.of(Camera(i, np.array([0., y, 0.]), np.zeros(3), 35, (36, 24), 640, 480), model=model)
            for i, y in ((3, 0), (14, -600))
        }
        position = np.array([10., 0., 0.])
        pixels, _ = cameras[14].project((position + refiner.part_center)[None])
        pose = np.array([13., 0., 0., 0., 0., 0.])
        image = np.zeros((480, 640, 3), np.uint8)
        with patch.object(refiner, "detect", return_value=pixels[0]) as detect:
            self.assertIs(refiner.step(pose, "mono", {3}, cameras, {}), pose)
            detect.assert_not_called()
            corrected = refiner.step(pose, "mono", {3}, cameras, {3: image, 14: image})
            self.assertEqual(refiner.applied[0], [14])
            self.assertLess(np.linalg.norm(corrected[:3] - position), 0.4)
            np.testing.assert_array_equal(pose, [13., 0., 0., 0., 0., 0.])
            self.assertIs(refiner.step(pose, "multi", {3, 14}, cameras, {}), pose)
            self.assertIs(refiner.step(pose, "mono", {3}, cameras, {14: image}), pose)
            self.assertIsNone(refiner.applied)
            detect.assert_called_once()
