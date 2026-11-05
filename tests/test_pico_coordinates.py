"""Catch fractional-flow truncation and swapped crop axes."""
import unittest
import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None
from online.perception.appearance import model_points

@unittest.skipIf(torch is None, "optional PyTorch dependency")
class PicoCoordinates(unittest.TestCase):
    def test_fractional_translation_survives(self):
        size = 64
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        target = torch.stack([10 + x.T * 2., 20 + y.T * 3.])[None]
        source = torch.stack([x.float(), y.float(), torch.ones_like(x).float() * 5])[None]
        output = {"tar_pts_2d": target, "src_pts_3d": source, "tem_pose": torch.eye(4)[None]}
        flow = torch.zeros(1, 2, size, size)
        flow[:, 0], flow[:, 1] = .75, 1.25
        points, pixels = model_points(output, (flow, torch.full((1, 1, size, size), 10.)))
        expected = np.c_[10 + (points[:, 0] + .25) * 2, 20 + (points[:, 1] + .75) * 3]
        np.testing.assert_allclose(pixels, expected, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
