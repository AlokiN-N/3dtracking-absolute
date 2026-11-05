"""Publication must preserve every previously available scored prediction."""
import unittest

from online.evaluation.compare import compare


def row(frame, error, xyz=None):
    return {"frame": frame, "error_m": error, "xyz": xyz if xyz is not None else [0., 0., 0.]}


class ReleaseGate(unittest.TestCase):
    def test_equal_maximum_with_other_improvements_is_allowed(self):
        report = compare([row(0, 2), row(1, 5)], [row(0, 1), row(1, 5)])
        self.assertTrue(report["release_passes"])
        self.assertFalse(report["passes"])

    def test_any_regression_blocks_publication(self):
        report = compare([row(0, 2), row(1, 5)], [row(0, 1), row(1, 5.001)])
        self.assertFalse(report["release_passes"])
        self.assertEqual(report["worsened_frames"], [1])

    def test_unscored_existing_coordinate_must_match(self):
        before = [row(0, 2), row(1, None, [1., 0., 0.])]
        after = [row(0, 1), row(1, None, [2., 0., 0.])]
        self.assertFalse(compare(before, after)["release_passes"])

    def test_missing_prediction_blocks_publication(self):
        before = [row(0, 2), row(1, 5)]
        after = [row(0, 1), row(1, None, [None, None, None])]
        self.assertFalse(compare(before, after)["release_passes"])


if __name__ == "__main__":
    unittest.main()
