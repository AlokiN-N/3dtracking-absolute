import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from core.camera import Camera
from core.dataset import load_scene, read_transform_csv
from core.video_sync import frame_indices
from online.estimation.speed import SpeedMeter
from online.run import sources


class InputTests(unittest.TestCase):
    def test_modern_export_supports_extra_cameras_and_zoom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "matrix_size.txt").write_text("23,76 mm * 13,365 mm", encoding="utf-8")
            (root / "Way2.FinalImage.CineCameraActor12.mp4").touch()
            (root / "object_transform_CineCameraActor12.csv").write_text(
                "Focal_Length_mm;Frame;Loc_Z;Loc_X;Loc_Y;Rot_Pitch;Rot_Yaw;Rot_Roll\n"
                "35;0;300;100;200;0;0;0\n"
                "60;1;300;100;200;0;0;0\n"
                "invalid;2;300;100;200;0;0;0\n", encoding="utf-8-sig")
            for tracks in (sources(root), load_scene(root, require_gt=False).cameras):
                self.assertEqual(set(tracks), {12})
                track = tracks[12]
                self.assertEqual(set(track.poses), {0, 1})
                camera = Camera.from_track(track, 1, 1920, 1080)
                np.testing.assert_array_equal(camera.loc, [100, 200, 300])
                self.assertAlmostEqual(camera.fx, 1920 * 60 / 23.76)

    def test_legacy_calibration_still_reads_embedded_sensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_bytes(("Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n"
                              "0;100;200;300;0;0;0;;;матрица;23,76*13,365мм\n").encode("cp1251"))
            poses, meta = read_transform_csv(path)
            self.assertEqual(set(poses), {0})
            self.assertEqual(meta["sensor_mm"], (23.76, 13.365))
            self.assertEqual(meta["focal_mm"], 35)
            self.assertEqual(meta["focal_by_frame"], {})

    def test_presentation_timestamps_preserve_offsets_and_gaps(self):
        self.assertEqual(frame_indices(["0.100", "0.033333", "0.133333"], Fraction(30)), {1, 3, 4})
        for times in (["0.033333", "0.033333"], ["0.05"], ["-0.033333"]):
            with self.assertRaises(ValueError):
                frame_indices(times, Fraction(30))


class SpeedTests(unittest.TestCase):
    def test_causal_speed_excludes_track_discontinuities(self):
        meter = SpeedMeter(30, 42)
        speeds = [meter.step(frame, np.array([frame / 3, 0.0, 0.0])) for frame in range(12)]
        self.assertEqual(speeds[:3], [None] * 3)
        np.testing.assert_allclose(speeds[3:], 10)
        self.assertIsNone(meter.step(12, np.array([50.0, 0.0, 0.0])))
        self.assertIsNone(meter.step(13, None))
        self.assertIsNone(meter.step(14, np.zeros(3)))
