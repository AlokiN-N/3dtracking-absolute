"""Check that numerical state uses only its observed prefix and resets on loss."""
import unittest
import importlib.util
import numpy as np
from online.target import Target


@unittest.skipUnless(importlib.util.find_spec("gtsam"), "optional GTSAM dependency")
class StateContracts(unittest.TestCase):
    def test_reference_uncertainty_includes_orientation(self):
        from online.estimation.state import PoseState
        from online.geometry import DEFAULT_TARGET

        state = PoseState(DEFAULT_TARGET, 30)
        state.previous = np.zeros(6)
        state.covariance = np.zeros((9, 9))
        state.covariance[2, 2] = .01
        expected = .01 * np.dot(DEFAULT_TARGET.center[:2], DEFAULT_TARGET.center[:2])
        self.assertAlmostEqual(np.trace(state.reference_covariance()), expected)

    def test_absolute_correction_does_not_become_a_velocity_spike(self):
        from online.estimation.state import PoseState
        from online.estimation.measurements import PoseMeasurement
        from online.geometry import DEFAULT_TARGET

        state = PoseState(DEFAULT_TARGET, 30)
        baseline = np.array([10., 0., 2., 0., 0., 0.])
        state.step(0, baseline, None)
        measured = baseline.copy()
        measured[0] += 1
        result = state.step(1, baseline, PoseMeasurement(measured, np.eye(6) * .01, (1, 2), .2))
        self.assertLess(np.linalg.norm(result[:3] - measured[:3]), .1)
        self.assertLess(np.linalg.norm(state.velocity), 5)
        self.assertEqual(state.covariance.shape, (9, 9))
        self.assertGreater(np.linalg.eigvalsh(state.covariance).min(), 0)

    def test_prefix_and_reset(self):
        from online.estimation.state import PoseState
        vertices=np.array([[0,0,0],[2,0,0],[0,1,0],[0,0,1]],float)
        model=Target("asymmetric", (vertices,), speed=5)
        def sequence(count):
            state=PoseState(model,25)
            poses=[]
            for frame in range(count):
                baseline=np.array([10+frame*.04,.2,1.,frame*.005,.1,-.1])
                poses.append(state.step(frame,baseline,None))
            return poses,state
        full,state=sequence(20)
        prefix,_=sequence(11)
        np.testing.assert_array_equal(prefix,full[:11])
        self.assertIsNone(state.step(20,None,None))
        self.assertIsNone(state.previous)
        recovered=np.array([30.,0.,2.,.1,.1,.2])
        np.testing.assert_allclose(state.step(21,recovered,None),recovered,atol=1e-12)

if __name__=="__main__":
    unittest.main()
