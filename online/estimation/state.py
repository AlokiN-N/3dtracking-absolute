"""One causal pose posterior, with metric translation and full rotation."""
import numpy as np

from online.geometry import Rot, center_of, pose_angles, skew


class PoseState:
    def __init__(self, model, fps):
        import gtsam

        self.gtsam = gtsam
        self.model = model
        self.dt = 1 / fps
        self.previous = None
        self.velocity = np.zeros(3)
        self.covariance = None
        self.frame = -1

    def reference_covariance(self):
        """Propagate center and orientation uncertainty to the requested CAD origin."""
        if self.previous is None or self.covariance is None:
            return None
        rotation = Rot(*self.previous[3:])
        jacobian = np.hstack([rotation @ skew(self.model.center), rotation, np.zeros((3, 3))])
        return jacobian @ self.covariance @ jacobian.T

    def step(self, frame, baseline, measurement):
        if frame != self.frame + 1:
            raise ValueError("State updates must be consecutive")
        self.frame = frame
        if baseline is None:
            self.previous = None
            self.covariance = None
            self.velocity[:] = 0
            return None
        if measurement is None:
            if self.previous is not None:
                movement = (center_of(baseline, self.model) - center_of(self.previous, self.model)) / self.dt
                if np.linalg.norm(movement) <= self.model.speed:
                    self.velocity = movement
            self.previous = baseline.copy()
            self.covariance = np.diag([.15**2] * 3 + [(.15 * self.model.size)**2] * 3 +
                                      [(.5 * self.model.speed)**2] * 3)
            return baseline
        gtsam = self.gtsam
        pose = measurement.pose
        rotation = Rot(*pose[3:])
        center = center_of(pose, self.model)
        observed = gtsam.Pose3(gtsam.Rot3(rotation), center)
        reorder = np.zeros((6, 6))
        reorder[:3, 3:] = rotation.T
        reorder[3:, :3] = rotation.T
        covariance = reorder @ measurement.covariance @ reorder.T + np.eye(6) * 1e-8
        observation_noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.PriorFactorPose3(0, observed, observation_noise))
        prior_velocity = self.velocity.copy()
        if self.previous is not None:
            previous_rotation = Rot(*self.previous[3:])
            predicted_center = center_of(self.previous, self.model) + self.velocity * self.dt
            predicted = gtsam.Pose3(gtsam.Rot3(previous_rotation), predicted_center)
            transition = np.eye(9)
            transition[3:6, 6:] = previous_rotation.T * self.dt
            acceleration = np.zeros((9, 3))
            acceleration[3:6] = previous_rotation.T * (.5 * self.dt**2)
            acceleration[6:] = np.eye(3) * self.dt
            process = acceleration @ acceleration.T * (.5 * self.model.speed)**2
            process[:3, :3] = np.eye(3) * .5**2 * self.dt
            prior_covariance = transition @ self.covariance @ transition.T + process + np.eye(9) * 1e-9

            def prior_error(factor, values, jacobians):
                current = values.atPose3(0)
                error = np.r_[predicted.localCoordinates(current), values.atVector(1) - prior_velocity]
                if jacobians is not None:
                    derivative = np.zeros((9, 6), order="F")
                    step = 1e-6
                    for axis in range(6):
                        delta = np.zeros(6)
                        delta[axis] = step
                        derivative[:6, axis] = (
                            predicted.localCoordinates(current.retract(delta)) -
                            predicted.localCoordinates(current.retract(-delta))) / (2 * step)
                    jacobians[0] = derivative
                    jacobians[1] = np.asfortranarray(np.vstack([np.zeros((6, 3)), np.eye(3)]))
                return error

            graph.add(gtsam.CustomFactor(gtsam.noiseModel.Gaussian.Covariance(prior_covariance),
                                         [0, 1], prior_error))
        else:
            graph.add(gtsam.PriorFactorVector(
                1, prior_velocity, gtsam.noiseModel.Isotropic.Sigma(3, .5 * self.model.speed)))
        initial = gtsam.Values()
        initial.insert(0, observed)
        initial.insert(1, prior_velocity)
        solution = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
        result = solution.atPose3(0)
        self.covariance = gtsam.Marginals(graph, solution).jointMarginalCovariance([0, 1]).fullMatrix()
        rotation = result.rotation().matrix()
        center = result.translation()
        output = np.r_[center - rotation @ self.model.center, pose_angles(rotation)]
        velocity = solution.atVector(1)
        self.velocity = velocity * min(1., self.model.speed / max(np.linalg.norm(velocity), 1e-9))
        self.previous = output.copy()
        return output
