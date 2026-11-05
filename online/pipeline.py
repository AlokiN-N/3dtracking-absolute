"""Optional modern measurements sharing a common calibrated pose solver."""
from contextlib import contextmanager

import numpy as np

from online.estimation.measurements import fit_measurement
from online.estimation.joint_pose import motion_evidence, motion_supports
from online.geometry import Observation, Rot
from online.perception.verification import boundaries_support


class MeasurementPipeline:
    def __init__(self, model, fps, gotrack=None, matcher=None, points=None, state=False, sequential=False, verify=True):
        self.model = model
        self.sequential = sequential
        self.verify = verify
        self.previous_baseline = None
        self.gotrack = gotrack
        self.matcher = matcher
        self.points = points
        self.state = None
        if state:
            from online.estimation.state import PoseState

            self.state = PoseState(model, fps)

    @classmethod
    def from_options(cls, options, model, fps):
        gotrack = matcher = points = None
        if options.gotrack_source:
            from online.perception.gotrack import GoTrackRefiner

            gotrack = GoTrackRefiner(options.gotrack_source, options.gotrack_checkpoint, model, options.renderer)
            if options.sequential_models:
                gotrack.network.cpu()
        if options.roma_source:
            from online.perception.matching import RoMaMatcher

            matcher = RoMaMatcher(options.roma_source, model)
            if options.sequential_models:
                matcher.network.cpu()
        if options.tapnext_source:
            from online.perception.point_tracking import PointTracker

            points = PointTracker(options.tapnext_source, options.tapnext_checkpoint, model)
            if options.sequential_models:
                points.network.cpu()
        return cls(model, fps, gotrack, matcher, points, options.unified,
                   options.sequential_models, not options.unverified_measurements)

    def step(self, frame, pose, images, views, memories):
        previous = self.previous_baseline
        self.previous_baseline = None if pose is None else pose.copy()
        sources = []
        evidence = motion_evidence(frame, previous, pose, views, memories) if pose is not None and self.verify else []
        # A nonnegative residual cannot improve by more than its current value.
        can_measure = not self.verify or sum(item.before / item.sigma >= 0.5 for item in evidence) >= 2
        if pose is not None:
            if self.gotrack is not None and can_measure:
                sources.append(("gotrack", self.measure(self.gotrack, pose, images, views)))
            if self.matcher is not None and can_measure:
                sources.append(("roma", self.measure(self.matcher, pose, images, views)))
            if self.points is not None:
                tracked = self.measure(self.points, frame, pose, images, views)
                if can_measure:
                    sources.append(("tapnext", tracked))
        elif self.points is not None:
            self.points.states.clear()
        accepted = []
        for source, groups in sources:
            candidate = fit_measurement(pose, groups, self.model) if groups else None
            if candidate is not None and self.accepts(frame, previous, pose, candidate, images, views, memories, evidence):
                candidate.source = source
                accepted.append(candidate)
        measurement = min(accepted, key=lambda item: item.reference_variance(self.model), default=None)
        if measurement is not None:
            rotation = Rot(*measurement.pose[3:])
            for alternative in accepted:
                if (np.linalg.norm(measurement.pose[:3] - alternative.pose[:3]) > .035 * self.model.size
                        or np.trace(rotation.T @ Rot(*alternative.pose[3:])) < 1 + 2 * np.cos(.12)):
                    measurement = None
                    break
        if self.state is not None:
            output = self.state.step(frame, pose, measurement)
        else:
            output = measurement.pose if measurement is not None else pose
        if measurement is not None and self.points is not None and measurement.source != "tapnext":
            anchors = [view for view in views if view.camera.index in measurement.cameras]
            with self.active(self.points):
                self.points.reanchor(frame, output, images, anchors)
        return output, measurement

    def accepts(self, frame, previous, baseline, candidate, images, views, memories, evidence):
        witnesses = [view for view in views if view.camera.index in candidate.cameras]
        observations = [Observation(view) for view in witnesses]
        if any(observation.overlap(candidate.pose) < observation.overlap(baseline) - .05
               for observation in observations):
            return False
        if not self.verify:
            return True
        relevant = [item for item in evidence if item.view.camera.index in candidate.cameras]
        return (motion_supports(frame, previous, baseline, candidate.pose, witnesses, memories, relevant)
                and boundaries_support(baseline, candidate.pose, images, witnesses, self.model))

    def measure(self, provider, *args):
        with self.active(provider):
            return provider.correspondences(*args)

    @contextmanager
    def active(self, provider):
        if self.sequential:
            provider.network.cuda()
        try:
            yield
        finally:
            if self.sequential:
                provider.network.cpu()
                provider.torch.cuda.empty_cache()
