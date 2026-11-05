"""Online TAPNext tracks anchored to visible CAD surface points."""
from pathlib import Path
import sys

import cv2
import numpy as np

from online.estimation.measurements import Correspondences
from online.estimation.surface_pose import Surface, seed_tracks


class PointTracker:
    def __init__(self, source, checkpoint, model):
        import torch

        sys.path.insert(0, str(Path(source).resolve()))
        from tapnet.tapnext.tapnext_torch import TAPNext
        from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint

        self.network = restore_model_from_jax_checkpoint(TAPNext(image_size=(256, 256)), checkpoint)
        self.network = self.network.cuda().eval()
        self.torch = torch
        self.surface = Surface(model)
        self.states = {}

    def frame_tensor(self, image):
        rgb = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2RGB)
        return self.torch.as_tensor(rgb, device="cuda", dtype=self.torch.float32)[None, None] / 127.5 - 1

    def reanchor(self, frame, pose, images, views):
        """Attach fresh tracks after an independently verified metric correction."""
        self.states.clear()
        self.correspondences(frame, pose, images, views)

    def correspondences(self, frame, pose, images, views):
        active = {view.camera.index for view in views}
        self.states = {key: value for key, value in self.states.items()
                       if key in active and value[0] == frame - 1}
        groups = []
        for view in views:
            key = view.camera.index
            image = images[key]
            scale = np.array(image.shape[:2][::-1]) / 256
            history = self.states.get(key)
            if history is None or history[1] != image.shape:
                tracks = seed_tracks(self.surface, view, pose, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                if tracks is None:
                    continue
                query = np.c_[np.zeros(len(tracks.pixels)), (tracks.pixels / scale)[:, ::-1]]
                query = self.torch.as_tensor(query, device="cuda", dtype=self.torch.float32)[None]
                with self.torch.inference_mode():
                    _, _, _, state = self.network(self.frame_tensor(image), query_points=query)
                self.states[key] = (frame, image.shape, state, tracks.points)
                continue
            with self.torch.inference_mode():
                pixels, _, visibility, state = self.network(self.frame_tensor(image), state=history[2])
            pixels = pixels[0, 0].float().cpu().numpy()[:, ::-1] * scale
            confidence = visibility[0, 0].sigmoid().float().cpu().numpy().reshape(-1)
            confidence[confidence < 0.5] = 0
            groups.append(Correspondences(view, history[3], pixels, confidence))
            if np.count_nonzero(confidence) >= 16:
                self.states[key] = (frame, image.shape, state, history[3])
            else:
                self.states.pop(key, None)
        return groups
