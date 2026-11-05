"""SAM2 sessions consuming exactly one new image per camera and timestamp."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from online.perception.vision import Instance
from online.tracker import box_overlap


class VideoMasks:
    def __init__(self, checkpoint, device="cuda", revision=None):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.torch = torch
        self.device = device
        if revision is None:
            revision = {
                "facebook/sam2.1-hiera-tiny": "de431c4043854a71d8101e17995dfe596bf101a5",
                "facebook/sam2.1-hiera-base-plus": "b7320756a13354e7530a63935656d35b2f91a290",
                "facebook/sam2.1-hiera-large": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
            }.get(checkpoint)
        self.model = Sam2VideoModel.from_pretrained(checkpoint, revision=revision).to(device).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(checkpoint, revision=revision)
        self.revision = getattr(self.model.config, "_commit_hash", revision)
        self.sessions = {}

    def discard_missing(self, frame, camera_ids):
        self.sessions = {key: value for key, value in self.sessions.items()
                         if key in camera_ids and value[0] == frame - 1}

    def step(self, frame, key, image, seed):
        history = self.sessions.get(key)
        if history is not None and (history[0] != frame - 1 or history[1] != image.shape):
            history = None
        if history is None and seed is None:
            return None
        session = history[2] if history else self.processor.init_video_session(
            inference_device=self.device, max_vision_features_cache_size=1)
        index = history[3] + 1 if history else 0
        inputs = self.processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                return_tensors="pt").to(self.device)
        if history is None:
            self.processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=0, obj_ids=1,
                input_boxes=[[seed.box.tolist()]], original_size=tuple(image.shape[:2]))
        with self.torch.inference_mode():
            output = self.model(inference_session=session, frame_idx=index, frame=inputs.pixel_values[0])
        mask = self.processor.post_process_masks(
            [output.pred_masks], original_sizes=inputs.original_sizes, binarize=True)[0]
        mask = mask.squeeze().cpu().numpy().astype(np.uint8)
        # Streaming calls use explicit timestamps, so raw images can be evicted.
        session.processed_frames = {i: value for i, value in session.processed_frames.items() if i >= index - 1}
        memory = max(self.model.config.num_maskmem, self.model.config.max_object_pointers_in_encoder) + 1
        for outputs in session.output_dict_per_obj.values():
            outputs["non_cond_frame_outputs"] = {
                i: value for i, value in outputs["non_cond_frame_outputs"].items() if i >= index - memory}
        for tracked in session.frames_tracked_per_obj.values():
            for old in list(tracked):
                if old < index - memory:
                    del tracked[old]
        self.sessions[key] = (frame, image.shape, session, index)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 50:
            return None
        return Instance(contour[:, 0].astype(float), 0.8, -1, source="sam-video")


@dataclass
class MaskHistory:
    seed_frame: int
    frame: int
    shape: tuple
    mask: Instance
    pending: list = field(default_factory=list)


class OnDemandMasks:
    """Keep bounded, causal histories and run the reserve model only on request."""

    def __init__(self, backend, horizon):
        self.backend = backend
        self.horizon = horizon
        self.histories = {}

    def remember(self, frame, cameras, images, detections):
        for key, history in list(self.histories.items()):
            if (key not in cameras or history.frame != frame - 1 or
                    frame - history.seed_frame > self.horizon or images[key].shape != history.shape):
                del self.histories[key]
                self.backend.sessions.pop(key, None)
        for key in cameras:
            image = images[key]
            detection = detections.get(key)
            if detection is not None and not detection.flow:
                self.backend.sessions.pop(key, None)
                self.histories[key] = MaskHistory(frame, frame, image.shape, detection, [(frame, image, detection)])
            elif key in self.histories:
                history = self.histories[key]
                history.pending.append((frame, image, None))
                history.frame = frame

    def candidates(self, frame):
        eligible = {key: history for key, history in self.histories.items()
                    if history.seed_frame < frame == history.frame}
        if len(eligible) < 2:
            return {}
        result = {}
        for key, history in eligible.items():
            for index, image, seed in history.pending:
                mask = self.backend.step(index, key, image, seed)
                if seed is not None:
                    continue
                if mask is None or box_overlap(mask.box, history.mask.box) < 0.45:
                    del self.histories[key]
                    self.backend.sessions.pop(key, None)
                    break
                history.mask = mask
            else:
                history.pending.clear()
                result[key] = history.mask
        return result
