"""SAM2 sessions consuming exactly one new image per camera and timestamp."""
import cv2
import numpy as np

from online.perception.vision import Instance


class VideoMasks:
    def __init__(self, checkpoint, device="cuda", revision=None):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.torch = torch
        self.device = device
        if revision is None and checkpoint == "facebook/sam2.1-hiera-tiny":
            revision = "de431c4043854a71d8101e17995dfe596bf101a5"
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
