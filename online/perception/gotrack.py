"""GoTrack neural correspondence estimation in a current-pose CAD render."""
from pathlib import Path

import cv2
import numpy as np

from online.estimation.measurements import Correspondences
from online.perception.rendering import image_crop, render_crop
from online.perception.vendor import source_modules


class GoTrackRefiner:
    def __init__(self, source, checkpoint, model, renderer="numpy"):
        import torch

        source = Path(source).resolve()
        with source_modules(source, ("model", "utils")):
            from model.blocks.config import DecoderOpts
            from model.blocks.decoder import Decoder
            from model.heads.dpt.config import DPTHeadOpts
            from model.heads.dpt.model import DPTHead

            network = torch.nn.Module()
            network.backbone = torch.nn.Module()
            network.backbone.model = torch.hub.load(
                str(source.parent / "dinov2"), "dinov2_vits14_reg", source="local", pretrained=False)
            network.decoder = Decoder(DecoderOpts())
            network.pose_head = DPTHead(DPTHeadOpts())
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["model_state_dict"]
        state = {key.removeprefix("models.1."): value for key, value in state.items()
                 if key.startswith("models.1.")}
        network.load_state_dict(state, strict=True)
        self.network = network.eval().cuda()
        self.torch = torch
        self.model = model
        self.render = render_crop
        if renderer == "pytorch3d":
            from online.perception.pytorch_rendering import render_crop as torch_render

            self.render = torch_render

    def correspondences(self, pose, images, views):
        result = []
        torch = self.torch
        for view in views:
            if min(view.instance.box[2:] - view.instance.box[:2]) < 32:
                continue
            rendered = self.render(self.model, pose, view, view.instance.box)
            query = cv2.cvtColor(image_crop(images[view.camera.index], rendered.transform, 280), cv2.COLOR_BGR2RGB)
            template = rendered.rgb.copy()
            template[~rendered.mask] = 127
            tensors = torch.as_tensor(np.stack([query, template]), device="cuda", dtype=torch.float32)
            tensors = tensors.permute(0, 3, 1, 2) / 255
            mean = tensors.new_tensor([.485, .456, .406])[None, :, None, None]
            std = tensors.new_tensor([.229, .224, .225])[None, :, None, None]
            with torch.inference_mode():
                features = self.network.backbone.model.get_intermediate_layers(
                    (tensors - mean) / std, n=[9], reshape=True, norm=True)[0]
                _, decoded = self.network.decoder(features[:1], features[1:], (280, 280))
                flow, confidence = self.network.pose_head(decoded, (280, 280))
            flow = flow[0].permute(1, 2, 0).float().cpu().numpy()
            confidence = confidence.squeeze().float().cpu().numpy()
            yy, xx = np.indices(rendered.mask.shape)
            pixels = np.stack([xx, yy], axis=-1) + flow
            keep = rendered.mask & (confidence >= 0.5) & np.isfinite(pixels).all(-1)
            keep &= (pixels >= 0).all(-1) & (pixels < 280).all(-1)
            result.append(Correspondences(view, rendered.points[keep],
                                         rendered.original_pixels(pixels[keep]), confidence[keep]))
        return result
