"""
MaST ONNX tracker (vendored from github.com/TsingWei/MaST @ 3d507e9, ECCV 2026).
SPATO ONNX tracker — fully self-contained, no PyTorch dependency.

Combines preprocessing, ONNX inference, and postprocessing in one file.
"""

import math
import multiprocessing
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
BoxXYWH = Union[Sequence[float], np.ndarray]


class NumpyPreprocessor:
    """ImageNet normalization: (img / 255 - mean) / std, HWC uint8 -> 1CHW float32."""

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]

    def process(self, img_arr: np.ndarray) -> np.ndarray:
        x = img_arr.astype(np.float32).transpose(2, 0, 1)[None, ...]  # 1,C,H,W
        x = x / 255.0
        return (x - self.mean) / self.std


def sample_target(
    im: np.ndarray,
    target_bb: BoxXYWH,
    search_area_factor: float,
    output_sz: int,
) -> Tuple[np.ndarray, float]:
    """Extract a square crop centered at target_bb, resized to output_sz."""
    if isinstance(target_bb, np.ndarray):
        x, y, w, h = [float(v) for v in target_bb.reshape(-1).tolist()]
    else:
        x, y, w, h = [float(v) for v in target_bb]

    crop_sz = int(math.ceil(math.sqrt(w * h) * float(search_area_factor)))
    if crop_sz < 1:
        raise ValueError("Too small bounding box.")

    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    x2 = x1 + crop_sz
    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    im_crop = im[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]
    im_crop_padded = cv2.copyMakeBorder(
        im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT
    )
    resize_factor = float(output_sz) / float(crop_sz)
    im_crop_padded = cv2.resize(im_crop_padded, (int(output_sz), int(output_sz)))
    return im_crop_padded, resize_factor


def transform_image_to_crop(
    box_in: BoxXYWH,
    box_extract: BoxXYWH,
    resize_factor: float,
    crop_sz: float,
    normalize: bool = False,
) -> np.ndarray:
    """Transform box coords from image space to crop space."""
    box_in = np.asarray(box_in, dtype=np.float32).reshape(-1)
    box_extract = np.asarray(box_extract, dtype=np.float32).reshape(-1)

    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (float(crop_sz) - 1.0) / 2.0 + (
        box_in_center - box_extract_center
    ) * float(resize_factor)
    box_out_wh = box_in[2:4] * float(resize_factor)

    box_out = np.concatenate(
        [box_out_center - 0.5 * box_out_wh, box_out_wh], axis=0
    ).astype(np.float32)
    if normalize:
        return box_out / (float(crop_sz) - 1.0)
    return box_out


def clip_box(box_xywh: List[float], H: int, W: int, margin: int = 0) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    x2, y2 = x + w, y + h
    x = min(max(0.0, x), float(W - margin))
    x2 = min(max(float(margin), x2), float(W))
    y = min(max(0.0, y), float(H - margin))
    y2 = min(max(float(margin), y2), float(H))
    w = max(float(margin), x2 - x)
    h = max(float(margin), y2 - y)
    return [x, y, w, h]


# ---------------------------------------------------------------------------
# Numpy helper functions
# ---------------------------------------------------------------------------
def _hann1d(side: int, centered: bool = True) -> np.ndarray:
    if side <= 0:
        raise ValueError("Feature map size must be positive.")
    if centered:
        idx = np.arange(1, side + 1, dtype=np.float32)
        return 0.5 * (1.0 - np.cos((2.0 * np.pi / (side + 1)) * idx))
    idx = np.arange(0, side // 2 + 1, dtype=np.float32)
    w = 0.5 * (1.0 + np.cos((2.0 * np.pi / (side + 2)) * idx))
    right = np.flip(w[1 : side - side // 2], axis=0)
    return np.concatenate([w, right], axis=0).astype(np.float32)


def _hann2d(side: int, centered: bool = True) -> np.ndarray:
    h1 = _hann1d(side, centered=centered)
    return np.outer(h1, h1).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=-1)


def _get_2d_gaussian_map_np(
    boxes: np.ndarray, heatmap_size: int = 16, alpha: float = 1.0
) -> np.ndarray:
    boxes_sized = boxes * heatmap_size
    cx = boxes_sized[:, 1]
    cy = boxes_sized[:, 0]
    w = boxes_sized[:, 3]
    h = boxes_sized[:, 2]

    grid = np.arange(heatmap_size, dtype=np.float32)
    x, y = np.meshgrid(grid, grid, indexing="ij")

    gauss = np.exp(
        -(
            (x[None] - cx[:, None, None]) ** 2
            / (alpha * w[:, None, None] + 1e-30)
            + (y[None] - cy[:, None, None]) ** 2
            / (alpha * h[:, None, None] + 1e-30)
        )
    )
    return gauss.astype(np.float32)


# ---------------------------------------------------------------------------
# SPATO ONNX Tracker
# ---------------------------------------------------------------------------
class SPATOTracker:
    """SPATO single-object tracker using ONNX Runtime only."""

    def __init__(self, model_cfg: dict, device_id: int = 0):
        """
        Args:
            model_cfg: dict with keys: onnx, search_size, template_size,
                       search_factor, template_factor, stride, use_reparam
            device_id: CUDA device id (0-based), ignored when no GPU.
        """
        self.search_size = int(model_cfg["search_size"])
        self.template_size = int(model_cfg["template_size"])
        self.search_factor = float(model_cfg["search_factor"])
        self.template_factor = float(model_cfg["template_factor"])
        stride = int(model_cfg["stride"])

        self.preprocessor = NumpyPreprocessor()
        self.state = None
        self.frame_id = 0

        self.feat_sz = max(1, self.search_size // max(1, stride))
        self.output_window = _hann2d(self.feat_sz, centered=True)

        self.use_reparam = bool(model_cfg.get("use_reparam", True))
        self.reparam_alpha = 4.0

        norm_coords = np.arange(self.feat_sz, dtype=np.float32) / float(self.feat_sz)
        self.grid_x = np.tile(norm_coords[None, :], (self.feat_sz, 1))
        self.grid_y = self.grid_x.T

        # Create ONNX Runtime session
        onnx_path = model_cfg["onnx"]
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        self.ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=[p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()],
        )
        self.ort_input_names = [i.name for i in self.ort_session.get_inputs()]
        self.ort_output_names = [o.name for o in self.ort_session.get_outputs()]

        self._input_signature = self._infer_input_signature(self.ort_input_names)
        self._output_mode = self._infer_output_mode(self.ort_output_names)

        self.template = None
        self.template_anno = None



    # --- Input/output signature detection ---

    @staticmethod
    def _infer_input_signature(input_names) -> str:
        nset = set(input_names)
        if {"template", "search", "template_anno", "task_index"}.issubset(nset):
            return "modern"
        if {"template", "search", "template_anno"}.issubset(nset):
            return "modern_no_task"
        if {"template", "search", "reparam"}.issubset(nset):
            return "template_search_reparam"
        if {"template", "search"}.issubset(nset):
            return "template_search"
        if {"zf", "x"}.issubset(nset):
            return "legacy_zfx"
        raise RuntimeError(f"Unsupported ORT input signature: {input_names}")

    @staticmethod
    def _infer_output_mode(output_names) -> str:
        nset = set(output_names)
        if {"score_map", "pred_boxes"}.issubset(nset):
            return "mlp_scores_boxes"
        if {"score_map", "size_map", "offset_map"}.issubset(nset):
            return "center_maps"
        if {"best_box", "best_score"}.issubset(nset):
            return "best_box"
        if "pred_boxes" in nset:
            return "pred_only"
        raise RuntimeError(f"Unsupported ORT outputs: {output_names}")

    # --- Core tracking API ---

    def initialize(self, image: np.ndarray, info: dict):
        """Initialize tracker with first frame and bounding box.

        Args:
            image: HWC uint8 RGB image.
            info: dict with 'init_bbox' as [x, y, w, h].
        """
        z_patch_arr, resize_factor = sample_target(
            image, info["init_bbox"], self.template_factor, self.template_size
        )
        self.template = self.preprocessor.process(z_patch_arr).astype(np.float32)

        init_box = np.asarray(info["init_bbox"], dtype=np.float32)
        template_box = transform_image_to_crop(
            init_box, init_box, resize_factor, float(self.template_size), normalize=True
        )
        self.template_anno = template_box[None, :].astype(np.float32)

        self.state = [float(v) for v in info["init_bbox"]]
        self.frame_id = 0

    def track(self, image: np.ndarray) -> dict:
        """Track object in a new frame.

        Args:
            image: HWC uint8 RGB image.

        Returns:
            dict with 'target_bbox' [x, y, w, h] and 'best_score'.
        """
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor = sample_target(
            image, self.state, self.search_factor, self.search_size
        )
        search = self.preprocessor.process(x_patch_arr).astype(np.float32)

        reparam = None
        if self.use_reparam:
            state_w = float(self.state[2])
            state_h = float(self.state[3])
            w_norm = state_w * float(resize_factor) / float(self.search_size)
            h_norm = state_h * float(resize_factor) / float(self.search_size)
            last_box = np.array([[0.5, 0.5, w_norm, h_norm]], dtype=np.float32)
            gmap = _get_2d_gaussian_map_np(
                last_box, self.feat_sz, alpha=self.reparam_alpha
            )
            reparam = gmap.reshape(1, -1).astype(np.float32)

        out = self._ort_forward(search, reparam=reparam)
        best_box_xyxy, best_score = self._select_best_xyxy(out)

        best_box_cxcywh = _xyxy_to_cxcywh(best_box_xyxy)
        pred = best_box_cxcywh[0] * (float(self.search_size) / float(resize_factor))
        self.state = clip_box(
            self._map_box_back(pred.tolist(), float(resize_factor)), H, W, margin=10
        )

        return {"target_bbox": self.state, "best_score": float(best_score[0])}

    # --- Internal methods ---

    def _ort_forward(
        self, search: np.ndarray, reparam: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        provided = {}
        if self._input_signature in ("modern", "modern_no_task"):
            provided["template"] = self.template
            provided["search"] = search
            provided["template_anno"] = self.template_anno
            if self._input_signature == "modern":
                provided["task_index"] = np.zeros((search.shape[0],), dtype=np.int64)
        elif self._input_signature == "template_search_reparam":
            provided["template"] = self.template
            provided["search"] = search
            if reparam is not None:
                provided["reparam"] = reparam
            else:
                provided["reparam"] = np.ones(
                    (search.shape[0], self.feat_sz * self.feat_sz), dtype=np.float32
                )
        elif self._input_signature == "template_search":
            provided["template"] = self.template
            provided["search"] = search
        elif self._input_signature == "legacy_zfx":
            provided["zf"] = self.template
            provided["x"] = search

        ort_inputs = {name: provided[name] for name in self.ort_input_names}
        ort_outputs = self.ort_session.run(self.ort_output_names, ort_inputs)
        return {name: out for name, out in zip(self.ort_output_names, ort_outputs)}

    def _select_best_xyxy(
        self, ort_out: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._output_mode == "mlp_scores_boxes":
            score_map = np.asarray(ort_out["score_map"], dtype=np.float32)
            pred_boxes = np.asarray(ort_out["pred_boxes"], dtype=np.float32)
            global_index_s = ort_out.get("global_index_s", None)
            if global_index_s is not None:
                global_index_s = np.asarray(global_index_s, dtype=np.int64)

            HW = self.feat_sz * self.feat_sz
            N_score = score_map.reshape(score_map.shape[0], -1).shape[1]

            is_sparse = (
                global_index_s is not None
                and N_score == global_index_s.shape[1]
                and N_score < HW
            )

            if is_sparse:
                hann_flat = self.output_window.reshape(-1)
                hann_weights = hann_flat[global_index_s]
                weighted_score = (
                    _sigmoid(score_map.reshape(score_map.shape[0], -1)) * hann_weights
                )
                best_idx = np.argmax(weighted_score, axis=1)
                best_score = weighted_score[
                    np.arange(weighted_score.shape[0]), best_idx
                ]
                boxes_flat = pred_boxes.reshape(pred_boxes.shape[0], -1, 4)
                best_box_xyxy = boxes_flat[
                    np.arange(boxes_flat.shape[0]), best_idx
                ]
                return best_box_xyxy.astype(np.float32), best_score.astype(np.float32)

            score_flat = score_map.reshape(score_map.shape[0], -1)
            boxes_flat = pred_boxes.reshape(pred_boxes.shape[0], -1, 4)

            if global_index_s is not None and global_index_s.shape[1] < HW:
                B = score_flat.shape[0]
                hann_flat = self.output_window.reshape(-1)
                idx = global_index_s
                score_kept = np.take_along_axis(
                    _sigmoid(score_flat), idx, axis=1
                )
                hann_kept = hann_flat[idx]
                weighted = score_kept * hann_kept
                best_i = np.argmax(weighted, axis=1)
                best_score = weighted[np.arange(B), best_i]
                best_global = idx[np.arange(B), best_i]
                best_box_xyxy = boxes_flat[np.arange(B), best_global]
                return best_box_xyxy.astype(np.float32), best_score.astype(np.float32)

            score_2d = _sigmoid(score_flat).reshape(
                score_flat.shape[0], self.feat_sz, self.feat_sz
            )
            response = score_2d * self.output_window[None, ...]
            flat_resp = response.reshape(response.shape[0], -1)
            best_idx = np.argmax(flat_resp, axis=1)
            best_score = flat_resp[np.arange(flat_resp.shape[0]), best_idx]
            best_box_xyxy = boxes_flat[np.arange(boxes_flat.shape[0]), best_idx]
            return best_box_xyxy.astype(np.float32), best_score.astype(np.float32)

        if self._output_mode == "center_maps":
            score_map = np.asarray(ort_out["score_map"], dtype=np.float32)
            size_map = np.asarray(ort_out["size_map"], dtype=np.float32)
            offset_map = np.asarray(ort_out["offset_map"], dtype=np.float32)

            if score_map.ndim == 4 and score_map.shape[1] == 1:
                score_map = score_map[:, 0]
            elif (
                score_map.ndim == 2
                and score_map.shape[1] == self.feat_sz * self.feat_sz
            ):
                score_map = score_map.reshape(
                    score_map.shape[0], self.feat_sz, self.feat_sz
                )
            if size_map.ndim == 3:
                size_map = size_map[None, ...]
            if offset_map.ndim == 3:
                offset_map = offset_map[None, ...]

            response = _sigmoid(score_map) * self.output_window[None, ...]
            flat_score = response.reshape(response.shape[0], -1)
            best_idx = np.argmax(flat_score, axis=1)
            best_score = flat_score[np.arange(flat_score.shape[0]), best_idx]

            boxes = self._decode_boxes_from_maps(size_map, offset_map)
            best_box_xyxy = boxes[np.arange(boxes.shape[0]), best_idx]
            return best_box_xyxy.astype(np.float32), best_score.astype(np.float32)

        if self._output_mode == "best_box":
            best_box = np.asarray(ort_out["best_box"], dtype=np.float32).reshape(-1, 4)
            best_score = np.asarray(ort_out["best_score"], dtype=np.float32).reshape(-1)
            return best_box, best_score

        # pred_only
        best_box = np.asarray(ort_out["pred_boxes"], dtype=np.float32).reshape(-1, 4)
        best_score = np.ones((best_box.shape[0],), dtype=np.float32)
        return best_box, best_score

    def _decode_boxes_from_maps(
        self, size_map: np.ndarray, offset_map: np.ndarray
    ) -> np.ndarray:
        x_centers = self.grid_x[None, ...] + offset_map[:, 0]
        y_centers = self.grid_y[None, ...] + offset_map[:, 1]
        half_w = size_map[:, 0] * 0.5
        half_h = size_map[:, 1] * 0.5
        x1 = x_centers - half_w
        y1 = y_centers - half_h
        x2 = x_centers + half_w
        y2 = y_centers + half_h
        bs = size_map.shape[0]
        return np.stack([x1, y1, x2, y2], axis=-1).reshape(bs, -1, 4)

    def _map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = float(self.state[0]) + 0.5 * float(self.state[2])
        cy_prev = float(self.state[1]) + 0.5 * float(self.state[3])
        cx, cy, w, h = [float(v) for v in pred_box]
        half_side = 0.5 * float(self.search_size) / float(resize_factor)
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
