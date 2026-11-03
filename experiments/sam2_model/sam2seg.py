import numpy as np, cv2, onnxruntime as ort
import os
# ONNX: huggingface.co/vietanhdev/segment-anything-2.1-onnx-models (sam2.1_hiera_small_20260221.zip)
M = os.environ.get('SAM2_ONNX', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'sam2', 'sam2.1_hiera_small'))
class SAM2:
    def __init__(s, threads=2):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.log_severity_level = 3
        s.enc = ort.InferenceSession(M + '.encoder.onnx', so); s.dec = ort.InferenceSession(M + '.decoder.onnx', so)
    def encode(s, crop):
        x = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (1024, 1024)).astype(np.float32) / 255
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        h0, h1, emb = s.enc.run(None, {'image': x})
        return dict(h0=h0, h1=h1, emb=emb, shape=crop.shape[:2])
    def decode(s, E, pts, labels):
        """pts в пикселях обрезки; labels: 1 — объект, 0 — фон, 2/3 — углы рамки"""
        H, W = E['shape']; p = np.array(pts, np.float32).copy(); p[:, 0] *= 1024 / W; p[:, 1] *= 1024 / H
        m, iou = s.dec.run(None, {'image_embed': E['emb'], 'high_res_feats_0': E['h0'], 'high_res_feats_1': E['h1'],
                                  'point_coords': p[None], 'point_labels': np.array(labels, np.float32)[None],
                                  'mask_input': np.zeros((1, 1, 256, 256), np.float32), 'has_mask_input': np.zeros(1, np.float32)})
        k = int(np.argmax(iou[0])); lg = cv2.resize(m[0, k], (W, H))
        return lg > 0, float(iou[0, k])

def crop_box(b, W=1920, H=1080, k=1.6, mn=256):
    x0, y0, x1, y1 = b; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(k * max(x1 - x0, y1 - y0), mn); s = min(s, H)
    a = int(np.clip(cx - s / 2, 0, W - s)); c = int(np.clip(cy - s / 2, 0, H - s))
    return a, c, int(s)
