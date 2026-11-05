"""Нейросетевой детектор объекта в одной камере: YOLO11n (ONNX, onnxruntime) + окно поиска.

Онлайн-логика (как в контуре «захват -> сопровождение -> повторный захват»):
  * ЗАХВАТ: полный кадр нарезается на 2 перекрывающихся тайла 1080x1080 в полном
    разрешении (объект мелкий: на 500 м фура ~90 px) -> детекции любых классов, кроме «человек».
  * СОПРОВОЖДЕНИЕ: детектор запускается только в окне вокруг последнего положения
    объекта (окно = 3 размера объекта, не меньше 640 px) — в 5-6 раз быстрее.
  * ПОТЕРЯ: нет детекций в окне 5 кадров подряд -> снова полный проход; плюс полный
    проход каждые 30 кадров, чтобы не пропустить объект в другой части кадра.
Классы не фильтруются: обученная на COCO сеть называет белую фуру truck/airplane/boat/bench;
ложные срабатывания отсеивает 3D-фильтр (согласованность между камерами).
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

CONF = 0.05
MODEL = os.environ.get("YOLO_ONNX", os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolo", "yolo11n.onnx"))


class YoloDetector:
    def __init__(self, threads=1):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(MODEL, so, providers=[p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()])
        self.last = None; self.miss = 0; self.f = 0

    def _infer(self, img, sz, x0=0, y0=0):
        h, w = img.shape[:2]; r = sz / max(h, w)
        nw, nh = int(round(w * r)), int(round(h * r)); ph, pw = (32 - nh % 32) % 32, (32 - nw % 32) % 32
        im = cv2.copyMakeBorder(cv2.resize(img, (nw, nh)), 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        o = self.sess.run(None, {"images": im[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0})[0][0]
        sc = o[4:].max(0); cl = o[4:].argmax(0); k = (sc > CONF) & (cl != 0)
        b = o[:4, k].T / r
        return [(cx - bw / 2 + x0, cy - bh / 2 + y0, bw, bh, float(s)) for (cx, cy, bw, bh), s in zip(b, sc[k])]

    @staticmethod
    def _nms(d):
        if not d: return []
        idx = cv2.dnn.NMSBoxes([list(map(float, x[:4])) for x in d], [x[4] for x in d], CONF, 0.5)
        return [d[i] for i in np.array(idx).ravel()]

    def step(self, img):
        H, W = img.shape[:2]; self.f += 1
        full = self.last is None or self.miss >= 5 or self.f % 30 == 0
        if full:
            d = []
            for x0 in (0, W - H):
                d += self._infer(img[:, x0:x0 + H], 1088, x0, 0)
        else:
            cx, cy, s = self.last
            s = int(max(640, min(H, 3 * s)))
            x0 = int(np.clip(cx - s / 2, 0, W - s)); y0 = int(np.clip(cy - s / 2, 0, H - s))
            d = self._infer(img[y0:y0 + s, x0:x0 + s], 640, x0, y0)
        d = self._nms(d)
        if d:
            if self.last is None or full:
                b = max(d, key=lambda t: t[4])
            else:
                b = min(d, key=lambda t: np.hypot(t[0] + t[2] / 2 - self.last[0], t[1] + t[3] / 2 - self.last[1]))
            self.last = (b[0] + b[2] / 2, b[1] + b[3] / 2, max(b[2], b[3])); self.miss = 0
        else:
            self.miss += 1
        return [dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, area=w * h, score=s) for x, y, w, h, s in d]
