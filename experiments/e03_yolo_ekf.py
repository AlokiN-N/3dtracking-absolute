"""E03 — нейросетевой детектор (YOLO11n, ONNX) + проверка движения + многокамерный 3D EKF.

1. Детекция (`yolo_ekf/detector.py`): захват — полный кадр тайлами в полном разрешении,
   сопровождение — детектор только в окне вокруг последнего положения объекта,
   повторный захват — полный проход при потере (и каждые 30 кадров).
2. Проверка движения: рамка детектора принимается, только если в ней есть движущиеся
   пиксели (маска вычитания фона из `bgsub_ekf/detect.py`). Без неё сеть, обученная на COCO,
   стабильно срабатывает на неподвижные деревья/склоны во всех камерах, и такие ложные
   срабатывания согласованно триангулируются (MAE 84 м). ТЗ: объект — движущийся.
3. 3D-трекер (`bgsub_ekf/fuse.py`, тот же, что в e01): EKF [x,y,z,vx,vy,vz], измерение —
   центр рамки детектора в каждой камере; совместная ассоциация по камерам, отсев выбросов
   по хи-квадрат, несколько гипотез, перепривязка, работа при 1..N камерах.
4. Выход — 3D-центр объекта. Параметры не подбираются по GT. Строго онлайн.
"""
from __future__ import annotations
import queue, threading, time
import cv2, numpy as np
from core.dataset import Scene
from core.validation import Prediction
from experiments.base import Experiment
from experiments.bgsub_ekf.cams import TrackCamera
from experiments.bgsub_ekf.fuse import Tracker
from experiments.yolo_ekf.detector import YoloDetector
from experiments.bgsub_ekf.detect import Detector as MotionDetector


def moving(b, blobs, frac=0.05):
    """В рамке детектора есть движущиеся пиксели (пересечение с блобом движения)."""
    for m in blobs:
        ix = min(b['x'] + b['w'], m['x'] + m['w']) - max(b['x'], m['x'])
        iy = min(b['y'] + b['h'], m['y'] + m['h']) - max(b['y'], m['y'])
        if ix > 0 and iy > 0 and ix * iy > frac * b['w'] * b['h']:
            return True
    return False


class YoloEKF(Experiment):
    name = "e03_yolo_ekf"

    def __init__(self, **_) -> None:
        pass

    def run(self, scene: Scene) -> Prediction:
        idx = sorted(i for i in scene.cameras if scene.cameras[i].video is not None)
        cams = [TrackCamera(scene.cameras[i], scene.width, scene.height) for i in idx]
        caps = [cv2.VideoCapture(str(scene.cameras[i].video)) for i in idx]
        dets = [YoloDetector(threads=1) for _ in idx]
        motion = [MotionDetector(c) for c in cams]
        tracker = Tracker(cams, merge=0)   # рамка детектора — весь объект, склейка не нужна

        def worker(k, qin, qout):
            while True:
                f = qin.get()
                if f is None: return
                ok, img = caps[k].read()
                if not ok:
                    qout.put((k, None)); continue
                blobs = motion[k].step(img, f)
                qout.put((k, [b for b in dets[k].step(img) if moving(b, blobs)]))
        qins = [queue.Queue() for _ in cams]; qout = queue.Queue()
        for k in range(len(cams)):
            threading.Thread(target=worker, args=(k, qins[k], qout), daemon=True).start()
        xyz, n_cams = {}, {}; t0 = time.time(); f = 0
        while True:
            for q in qins: q.put(f)
            res = dict(qout.get() for _ in cams)
            if any(res[k] is None for k in range(len(cams))): break
            out, _, used = tracker.step(f, [res[k] for k in range(len(cams))])
            if out is not None:
                xyz[f] = out; n_cams[f] = int(used)
            if f % 100 == 0:
                print(f"  кадр {f}: камер {used}, {f / (time.time() - t0 + 1e-9):.1f} fps", flush=True)
            f += 1
        for q in qins: q.put(None)
        return Prediction(name=self.name, xyz=xyz, n_cams=n_cams,
                          meta={"detector": "YOLO11n ONNX + окно поиска + проверка движения", "frames": len(xyz),
                                "fps": round(f / (time.time() - t0), 2),
                                "mean_cameras_used": float(np.mean(list(n_cams.values()))) if n_cams else 0.0})
