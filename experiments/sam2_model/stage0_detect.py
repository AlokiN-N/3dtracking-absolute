# Этап 0: кэш детекций по каждой камере (YOLO11n ONNX + блобы вычитания фона) -> cache/det_yolo_cam{i}.pkl, cache/det_cam{i}.pkl
# Нужны только для подсказок SAM 2 на первом проходе (этап 1).
import os, sys, pickle, time, cv2
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, ROOT)
from core.dataset import load_scene
from experiments.bgsub_ekf.cams import TrackCamera
from experiments.yolo_ekf.detector import YoloDetector
from experiments.bgsub_ekf.detect import Detector as MotionDetector
DET = os.environ.get('DET_DIR', 'cache'); os.makedirs(DET, exist_ok=True)
scene = load_scene(os.environ.get('PELENG_DATA', os.path.join(ROOT, 'data')))
for i in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else sorted(scene.cameras)):
    cam = TrackCamera(scene.cameras[i], scene.width, scene.height)
    cap = cv2.VideoCapture(str(scene.cameras[i].video)); yd = YoloDetector(threads=2); md = MotionDetector(cam)
    Y, Mo = [], []; f = 0; t0 = time.time()
    while True:
        ok, img = cap.read()
        if not ok: break
        Mo.append(md.step(img, f)); Y.append(yd.step(img)); f += 1
    pickle.dump(Y, open(f'{DET}/det_yolo_cam{i}.pkl', 'wb')); pickle.dump(Mo, open(f'{DET}/det_cam{i}.pkl', 'wb'))
    print('camera', i, f, 'frames', '%.0fs' % (time.time() - t0), flush=True)
