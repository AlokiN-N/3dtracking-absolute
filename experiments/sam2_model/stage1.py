# Этап 1: грубая 3D-траектория (e03, онлайн) -> 2D-рамки-подсказки для SAM 2 в каждой камере
import os, sys, pickle, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, ROOT)
DET = os.environ.get('DET_DIR', 'cache')
from core.dataset import load_scene
from experiments.bgsub_ekf.cams import TrackCamera
from experiments.bgsub_ekf.fuse import Tracker, merge_blobs
from experiments.e03_yolo_ekf import moving
scene = load_scene(os.environ.get('PELENG_DATA', os.path.join(ROOT, 'data')))
idx = sorted(scene.cameras); cams = [TrackCamera(scene.cameras[i]) for i in idx]
det = [pickle.load(open(f'{DET}/det_yolo_cam{i}.pkl', 'rb')) for i in idx]
mot = [pickle.load(open(f'{DET}/det_cam{i}.pkl', 'rb')) for i in idx]
tr = Tracker(cams, merge=0); xyz = {}; OBS = []
N = min(map(len, det))
for f in range(N):
    obs = []
    for k in range(len(idx)):
        dd = [b for b in det[k][f] if moving(b, mot[k][f])]
        obs.append(dd if dd else merge_blobs(mot[k][f], k=1.0))
    OBS.append(obs)
    o, _, u = tr.step(f, obs)
    if o is not None: xyz[f] = o
fs = np.array(sorted(xyz)); P = np.array([xyz[f] for f in fs])
boxes = {}   # (k, f) -> [x0, y0, x1, y1]
for f, X in zip(fs, P):
    for k in range(len(idx)):
        try:
            uv, z = cams[k].project(f, X) if hasattr(cams[k], 'project') else (None, None)
        except Exception:
            uv = None
        if uv is None:
            p = cams[k].P(f) @ np.append(X, 1); z = p[2]; uv = p[:2] / p[2]
        if z <= 0: continue
        best, bd = None, 1e9
        for b in OBS[f][k]:
            c = np.array([b['x'] + b['w'] / 2, b['y'] + b['h'] / 2])
            d = np.linalg.norm(c - uv) / max(b['w'], b['h'], 40)
            if d < bd: best, bd = b, d
        if best is not None and bd < 1.5:
            boxes[(k, int(f))] = [best['x'], best['y'], best['x'] + best['w'], best['y'] + best['h']]
pickle.dump({'fs': fs, 'P': P, 'boxes': boxes}, open('stage1.pkl', 'wb'))
from collections import Counter
print(len(fs), 'frames tracked;', len(boxes), 'cam-frames with box', Counter(k for k, f in boxes))
