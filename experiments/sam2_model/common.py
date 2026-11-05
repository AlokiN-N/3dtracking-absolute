import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cams_csv import load_cams, load_track
# папка с данными: object_transform_*.csv и видео Way1.FinalImage.CineCameraActor{1..5}.mp4
D = os.environ.get('PELENG_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))
cams = load_cams(D)
gt = load_track(f'{D}/object_transform_SM_AA_track.csv') if os.path.exists(f'{D}/object_transform_SM_AA_track.csv') else None  # истина нужна только для оценки
import glob
VID = [sorted(glob.glob(f'{D}/*CineCameraActor{i}.mp4'))[0] for i in range(1, 6)]
def read_frames(i, frames):
    c = cv2.VideoCapture(VID[i]); out = {}; f = 0; fs = set(frames)
    while f <= max(frames):
        ok, im = c.read()
        if not ok: break
        if f in fs: out[f] = im
        f += 1
    return out
import os, pickle
def load_masks():
    M = {}
    for p in os.environ.get('MASKS', 'masks.pkl').split(','):
        if not os.path.exists(p): continue
        for kf, d in pickle.load(open(p, 'rb')).items():
            if d.get('proj') and (d['piou'] < 0.5 or d['score'] < 0.8): continue    # маска по проекции не согласуется с моделью — пропускаем
            M[kf] = d
    return M
