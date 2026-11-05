# Этап 2b: SAM 2 по подсказке «проекция подогнанной модели» — для камер/кадров, где маски нет или она плохая
import pickle, sys, time, os, numpy as np, cv2
from common import VID, cams
from sam2seg import SAM2, crop_box
from model import render
from fit import unpack
xyz = np.load(sys.argv[1]); frames = range(*map(int, sys.argv[2].split(':'))); OUT = sys.argv[3]
M = pickle.load(open('masks.pkl', 'rb'))
def pose(t):
    a, b = max(t - 6, 0), min(t + 6, len(xyz) - 1); d = xyz[b] - xyz[a]
    return np.r_[xyz[t], np.arctan2(d[1], d[0]), np.arctan2(d[2], np.hypot(d[0], d[1]))]
out = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
sam = SAM2(); caps = [cv2.VideoCapture(v) for v in VID]; t0 = time.time(); n = 0
todo = []
for f in frames:
    for k in range(5):
        r = render(cams[k].P(f), pose(f), 3.0, (1080, 1920))
        if r.sum() < 150: continue                                      # не в кадре или слишком мал
        if (k, f) in M:
            d = M[(k, f)]; m = np.zeros((1080, 1920), bool); a, c = d['off']; mm = unpack(d); m[c:c + mm.shape[0], a:a + mm.shape[1]] = mm
            iou = (m & (r > 0)).sum() / max((m | (r > 0)).sum(), 1)
            if iou > 0.4 and not d.get('leak'): continue                 # маска есть и согласована
        todo.append((k, f, r))
todo.sort(key=lambda t: (t[1] % 4 != 0, t[1], t[0]))
print('todo', len(todo), sum(1 for t in todo if t[1] % 4 == 0), flush=True)
for k, f, r in todo:
    if (k, f) in out: continue
    ys, xs = np.nonzero(r); b = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    w, h = b[2] - b[0], b[3] - b[1]; b += [-0.05 * w, -0.05 * h, 0.05 * w, 0.05 * h]; b = np.clip(b, 0, [1919, 1079, 1919, 1079])
    caps[k].set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = caps[k].read()
    if not ok: continue
    a, c, s = crop_box(b, k=1.6, mn=256); cr = im[c:c + s, a:a + s]; E = sam.encode(cr)
    bb = np.clip(b - [a, c, a, c], 0, s - 1); m, sc = sam.decode(E, [[bb[0], bb[1]], [bb[2], bb[3]]], [2, 3])
    rp = r[c:c + s, a:a + s] > 0; piou = (m & rp).sum() / max((m | rp).sum(), 1)
    out[(k, f)] = dict(off=(a, c), packed=np.packbits(m), shape=m.shape, score=sc, box=b, grown=False, leak=False, proj=True, piou=piou)
    n += 1
    if n % 20 == 0:
        pickle.dump(out, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT); print(n, '%.2f s/it' % ((time.time() - t0) / n), flush=True)
pickle.dump(out, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT); print('DONE', len(out))
