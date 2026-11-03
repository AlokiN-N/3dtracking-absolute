# Одна камера: ориентацию не подбираем (она плохо определяется по одному силуэту и «размазывает» глубину),
# а берём из траектории: касательная + поправка курса, измеренная на ближайших кадрах с >=2 камерами.
# Подбираем только положение (x, y, z) -> глубина определяется размером силуэта, затем поправка q (monocorr).
import numpy as np, pickle, sys
from fit import Obs, fit_frame, wrap
from monocorr import q_value
from common import cams, load_masks
SC = 3.1
def tangent(T, f, h=6):
    a, b = max(f - h, 0), min(f + h, len(T) - 1); d = T[b] - T[a]
    return np.arctan2(d[1], d[0]), np.arctan2(d[2], np.hypot(d[0], d[1]))
def interp_side(R, f):
    R = np.array(R)
    if not len(R): return None
    bf, af = R[R[:, 0] <= f], R[R[:, 0] >= f]
    vb = np.median(bf[-6:, 1]) if len(bf) else None; va = np.median(af[:6, 1]) if len(af) else None
    if vb is not None and va is not None:
        w = (f - bf[-1, 0]) / max(af[0, 0] - bf[-1, 0], 1); return (1 - w) * vb + w * va
    return vb if vb is not None else va
if __name__ == '__main__':
    F = pickle.load(open(sys.argv[1], 'rb')); T = np.load(sys.argv[2]); M = load_masks(); frames = None
    if len(sys.argv) > 4: a, b = map(int, sys.argv[4].split(':')); frames = (a, b)
    dyaw, qref = [], {k: [] for k in range(5)}
    for f, r in sorted(F.items()):
        if len(r['cams']) < 2 or min(r['ious']) < 0.7: continue
        ty, _ = tangent(T, f); dyaw.append((f, wrap(r['x'][3] - ty)))
        x = np.r_[T[f], r['x'][3:5]]
        for k in r['cams']:
            q = q_value(Obs(k, f, M[(k, f)]), x)
            if q is not None: qref[k].append((f, q))
    out = {}
    for f, r in sorted(F.items()):
        if len(r['cams']) != 1 or (frames and not frames[0] <= f < frames[1]): continue
        k = r['cams'][0]; o = Obs(k, f, M[(k, f)])
        ty, tp = tangent(T, f); yaw = ty + (interp_side(dyaw, f) or 0.0)
        x0 = np.r_[T[f], yaw, tp]; x, c = fit_frame([o], x0, SC, free=(0, 1, 2))
        q = interp_side(qref[k], f) or 1.0
        C, _ = cams[k].pose(f); X = C + (x[:3] - C) * q
        out[f] = dict(k=k, q=q, X=X, iou=o.iou(x, SC), yaw=yaw)
        print(f, k + 1, 'iou %.2f q %.3f' % (out[f]['iou'], q), flush=True)
    pickle.dump(out, open(sys.argv[3], 'wb'))
