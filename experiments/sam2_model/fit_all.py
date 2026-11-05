# Этап 3: последовательная подгонка позы модели по кадрам (x, y, z, yaw, pitch), масштаб модели фиксирован
import pickle, sys, time, numpy as np
from fit import Obs, fit_frame, cost
SC = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
OUT = sys.argv[2] if len(sys.argv) > 2 else 'fits.pkl'
FR = sys.argv[3] if len(sys.argv) > 3 else None
from common import load_masks
M = load_masks(); S = pickle.load(open('stage1.pkl', 'rb'))
fs1, P1 = S['fs'], S['P']
def init_e03(f):
    i = min(np.searchsorted(fs1, f), len(fs1) - 1); a, b = max(i - 10, 0), min(i + 10, len(fs1) - 1)
    v = P1[b] - P1[a]; yaw = np.arctan2(v[1], v[0]); h = np.array([np.cos(yaw), np.sin(yaw), 0])
    return np.r_[P1[i] + 200 * h - [0, 0, 150], yaw, 0.0]
import os
if os.environ.get('INIT'):
    _X = np.load(os.environ['INIT'])
    def init_e03(f):
        a, b = max(f - 6, 0), min(f + 6, len(_X) - 1); d = _X[b] - _X[a]
        return np.r_[_X[f], np.arctan2(d[1], d[0]), np.arctan2(d[2], np.hypot(d[0], d[1]))]
def good(d): return not d.get('leak', False) and d['score'] > 0.6
frames = sorted({f for k, f in M})
if FR: a, b, st = map(int, FR.split(':')); frames = [f for f in frames if a <= f < b and f % st == 0]
fits = {}; hist = []; t0 = time.time()
for f in frames:
    obs = [Obs(k, f, M[(k, f)]) for k in range(5) if (k, f) in M and good(M[(k, f)])]
    obs = [o for o in obs if o.area >= 12]
    if not obs: continue
    cands = [init_e03(f)]
    if len(hist) >= 2 and f - hist[-1][0] <= 16:
        (f1, x1), (f2, x2) = hist[-2], hist[-1]; v = (x2 - x1) / max(f2 - f1, 1); v[3:] = 0
        cands.insert(0, x2 + v * (f - f2))
    elif hist and f - hist[-1][0] <= 16: cands.insert(0, hist[-1][1])
    best = None
    for c in cands:
        x, cst = fit_frame(obs, c, SC, yaw0=init_e03(f)[3])
        if best is None or cst < best[1]: best = (x, cst)
        if cst / len(obs) < 0.3: break
    x, cst = best
    ious = [o.iou(x, SC) for o in obs]
    # отбрасываем камеру-выброс (ложная маска), если остальные согласованы
    if len(obs) >= 2 and min(ious) < 0.3 and np.median(ious) > 0.5:
        obs = [o for o, i in zip(obs, ious) if i >= 0.3]; x, cst = fit_frame(obs, x, SC, yaw0=init_e03(f)[3]); ious = [o.iou(x, SC) for o in obs]
    fits[f] = dict(x=x, cost=cst, ious=ious, cams=[o.k for o in obs])
    if np.mean(ious) > 0.55: hist.append((f, x))
    print(f, [o.k + 1 for o in obs], 'iou', np.round(ious, 2), 'x', (x[:3] / 100).round(1), 'yaw %.0f pitch %.1f' % tuple(np.degrees(x[3:5])), flush=True)
pickle.dump(fits, open(OUT, 'wb')); print('done', len(fits), '%.0fs' % (time.time() - t0))
