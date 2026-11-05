# Одна камера: глубину считаем по ОТНОСИТЕЛЬНОМУ изменению размера силуэта.
# q_k(f) = sqrt(площадь маски / площадь силуэта модели в эталонной позе) — во сколько раз камера k
# «видит» грузовик крупнее/мельче модели (ореол маски SAM, размытие и т.п.). На кадрах, где грузовик
# видят >=2 камеры, позиция известна точно -> q_k калибруется. В кадрах с одной камерой q_k берём
# интерполяцией от соседних калиброванных кадров (до ухода второй камеры и после её появления),
# и поправляем глубину: d = d_fit * q_ref. Т.е. пройденное расстояние = изменение размера модели.
import numpy as np, pickle
from fit import Obs
from common import cams
SC = 3.1
def q_value(o, x):
    (val, which) = o.iou(x, SC, True)
    if which != 'all' or val < 0.75: return None
    r = o.render(x, SC) > 0
    return np.sqrt(o.area / max(r.sum(), 1))
def correct(F, M, T):
    ref = {k: [] for k in range(5)}
    for f, r in F.items():
        if len(r['cams']) < 2 or min(r['ious']) < 0.7: continue
        x = np.r_[T[f], r['x'][3:5]]
        for k in r['cams']:
            q = q_value(Obs(k, f, M[(k, f)]), x)
            if q is not None: ref[k].append((f, q))
    out = {}
    for f, r in F.items():
        if len(r['cams']) != 1: continue
        k = r['cams'][0]; R = np.array(ref[k])
        if len(R) < 3: continue
        # медиана по ближайшим калибровочным кадрам с каждой стороны, линейно по времени между ними
        before, after = R[R[:, 0] <= f], R[R[:, 0] >= f]
        qb = np.median(before[-6:, 1]) if len(before) else None; qa = np.median(after[:6, 1]) if len(after) else None
        if qb is not None and qa is not None:
            fb, fa = before[-1, 0], after[0, 0]; w = (f - fb) / max(fa - fb, 1); q = (1 - w) * qb + w * qa
        else: q = qb if qb is not None else qa
        C, _ = cams[k].pose(f); X = r['x'][:3]
        out[f] = dict(k=k, q=q, X=C + (X - C) * q)
    return out, ref
if __name__ == '__main__':
    import sys
    from common import load_masks, gt
    M = load_masks(); F = pickle.load(open(sys.argv[1], 'rb')); T = np.load(sys.argv[2])
    out, ref = correct(F, M, T)
    for k in range(5): print('cam', k + 1, 'calib frames', len(ref[k]), 'q median %.3f' % np.median([q for _, q in ref[k]]) if ref[k] else '')
    G = gt[['Loc_X', 'Loc_Y', 'Loc_Z']].values
    e0 = [np.linalg.norm(F[f]['x'][:3] - G[f]) / 100 for f in out]; e1 = [np.linalg.norm(out[f]['X'] - G[f]) / 100 for f in out]
    print('mono frames', len(out), 'err before median %.1f mean %.1f -> after median %.1f mean %.1f' % (np.median(e0), np.mean(e0), np.median(e1), np.mean(e1)))
    pickle.dump(out, open(sys.argv[3], 'wb'))
