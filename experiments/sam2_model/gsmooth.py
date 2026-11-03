# Глобальное сглаживание всей траектории (эквивалент RTS-сглаживателя):
#   min  sum_f (X_f - x_f)^T Info_f (X_f - x_f)  +  sum_t |X_{t+1} - 2X_t + X_{t-1}|^2 / s_acc^2
# Info_f — анизотропная точность из геометрии: по каждой камере поперёк луча точно (пиксели),
# вдоль луча (глубина по размеру силуэта) — грубо. Выбросы гасятся IRLS (Коши).
# s_acc из физики: ускорение тяжёлого грузовика (вкл. боковое в повороте) <= ~0.5 g = 5 м/с^2
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from common import cams
from fit import Obs, wrap
A_MAX = float(__import__("os").environ.get("A_MAX", 500.0)); DT = 1 / 30.; S_ACC = A_MAX * DT * DT          # см/кадр^2
def info_for(f, r, M):
    I = np.zeros((3, 3)); X = r['x'][:3]
    for k in r['cams']:
        o = Obs(k, f, M[(k, f)]); C, Rm = cams[k].pose(f); d = np.linalg.norm(X - C)
        size = np.sqrt(o.area) / o.r                                   # размер силуэта, пикс полного кадра
        sl = 2.0 * d / cams[k].fx + 70.0                               # поперёк луча: 2 пикс + 0.7 м (неточность ориентации)
        sd = d * 4.0 / max(size, 5) + 70.0                             # вдоль луча: глубина по размеру силуэта (каждый край маски ±2 пикс -> размер ±4)
        ray = (X - C) / d; P_perp = np.eye(3) - np.outer(ray, ray)
        I += P_perp / sl ** 2 + np.outer(ray, ray) / sd ** 2
    return I
def gsmooth(F, M, N=1001, min_iou=0.5, iters=5, cauchy=3.0):
    fs = [f for f in sorted(F) if np.mean(F[f]['ious']) > min_iou and f < N]
    Z = {f: F[f]['x'][:3] for f in fs}; INF = {f: info_for(f, F[f], M) for f in fs}
    D = sp.diags([np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)], [0, 1, 2], shape=(N - 2, N)); D3 = sp.kron(D, sp.eye(3))
    Hs = (D3.T @ D3) / S_ACC ** 2; w = {f: 1.0 for f in fs}
    for it in range(iters):
        rows, cols, vals = [], [], []; b = np.zeros(3 * N)
        for f in fs:
            I = INF[f] * w[f]
            for i in range(3):
                for j in range(3): rows.append(3 * f + i); cols.append(3 * f + j); vals.append(I[i, j])
            b[3 * f:3 * f + 3] += I @ Z[f]
        H = Hs + sp.csr_matrix((vals, (rows, cols)), shape=(3 * N, 3 * N))
        X = spl.spsolve(H.tocsc(), b).reshape(N, 3)
        for f in fs:
            e = X[f] - Z[f]; m = np.sqrt(e @ INF[f] @ e); w[f] = 1 / (1 + (m / cauchy) ** 2)
    return X, w
if __name__ == '__main__':
    import pickle, sys
    from common import gt, load_masks
    F = pickle.load(open(sys.argv[1], 'rb')); M = load_masks()
    if len(sys.argv) > 3:   # кадры с одной камерой: позиция по относительному изменению размера (monofit)
        for p in sys.argv[3].split(','):
            for f, m in pickle.load(open(p, 'rb')).items():
                if f in F and m['iou'] > 0.6: F[f] = dict(F[f]); F[f]['x'] = np.r_[m['X'], m['yaw'], F[f]['x'][4]]; F[f]['ious'] = [m['iou']]
    X, w = gsmooth(F, M)
    np.save(sys.argv[2], X)
    if gt is None: sys.exit(0)
    G = gt[['Loc_X', 'Loc_Y', 'Loc_Z']].values[:1001]
    e = np.linalg.norm(X - G, axis=1) / 100
    print('RMSE %.2f MAE %.2f median %.2f p90 %.2f' % (np.sqrt((e ** 2).mean()), e.mean(), np.median(e), np.percentile(e, 90)))
    for a in range(0, 1001, 50): print(a, e[a:a + 50].mean().round(2), e[a:a + 50].max().round(1))
    np.save(sys.argv[2], X)
