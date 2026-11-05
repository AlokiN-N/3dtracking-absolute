# Подгонка 3D-модели грузовика (CargoTruck.fbx) под маски SAM 2 во всех камерах сразу
import numpy as np, cv2
from scipy.optimize import minimize
from common import cams
from model import render, TRACTOR, TRAILER
RES = 256
LAM = 0.5   # сторона обрезки при рендере, пикс
_P = {}
def Pm(k, f):
    if (k, f) not in _P: _P[(k, f)] = cams[k].P(f)
    return _P[(k, f)]
def unpack(d):
    H, W = d['shape']; return np.unpackbits(d['packed'])[:H * W].reshape(H, W).astype(bool)
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
class Obs:
    """маска одной камеры в одном кадре, уменьшенная до RES"""
    def __init__(s, k, f, d):
        s.k, s.f = k, f; m = unpack(d); s.off = d['off']; s.r = RES / m.shape[0]
        s.m = cv2.resize(m.astype(np.uint8), (RES, RES), interpolation=cv2.INTER_AREA) > 0
        s.area = s.m.sum(); s.score = d['score']
        s.Dm = cv2.distanceTransform((~s.m).astype(np.uint8), cv2.DIST_L2, 3) / RES
        s.norm = max(np.sqrt(s.area) / RES, 0.05)
        ys = np.nonzero(s.m.any(1))[0]; s.ybot = ys.max() if len(ys) else RES
    def render(s, pose, sc, parts=None): return render(Pm(s.k, s.f), pose, sc, (RES, RES), s.off, s.r, parts)
    def parts(s, pose, sc):
        """силуэты двух частей: тягач с кабиной и кузов-прицеп"""
        a = s.render(pose, sc, TRACTOR) > 0; b = s.render(pose, sc, TRAILER) > 0
        return a, b
    def _iou(s, r):
        # устойчивое к перекрытию сходство: маска должна лежать внутри силуэта модели (штраф 1),
        # а часть модели вне маски может быть закрыта деревом/рельефом/краем (штраф LAM < 1)
        i = (r & s.m).sum(); d = s.area + LAM * (r.sum() - i)
        return i / d if d else 0.0
    def iou(s, pose, sc, which=False):
        # маска SAM может покрывать весь грузовик или только одну часть (часто только кузов):
        # берём лучшую интерпретацию, части — с небольшим штрафом, чтобы при полной маске выигрывал весь грузовик
        a, b = s.parts(pose, sc)
        u = a | b; uc = u.copy(); uc[s.ybot + 2:] = False      # низ скрыт рельефом/кустами: видна только верхняя часть
        c = [(s._iou(u), 'all'), (s._iou(b) - 0.08, 'body'), (s._iou(a) - 0.08, 'cab'), (s._iou(uc) - 0.12, 'top')]
        best = max(c)
        return best if which else best[0]
    def chamfer(s, pose, sc):
        a, b = s.parts(pose, sc); out = []
        for r, pen in ((a | b, 0.0), (b, 0.15), (a, 0.15)):
            if r.sum() == 0: out.append(3.0); continue
            r = r.astype(np.uint8); Dr = cv2.distanceTransform(1 - r, cv2.DIST_L2, 3) / RES
            out.append(min((s.Dm[r > 0].mean() + Dr[s.m].mean()) / s.norm, 3.0) + pen)
        return min(out)
def full(p, base, free):
    x = base.copy(); x[free] = p; return x
def cost(p, base, free, obs, sc, soft, yaw0):
    x = full(p, base, free)
    c = sum(0.5 * o.chamfer(x, sc) for o in obs) if soft else sum(1 - o.iou(x, sc) for o in obs)
    if yaw0 is not None: c += 4.0 * max(0.0, abs(wrap(x[3] - yaw0)) - 0.5) ** 2   # нос машины смотрит по ходу движения
    return c
STEPS = np.array([250., 250., 120., 0.25, 0.08])
def fit_frame(obs, x0, sc, free=(0, 1, 2, 3, 4), yaw0=None, starts=((0, 0), (0.35, 0), (-0.35, 0), (0, 800), (0, -800))):
    free = list(free); best = None
    for dyaw, dsh in (starts if 3 in free else ((0, 0), (0, 800), (0, -800))):
        b = np.array(x0, float); b[3] += dyaw; b[:2] += dsh * np.array([np.cos(b[3]), np.sin(b[3])]); p = b[free]
        r = minimize(cost, p, args=(b, free, obs, sc, True, yaw0), method='Nelder-Mead',
                     options=dict(initial_simplex=np.array([p] + [p + np.eye(len(p))[i] * STEPS[free][i] for i in range(len(p))]),
                                  maxfev=150 * len(p), xatol=3, fatol=1e-4))
        r2 = minimize(cost, r.x, args=(b, free, obs, sc, False, yaw0), method='Nelder-Mead',
                      options=dict(initial_simplex=np.array([r.x] + [r.x + np.eye(len(p))[i] * STEPS[free][i] * 0.25 for i in range(len(p))]),
                                   maxfev=100 * len(p), xatol=2, fatol=1e-4))
        x = full(r2.x, b, free); x[3] = wrap(x[3])
        if best is None or r2.fun < best[1]: best = (x, r2.fun)
    return best
