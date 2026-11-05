# Этап 2: маски SAM 2.1 (hiera-small) по подсказке-рамке, с дорасширением рамки, если маска упирается в её край
import pickle, time, os, numpy as np, cv2
from common import VID
from sam2seg import SAM2, crop_box
S = pickle.load(open('stage1.pkl', 'rb')); B = S['boxes']
STRIDE = int(os.environ.get('STRIDE', '1'))   # маски на каждом STRIDE-м кадре (качество/время)
B = {kf: b for kf, b in B.items() if kf[1] % STRIDE == 0}
order = sorted(B, key=lambda kf: ({0: 0, 2: 1}.get(kf[1] % 4, 2), kf[1], kf[0]))
out = pickle.load(open('masks.pkl', 'rb')) if os.path.exists('masks.pkl') else {}
sam = SAM2(); caps = [cv2.VideoCapture(v) for v in VID]; t0 = time.time(); n = 0
def grab(k, f):
    caps[k].set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = caps[k].read(); return im
for kf in order:
    if kf in out: continue
    k, f = kf; b = np.array(B[kf], float); im = grab(k, f)
    if im is None: continue
    a, c, s = crop_box(b, k=2.2, mn=320); cr = im[c:c + s, a:a + s]; E = sam.encode(cr)
    bb = np.clip(b - [a, c, a, c], 0, s - 1)
    def touches(m, bb):
        ys, xs = np.nonzero(m)
        if not len(xs): return None
        return dict(l=xs.min() <= bb[0] + 2 and a + bb[0] > 3, r=xs.max() >= bb[2] - 3 and a + bb[2] < 1916,
                    u=ys.min() <= bb[1] + 2 and c + bb[1] > 3, d=ys.max() >= bb[3] - 3 and c + bb[3] < 1076)
    def leaks(m):   # маска растеклась по фону: касается краёв обрезки, которые не являются краем кадра
        e = [m[:, 0].any() and a > 0, m[:, -1].any() and a + s < 1920, m[0].any() and c > 0, m[-1].any() and c + s < 1080]
        return sum(e) >= 1
    m, sc = sam.decode(E, [[bb[0], bb[1]], [bb[2], bb[3]]], [2, 3]); grown = False
    t = touches(m, bb)
    if t and any(t.values()):
        w, h = bb[2] - bb[0], bb[3] - bb[1]; nb = bb + [-0.6 * w * t['l'], -0.3 * h * t['u'], 0.6 * w * t['r'], 0.3 * h * t['d']]
        nb = np.clip(nb, 0, s - 1); m1, sc1 = sam.decode(E, [[nb[0], nb[1]], [nb[2], nb[3]]], [2, 3])
        if sc1 >= sc - 0.03 and m1.sum() <= 2.5 * max(m.sum(), 1) and not leaks(m1): m, sc, bb, grown = m1, sc1, nb, True
    out[kf] = dict(off=(a, c), packed=np.packbits(m), shape=m.shape, score=sc, box=bb + [a, c, a, c], grown=grown, leak=leaks(m))
    n += 1
    if n % 25 == 0:
        pickle.dump(out, open('masks.tmp', 'wb')); os.replace('masks.tmp', 'masks.pkl')
        print(n, len(out), '/', len(B), '%.2f s/it' % ((time.time() - t0) / n), flush=True)
pickle.dump(out, open('masks.tmp', 'wb')); os.replace('masks.tmp', 'masks.pkl'); print('DONE', len(out))
