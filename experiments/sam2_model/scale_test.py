# самокалибровка масштаба модели по кадрам с >=3 камерами (без истины): минимум суммарной (1 - IoU)
import pickle, sys, numpy as np
from fit import Obs, fit_frame
from common import load_masks
M = load_masks(); F = pickle.load(open('fits2.pkl', 'rb'))
fr = [f for f in sorted(F) if len(F[f]['cams']) >= 3 and min(F[f]['ious']) > 0.6][::2][:16]
for sc in map(float, sys.argv[1].split(',')):
    tot = 0
    for f in fr:
        obs = [Obs(k, f, M[(k, f)]) for k in F[f]['cams']]
        x, c = fit_frame(obs, F[f]['x'], sc, yaw0=F[f]['x'][3], starts=((0, 0),)); tot += c / len(obs)
    print('scale', sc, 'mean cost %.4f' % (tot / len(fr)), 'frames', len(fr), flush=True)
