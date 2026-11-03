"""Графики координат от времени (GT vs оценка), ошибка по кадрам и вид сверху.
python -m tools.plots out/<эксперимент>/errors.csv out/<эксперимент>/plots.png"""
import csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
f = np.array([int(r["frame"]) for r in rows]); t = f / 30.0
g = np.array([[float(r[k]) for k in ("gt_x", "gt_y", "gt_z")] for r in rows]) / 100
p = np.array([[float(r[k]) for k in ("pred_x", "pred_y", "pred_z")] for r in rows]) / 100
e = np.array([float(r["err_m"]) for r in rows])
fig = plt.figure(figsize=(15, 10))
for k, n in enumerate("XYZ"):
    a = fig.add_subplot(4, 2, 2 * k + 2); a.plot(t, g[:, k], "k", lw=2, label="истина"); a.plot(t, p[:, k], "r", lw=1, label="оценка")
    a.set_ylabel(f"{n}, м"); a.grid(alpha=.3)
    if k == 0: a.legend(loc="upper left")
a = fig.add_subplot(4, 2, 8); a.plot(t, e, "b"); a.set_ylabel("ошибка 3D, м"); a.set_xlabel("время, с"); a.grid(alpha=.3)
a = fig.add_subplot(1, 2, 1); a.plot(g[:, 0], g[:, 1], "k", lw=2, label="истина"); a.plot(p[:, 0], p[:, 1], "r", lw=1, label="оценка")
a.set_aspect("equal"); a.grid(alpha=.3); a.legend(); a.set_xlabel("X, м"); a.set_ylabel("Y, м")
a.set_title(f"Вид сверху. MAE {e.mean():.2f} м, RMSE {np.sqrt((e**2).mean()):.2f} м, медиана {np.median(e):.2f} м, p95 {np.percentile(e, 95):.2f} м")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110); print("->", sys.argv[2])
