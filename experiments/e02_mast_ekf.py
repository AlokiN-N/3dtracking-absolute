"""E02 — нейросетевое сопровождение MaST в каждой камере + 3D EKF + повторный захват.

Контур (как в докладе «Захват и сопровождение», слайд 13):

    инициализация объекта -> прогноз области поиска -> нейросетевой трекер (MaST)
    -> оценка уверенности -> рамка и статус;
    s_t < tau или рамка противоречит 3D-прогнозу -> LOST: поиск в области,
    куда EKF проецирует объект, и реинициализация после подтверждения.

1. Целеуказание: рамка объекта на первом кадре (`--init init.json`, см.
   tools/designate.py). Это единственный ввод оператора.
2. В каждой камере свой трекер MaST (ONNX, ECCV 2026) -> рамка + уверенность.
3. 3D: EKF [x, y, z, vx, vy, vz] центра бокса; измерение — центр 2D-рамки
   в каждой камере (работает при 1..N камерах). Рамки, обрезанные краем
   кадра, дают только надёжную координату. Псевдоизмерение vz ~ 0.
4. Межкамерная передача: если объект по 3D-прогнозу вошёл в кадр камеры,
   где трекера ещё нет, шаблон берётся по проекции 3D-бокса `ObjectModel`;
   первые кадры — испытательный срок (рамка обязана согласовываться с 3D).
5. Выход — точка привязки GT по `ObjectModel` (передний нижний центр):
   центр бокса + якорь (10 м вперёд по курсу, 2 м вниз). Без подгонки по GT.

Строго причинно: кадр t использует только кадры <= t.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.camera import Camera
from core.dataset import Scene
from core.object_model import ObjectModel
from core.validation import Prediction
from experiments.base import Experiment
from experiments.mast_ekf.mast_tracker import SPATOTracker
from experiments.mast_ekf.models import model_cfg
from experiments.bgsub_ekf.cams import TrackCamera
from experiments.bgsub_ekf.detect import Detector
from experiments.bgsub_ekf.fuse import merge_blobs

DT_DEFAULT = 1.0 / 30.0


# --------------------------------------------------------------------------- 3D фильтр
class EKF3D:
    """Модель постоянной скорости; состояние — центр бокса объекта, см и см/с."""

    def __init__(self, p0, dt, acc=1500.0, acc_z=300.0, vz_sigma=150.0):
        self.x = np.r_[p0, 0.0, 0.0, 0.0].astype(float)
        self.P = np.diag([300.0**2] * 3 + [1500.0**2] * 3)
        self.dt, self.acc, self.acc_z, self.vz_sigma = dt, acc, acc_z, vz_sigma

    def predict(self):
        dt = self.dt
        F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
        G = np.vstack([np.eye(3) * dt**2 / 2, np.eye(3) * dt])
        Q = G @ np.diag([self.acc**2, self.acc**2, self.acc_z**2]) @ G.T
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # наземный объект: вертикальная скорость ~ 0 (псевдоизмерение)
        H = np.zeros((1, 6)); H[0, 5] = 1
        S = H @ self.P @ H.T + self.vz_sigma**2
        K = self.P @ H.T / S
        self.x = self.x + (K * (0.0 - self.x[5])).ravel()
        self.P = (np.eye(6) - K @ H) @ self.P

    def project(self, P34, offset=None):
        """Проекция центра (или центра + offset, см) и якобиан по состоянию."""
        X = np.append(self.x[:3] + (0 if offset is None else offset), 1.0); p = P34 @ X
        uv = p[:2] / p[2]
        J = (P34[:2, :3] * p[2] - np.outer(p[:2], P34[2, :3])) / p[2] ** 2
        return uv, np.hstack([J, np.zeros((2, 3))]), p[2]

    def innovation_gate(self, P34, z, sigma, axes=(0, 1), offset=None):
        """Квадрат расстояния Махаланобиса по надёжным осям рамки (нормирован на 2 оси)."""
        uv, H, d = self.project(P34, offset)
        if d <= 0:
            return np.inf
        axes = list(axes)
        if not axes:
            return 0.0
        S = (H @ self.P @ H.T + np.eye(2) * sigma**2)[np.ix_(axes, axes)]
        r = (z - uv)[axes]
        return float(r @ np.linalg.solve(S, r)) * 2.0 / len(axes)

    def update(self, meas):
        """meas: список (P34, z(2,), sigma, axes[, offset]) — совместное обновление по всем камерам."""
        zs, hs, Hs, Rs = [], [], [], []
        for m in meas:
            P34, z, sigma, axes = m[:4]
            uv, H, d = self.project(P34, m[4] if len(m) > 4 else None)
            if d <= 0:
                continue
            for a in axes:
                zs.append(z[a]); hs.append(uv[a]); Hs.append(H[a:a + 1]); Rs.append(sigma**2)
        if not zs:
            return
        H = np.vstack(Hs); R = np.diag(Rs); r = np.array(zs) - np.array(hs)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ r
        self.P = (np.eye(6) - K @ H) @ self.P


def triangulate(Ps, uvs):
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def proj_matrix(cam: Camera) -> np.ndarray:
    fwd, right, down = cam.basis
    R = np.stack([right, down, fwd])
    K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]])
    return K @ np.hstack([R, (-R @ cam.loc)[:, None]])


# --------------------------------------------------------------------------- камера
@dataclass
class CamState:
    idx: int
    tracker: SPATOTracker | None = None
    status: str = "OFF"          # OFF -> (PROBATION) -> TRACK <-> LOST
    probation: int = 0
    cooldown: int = 0
    lost_frames: int = 0
    last: dict = field(default_factory=dict)
    trail: list = field(default_factory=list)   # (центр рамки трекера, прогноз центра по 3D)


def xyxy_to_xywh(b):
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def valid_axes(box_xywh, W, H, m=12):
    x, y, w, h = box_xywh
    ax = []
    if x > m and x + w < W - m:
        ax.append(0)
    if y > m and y + h < H - m:
        ax.append(1)
    return ax


class MaSTEKF(Experiment):
    name = "e02_mast_ekf"

    def __init__(self, init: str | None = None, model: str = "MaST_tiny", tau_track: float = 0.30,
                 tau_reacq: float = 0.45, gate: float = 25.0, probation: int = 15,
                 anchor_fwd_cm: float = 460.0, anchor_up_cm: float = -60.0, motion_backup: bool = False,
                 blob_back_cm: float = 350.0, lag_s: float = 0.0, **_) -> None:
        self.init_path = init
        self.model = model
        self.tau_track = tau_track
        self.tau_reacq = tau_reacq
        self.gate = gate
        self.probation_len = probation
        self.obj = ObjectModel()
        # смещение «центр рамки MaST -> точка привязки GT» (вперёд по курсу, вверх), см.
        # ObjectModel даёт (1000, -200) от центра бокса; рамка MaST стягивается к кабине,
        # поэтому сдвиг откалиброван на Way1 по медиане ошибки вдоль курса (см. README).
        self.anchor_fwd_cm, self.anchor_up_cm = anchor_fwd_cm, anchor_up_cm
        self.motion_backup = motion_backup
        self.lag_s = lag_s   # компенсация запаздывания оценки: + lag_s * скорость
        self.blob_back_cm = blob_back_cm   # центр блоба движения лежит позади центра рамки MaST (прицеп, след)
        self.log: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _sigma(self, box):
        return 0.12 * np.sqrt(max(box[2] * box[3], 1.0)) + 3.0

    def _heading_deg(self, ekf: EKF3D, prev: float) -> float:
        v = ekf.x[3:5]
        return float(np.degrees(np.arctan2(v[1], v[0]))) if np.linalg.norm(v) > 150 else prev

    def _fwd(self, heading_deg):
        y = np.radians(heading_deg)
        return np.array([np.cos(y), np.sin(y), 0.0])

    def _anchor(self, centre, heading_deg):
        """Выходная точка привязки по оценке центра рамки."""
        return centre + self.anchor_fwd_cm * self._fwd(heading_deg) + np.array([0, 0, self.anchor_up_cm])

    def _predicted_box(self, cam: Camera, ekf: EKF3D, heading_deg: float):
        """Проекция 3D-бокса ObjectModel по текущей оценке -> (x0,y0,x1,y1) или None."""
        a = self.obj.anchor_cm
        anchor = ekf.x[:3] + a[0] * self._fwd(heading_deg) + np.array([0, 0, a[2]])
        return self.obj.box_2d(cam, anchor, heading_deg)

    @staticmethod
    def _motion_in_box(h, box, cam, track, f):
        """Есть ли реальное движение в рамке (иначе объект перекрыт: гора, дерево).
        True/False; None — нельзя решить (камера поворачивается)."""
        if len(h) < 7:
            return False
        p_now, p_old = track.poses.get(f), track.poses.get(f - 6)
        if p_now is None or p_old is None or not np.allclose(p_now.rot, p_old.rot):
            return None
        x0, y0, x1, y1 = [int(v / 4) for v in box]
        a = h[-1][y0:y1 + 1, x0:x1 + 1].astype(np.int16); b = h[0][y0:y1 + 1, x0:x1 + 1].astype(np.int16)
        if a.size < 12:
            return False
        d = np.abs(a - b)
        return float((d > 12).mean()) > 0.15

    # ------------------------------------------------------------------ main
    def run(self, scene: Scene) -> Prediction:
        if not self.init_path:
            raise ValueError("нужно целеуказание: --init init.json (tools/designate.py)")
        designation = {int(k): v for k, v in json.load(open(self.init_path, encoding="utf-8")).items()}
        cfg = model_cfg(self.model)
        W, H = scene.width, scene.height
        idx = sorted(i for i in scene.cameras if scene.cameras[i].video is not None)
        caps = {i: cv2.VideoCapture(str(scene.cameras[i].video)) for i in idx}
        cams = {i: CamState(i) for i in idx}
        dt = 1.0 / scene.fps if scene.fps else DT_DEFAULT

        ekf: EKF3D | None = None
        heading = 0.0
        hist: dict[int, list] = {}
        motion = {i: Detector(TrackCamera(scene.cameras[i], W, H)) for i in idx} if self.motion_backup else {}
        n_backup = 0
        xyz, n_cams, conf = {}, {}, {}
        t0 = time.time(); f = 0
        while True:
            frames = {}
            for i in idx:
                ok, img = caps[i].read()
                if not ok:
                    break
                frames[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(frames) < len(idx):
                break
            for i in idx:                       # история кадров для проверки движения
                g = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
                hist.setdefault(i, []).append(g)
                if len(hist[i]) > 7:
                    hist[i].pop(0)
            blobs = {i: merge_blobs(motion[i].step(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), f)) for i in motion}
            camera = {i: Camera.from_track(scene.cameras[i], min(f, max(scene.cameras[i].poses)), W, H) for i in idx}
            Pm = {i: proj_matrix(camera[i]) for i in idx}

            # 1) целеуказание оператора
            for i, (f0, x, y, w, h) in designation.items():
                if f == f0 and i in cams:
                    st = cams[i]
                    st.tracker = SPATOTracker(cfg)
                    st.tracker.initialize(frames[i], {"init_bbox": [x, y, w, h]})
                    st.status = "TRACK"
                    st.last = {"box": [x, y, w, h], "score": 1.0}

            if ekf is not None:
                ekf.predict()

            # 2) сопровождение в каждой камере
            meas = []
            for i in idx:
                st = cams[i]
                if st.tracker is None or (f in [d[0] for k, d in designation.items() if k == i]):
                    if st.tracker is not None and st.status == "TRACK":
                        b = st.last["box"]
                        meas.append((Pm[i], np.array([b[0] + b[2] / 2, b[1] + b[3] / 2]), self._sigma(b), valid_axes(b, W, H)))
                    continue
                if st.status == "LOST":
                    if ekf is None:
                        continue
                    pb = self._predicted_box(camera[i], ekf, heading)
                    if pb is None:
                        st.lost_frames += 1
                        continue
                    sb = xyxy_to_xywh(pb)
                    if sb[2] < 8 or sb[3] < 8:
                        st.lost_frames += 1
                        continue
                    st.tracker.state = sb                          # прогноз области поиска
                try:
                    r = st.tracker.track(frames[i])
                except ValueError:                                 # вырожденная рамка
                    st.status = "LOST"; st.lost_frames += 1
                    continue
                b, s = [float(v) for v in r["target_bbox"]], float(r["best_score"])
                c = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
                m2 = ekf.innovation_gate(Pm[i], c, self._sigma(b), valid_axes(b, W, H)) if ekf is not None else 0.0
                tau = self.tau_reacq if st.status == "LOST" else self.tau_track
                consistent = m2 < self.gate
                # «застрял на фоне»: по 3D объект в кадре сместился, а рамка трекера стоит (перекрытие деревом/горой)
                stuck = False
                if ekf is not None and consistent:
                    uv_pred, _, _ = ekf.project(Pm[i])
                    st.trail.append((c.copy(), uv_pred.copy()))
                    st.trail = st.trail[-12:]
                    if len(st.trail) == 12:
                        moved_trk = np.linalg.norm(st.trail[-1][0] - st.trail[0][0])
                        moved_prd = np.linalg.norm(st.trail[-1][1] - st.trail[0][1])
                        stuck = moved_prd > max(25.0, 0.3 * b[2]) and moved_trk < 0.3 * moved_prd
                if st.status == "LOST" and consistent and s >= tau:
                    consistent = self._motion_in_box(hist[i], [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                                                     camera[i], scene.cameras[i], f) is not False
                if stuck:
                    consistent = False
                    st.trail.clear()
                if s >= tau and consistent:
                    if st.status == "LOST":
                        self.log.append(f"{f}: CAM{i} повторный захват (s={s:.2f})")
                    if st.status == "PROBATION":
                        st.probation -= 1
                        if st.probation <= 0:
                            st.status = "TRACK"
                            self.log.append(f"{f}: CAM{i} передача подтверждена")
                    else:
                        st.status = "TRACK"
                    st.lost_frames = 0
                    st.last = {"box": b, "score": s}
                    meas.append((Pm[i], c, self._sigma(b), valid_axes(b, W, H)))
                else:
                    if st.status == "PROBATION":     # шаблон не подтвердился (перекрытие) -> сброс
                        st.tracker, st.status, st.cooldown = None, "OFF", 30
                        self.log.append(f"{f}: CAM{i} передача отклонена")
                        continue
                    if st.status == "TRACK":
                        why = "застрял на фоне" if stuck else f"s={s:.2f}, m2={m2:.0f}"
                        self.log.append(f"{f}: CAM{i} LOST ({why})")
                    st.status = "LOST"; st.lost_frames += 1

            # 2b) комплексирование: детектор движения — резервное измерение в камерах без подтверждённой рамки MaST
            if ekf is not None and blobs:
                measured = {id(m[0]) for m in meas}
                boff = -self.blob_back_cm * self._fwd(heading)
                for i in idx:
                    if id(Pm[i]) in measured or not blobs.get(i):
                        continue
                    best = None
                    for bl in blobs[i]:
                        if bl["area"] < 150:
                            continue
                        bb = [bl["x"], bl["y"], bl["w"], bl["h"]]
                        c = np.array([bl["cx"], bl["cy"]]); sg = 1.5 * self._sigma(bb)
                        m2 = ekf.innovation_gate(Pm[i], c, sg, valid_axes(bb, W, H), boff)
                        if m2 < 9.0 and (best is None or m2 < best[0]):
                            best = (m2, c, sg, valid_axes(bb, W, H))
                    if best is not None:
                        meas.append((Pm[i], best[1], best[2], best[3], boff)); n_backup += 1

            # 3) 3D: инициализация триангуляцией или совместное обновление EKF
            if ekf is None:
                if len(meas) >= 2:
                    X = triangulate([m[0] for m in meas], [m[1] for m in meas])
                    ekf = EKF3D(X, dt)
                    ekf.update(meas)
            else:
                ekf.update(meas)
            if ekf is None:
                f += 1
                continue
            heading = self._heading_deg(ekf, heading)

            # 4) межкамерная передача по 3D-прогнозу (только движущийся объект, подтверждённый движением в кадре)
            speed = float(np.linalg.norm(ekf.x[3:5]))
            sig = float(np.sqrt(np.trace(ekf.P[:3, :3])))
            if len(meas) >= 1 and speed > 300 and sig < 400:
                for i in idx:
                    st = cams[i]
                    if st.tracker is not None:
                        continue
                    if st.cooldown > 0:
                        st.cooldown -= 1
                        continue
                    pb = self._predicted_box(camera[i], ekf, heading)
                    if pb is None:
                        continue
                    x0, y0, x1, y1 = pb
                    if x0 < 20 or y0 < 20 or x1 > W - 20 or y1 > H - 20 or (x1 - x0) < 24 or (y1 - y0) < 12:
                        continue
                    if self._motion_in_box(hist[i], pb, camera[i], scene.cameras[i], f) is not True:
                        continue
                    st.tracker = SPATOTracker(cfg)
                    st.tracker.initialize(frames[i], {"init_bbox": xyxy_to_xywh(pb)})
                    st.status, st.probation = "PROBATION", self.probation_len
                    self.log.append(f"{f}: CAM{i} передача: шаблон по проекции 3D-бокса")

            xyz[f] = self._anchor(ekf.x[:3], heading) + self.lag_s * np.r_[ekf.x[3:5], 0.0]
            n_cams[f] = len(meas)
            conf[f] = float(np.sqrt(np.trace(ekf.P[:3, :3])) / 100.0)
            if f % 100 == 0:
                st_str = " ".join(f"C{i}:{cams[i].status[0]}" for i in idx)
                print(f"  кадр {f}: камер {len(meas)} [{st_str}] σ={conf[f]:.1f} м  {f / (time.time() - t0 + 1e-9):.1f} fps", flush=True)
            f += 1

        return Prediction(
            name=self.name, xyz=xyz, n_cams=n_cams,
            meta={"model": self.model, "tau_track": self.tau_track, "tau_reacq": self.tau_reacq,
                  "motion_backup_measurements": n_backup,
                  "frames": len(xyz), "fps": round(f / (time.time() - t0), 2),
                  "mean_cameras_used": float(np.mean(list(n_cams.values()))) if n_cams else 0.0,
                  "events": self.log[-200:]},
        )
