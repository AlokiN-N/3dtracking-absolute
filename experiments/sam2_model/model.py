import numpy as np, cv2
import os, json
# Выпуклые оболочки частей CargoTruck.fbx (см, оси UE: вперёд, вправо, вверх), начало координат — опорная точка:
# передний нижний край кабины по осевой. Получено из FBX скриптом fbxlib.py (ConvexHull по вершинам каждой части).
_J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'truck_hulls.json')))
FRONT = np.array(_J['front_cm'])
HULLS = {k: np.array(v) for k, v in _J['hulls_cm'].items()}
TRACTOR = ['Cab', 'Frame', 'FrontTires', 'TruckTiresBack1', 'TruckTiresBack2']
TRAILER = ['Cargo', 'CargoHoldThing', 'CargoTires01', 'CargoTires02']

def Rot(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    f = np.array([cp*cy, cp*sy, sp]); r = np.array([-sy, cy, 0.0]); u = np.cross(f, r) * -1
    u = np.array([-sp*cy, -sp*sy, cp])
    return np.stack([f, r, u], 1)   # колонки: вперёд, вправо, вверх (UE)

def world_parts(pose, s, parts=None):
    X, yaw, pitch = pose[:3], pose[3], pose[4]
    R = Rot(yaw, pitch)
    return {k: (R @ (v * s).T).T + X for k, v in HULLS.items() if parts is None or k in parts}

def render(Pm, pose, s, shape, off=(0, 0), scale=1.0, parts=None):
    """силуэт модели в изображении (возможно в обрезке off и уменьшении scale)"""
    m = np.zeros(shape, np.uint8)
    for k, w in world_parts(pose, s, parts).items():
        p = Pm @ np.hstack([w, np.ones((len(w), 1))]).T
        if (p[2] <= 1).any(): continue
        uv = (p[:2] / p[2]).T
        uv = (uv - np.array(off)) * scale
        h = cv2.convexHull(uv.astype(np.float32))
        cv2.fillConvexPoly(m, np.round(h).astype(np.int32), 1)
    return m
