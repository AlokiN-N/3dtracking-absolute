import numpy as np, pandas as pd, re
SENSOR_W, SENSOR_H, W, H = 23.76, 13.365, 1920, 1080

def ue_axes(roll, pitch, yaw):
    R, P, Y = np.radians([roll, pitch, yaw])
    SR, CR, SP, CP, SY, CY = np.sin(R), np.cos(R), np.sin(P), np.cos(P), np.sin(Y), np.cos(Y)
    fwd = np.array([CP*CY, CP*SY, SP])
    right = np.array([SR*SP*CY - CR*SY, SR*SP*SY + CR*CY, -SR*CP])
    up = np.array([-(CR*SP*CY + SR*SY), CY*SR - CR*SP*SY, CR*CP])
    return fwd, right, up

class Camera:
    def __init__(self, path):
        head = open(path, encoding='cp1251').readline().strip().split(';')
        self.focal = float(head[-1])
        d = pd.read_csv(path, sep=';', encoding='cp1251').iloc[:, :7].astype(float)
        self.df = d.set_index('Frame')
        self.fx = self.focal / SENSOR_W * W
        self.fy = self.focal / SENSOR_H * H
        self.cx, self.cy = W / 2, H / 2
    def pose(self, f):
        r = self.df.loc[f]
        C = r[['Loc_X', 'Loc_Y', 'Loc_Z']].values
        fwd, right, up = ue_axes(r.Rot_Roll, r.Rot_Pitch, r.Rot_Yaw)
        Rm = np.stack([right, -up, fwd])  # world -> cam (x right, y down, z fwd)
        return C, Rm
    def P(self, f):
        C, Rm = self.pose(f)
        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        return K @ np.hstack([Rm, (-Rm @ C)[:, None]])
    def project(self, f, X):
        p = self.P(f) @ np.append(X, 1)
        return p[:2] / p[2], p[2]
    def ray(self, f, uv):
        C, Rm = self.pose(f)
        d = np.array([(uv[0]-self.cx)/self.fx, (uv[1]-self.cy)/self.fy, 1.0])
        d = Rm.T @ d
        return C, d / np.linalg.norm(d)

def load_cams(d):
    return [Camera(f'{d}/object_transform_CineCameraActor{i}.csv') for i in range(1, 6)]

def load_track(path):
    d = pd.read_csv(path, sep=';', encoding='cp1251', dtype=str)
    d = d.apply(lambda c: pd.to_numeric(c.str.replace(',', '.'), errors='coerce'))
    bad = d[d.isna().any(axis=1)]
    if len(bad): print('WARN: bad GT rows', bad.Frame.tolist())
    return d.set_index('Frame').interpolate()
