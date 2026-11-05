"""Motion proposals from the current frame and a calibrated two-frame lookback."""
import cv2
import numpy as np


def camera_warp(view, old, ratio=1):
    cam = view.camera
    K = np.array([[cam.fx*ratio, 0, cam.cx*ratio], [0, cam.fy*ratio, cam.cy*ratio], [0, 0, 1]])
    prev = old.camera
    old_k = K if (cam.fx, cam.fy, cam.cx, cam.cy) == (prev.fx, prev.fy, prev.cx, prev.cy) else np.array(
        [[prev.fx*ratio, 0, prev.cx*ratio], [0, prev.fy*ratio, prev.cy*ratio], [0, 0, 1]])
    return K @ view.matrix @ old.matrix.T @ np.linalg.inv(old_k)


def motion_boxes(current, past, view, old, compensate_exposure=False):
    if np.linalg.norm(old.camera.loc - view.camera.loc) > 5:
        return None
    ratio = min(0.5, 960 / current.shape[1])
    a, b = (cv2.GaussianBlur(cv2.resize(frame, None, fx=ratio, fy=ratio,
                                       interpolation=cv2.INTER_AREA), (3, 3), 0) for frame in (past, current))
    if not np.allclose(view.matrix, old.matrix):
        matrix = camera_warp(view, old, ratio)
        valid = cv2.warpPerspective(np.ones_like(a), matrix, a.shape[::-1])
        a = cv2.warpPerspective(a, matrix, a.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0
    else:
        valid = True
    if compensate_exposure:
        valid_grid = valid if np.isscalar(valid) else valid[::6, ::6]
        samples_a = a[::6, ::6] if np.isscalar(valid_grid) else a[::6, ::6][valid_grid]
        samples_b = b[::6, ::6] if np.isscalar(valid_grid) else b[::6, ::6][valid_grid]
        if samples_a.size >= 100:
            low_a, mid_a, high_a = np.percentile(samples_a, [20, 50, 80])
            low_b, mid_b, high_b = np.percentile(samples_b, [20, 50, 80])
            gain = np.clip((high_b - low_b) / max(high_a - low_a, 1), 0.5, 2.0)
            if abs(mid_a - mid_b) > 5 or abs(gain - 1) > 0.15:
                lut = np.clip((np.arange(256) - mid_a) * gain + mid_b, 0, 255).astype(np.uint8)
                a = cv2.LUT(a, lut)
    diff = ((cv2.absdiff(a, b) >= 12) & valid).astype(np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(diff)
    components = [row for row in stats[1:] if row[4] >= 24 and row[2] >= 5 and row[3] >= 7]
    components.sort(key=lambda row: row[4], reverse=True)
    remaining = components[:14]
    merged = []
    while remaining:
        x, y, w, h, area = remaining.pop(0)
        box = np.array([x, y, x+w, y+h])
        rest = []
        for x2, y2, w2, h2, area2 in remaining:
            gap = max(x - x2 - w2, x2 - x - w, 0)
            vert = min(y+h, y2+h2) - max(y, y2)
            if gap < 1.22 * max(h, h2) and vert > 0.24 * min(h, h2):
                box[:2] = np.minimum(box[:2], [x2, y2])
                box[2:] = np.maximum(box[2:], [x2+w2, y2+h2])
                area += area2
            else:
                rest.append((x2, y2, w2, h2, area2))
        remaining = rest
        size = box[2:] - box[:2]
        if area >= 60 and min(size) >= 10 and max(size) < 0.78 * min(b.shape):
            merged.append((box / ratio, int(area)))
    return sorted(merged, key=lambda entry: entry[1], reverse=True)
