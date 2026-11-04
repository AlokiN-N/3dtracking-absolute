"""Fixed-lag smoothing for a stream of measured positions."""
import argparse
from collections import deque
import csv
import json
from math import dist, fsum, isfinite, sqrt
from pathlib import Path

CSV_HEADER = ["Frame", "Available_At_Frame", "Loc_X", "Loc_Y", "Loc_Z"]
MISSING_XYZ = [None, None, None]
METHODS = ("linear", "mean")


def lag_frames(value):
    lag = int(value)
    if lag < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative frame count")
    return lag


class FixedLag:
    def __init__(self, lag, history=None, method="linear"):
        self.lag = lag
        self.history = lag if history is None else history
        self.method = method
        if min(self.lag, self.history) < 0 or method not in METHODS:
            raise ValueError("Invalid smoothing parameters")
        self.window = deque(maxlen=self.history + lag + 1)
        self.next_frame = 0

    def push(self, frame, xyz):
        """Emit the result for frame - lag once, upon receiving this frame."""
        if frame != self.next_frame:
            raise ValueError(f"Expected frame {self.next_frame}, got {frame}")
        self.next_frame += 1
        if len(xyz) != 3:
            raise ValueError("Expected three coordinates")
        position = tuple(xyz) if all(value is not None and isfinite(value) for value in xyz) else None
        self.window.append((frame, position))
        if frame < self.lag:
            return None

        target = frame - self.lag
        result = {"frame": target, "available_at": frame, "xyz": MISSING_XYZ.copy()}
        if self.window[-1 - self.lag][1] is None:
            return result
        observations = [(f - target, p) for f, p in self.window if p is not None]
        offsets, points = zip(*observations)
        count = len(points)
        sum_t = sum(offsets)
        if self.method == "mean" or count == 1 or sum_t == 0:
            fitted = [fsum(axis) / count for axis in zip(*points)]
        else:
            sum_t2 = sum(t * t for t in offsets)
            denominator = count * sum_t2 - sum_t * sum_t
            # The intercept of a least-squares line at the target frame (t=0).
            weights = [(sum_t2 - sum_t * t) / denominator for t in offsets]
            fitted = [fsum(w * value for w, value in zip(weights, axis)) for axis in zip(*points)]
        result["xyz"] = [round(float(value), 4) for value in fitted]
        return result


def csv_row(point):
    return [point["frame"], point["available_at"], *point["xyz"]]


def load_reference(path):
    if path.suffix == ".json":
        positions = json.loads(path.read_text(encoding="utf-8"))
    else:
        from core.dataset import read_transform_csv

        poses, _ = read_transform_csv(path)
        positions = {frame: pose.loc for frame, pose in poses.items()}
    return {int(frame): tuple(xyz) for frame, xyz in positions.items()
            if len(xyz) == 3 and all(isfinite(value) for value in xyz)}


def summarize(errors):
    if not errors:
        return {"n": 0}
    count = len(errors)
    ordered = sorted(errors)
    rank = 0.95 * (count - 1)
    lo = int(rank)
    p95 = ordered[lo] + (rank - lo) * (ordered[min(lo + 1, count - 1)] - ordered[lo])
    mse = fsum(error * error for error in errors) / count
    return {"n": count, "coverage": 1.0, "mse_m2": mse, "mae_m": fsum(errors) / count,
            "rmse_m": sqrt(mse), "median_m": (ordered[(count - 1) // 2] + ordered[count // 2]) / 2,
            "p95_m": p95, "max_m": ordered[-1]}


def lag_report(smoother, frames, delayed, reference=None):
    payload = {
        "k": smoother.lag,
        "history": smoother.history,
        "method": smoother.method,
        "video_frames": len(frames),
        "released_frames": len(delayed),
        "prediction_frames": sum(row["xyz"][0] is not None for row in delayed),
        "frames": delayed,
    }
    if reference is not None:
        before, after = [], []
        for row in delayed:
            frame = row["frame"]
            target = reference.get(frame)
            row["error_m"] = None
            if target is not None and row["xyz"][0] is not None:
                error = dist(row["xyz"], target) / 100
                row["error_m"] = error
                before.append(dist(frames[frame]["xyz"], target) / 100)
                after.append(error)
        payload["metrics"] = {"causal": summarize(before), "fixed_lag": summarize(after)}
    return payload


def write_replay(output, smoother, frames, delayed, reference=None):
    payload = lag_report(smoother, frames, delayed, reference)
    (output / "lag.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory containing replay.json")
    parser.add_argument("--lag", type=lag_frames, default=5)
    parser.add_argument("--history", type=lag_frames, help="Number of past frames (default: same as lag)")
    parser.add_argument("--method", choices=METHODS, default="linear")
    parser.add_argument("--reference", type=Path, help="Optional reference CSV for scoring")
    args = parser.parse_args()
    frames = json.loads((args.out / "replay.json").read_text(encoding="utf-8"))["frames"]
    smoother = FixedLag(args.lag, args.history, args.method)
    delayed = []
    with (args.out / "trajectory-lag.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(CSV_HEADER)
        for row in frames:
            point = smoother.push(row["frame"], row["xyz"])
            if point is not None:
                delayed.append(point)
                writer.writerow(csv_row(point))
    reference = load_reference(args.reference) if args.reference else None
    write_replay(args.out, smoother, frames, delayed, reference)


if __name__ == "__main__":
    main()
