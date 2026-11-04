"""Compare two evaluated runs and reject any frame that regresses."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_run(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError(f"Run online.evaluate first: {path}")
    frames = data["frames"]
    if not frames or any(row["frame"] != i for i, row in enumerate(frames)):
        raise ValueError(f"Expected a nonempty consecutive trajectory: {path}")
    return frames


def scores(errors):
    values = np.array(list(errors), dtype=float)
    if not len(values):
        return None
    return {"n": len(values), "mse_m2": float(np.mean(values**2)),
            "mae_m": float(np.mean(values)), "max_m": float(np.max(values))}


def compare(before, after, tail=51):
    if len(before) != len(after):
        raise ValueError("The runs have different frame counts")
    if tail <= 0:
        raise ValueError("The tail must contain at least one video frame")
    if any(not all(np.isfinite(row["xyz"])) or not np.isfinite(row["error_m"])
           for frames in (before, after) for row in frames if row["error_m"] is not None):
        raise ValueError("An evaluated prediction is invalid")

    paired, gained, lost = [], [], []
    for old, new in zip(before, after):
        frame = old["frame"]
        had_pose = old["xyz"][0] is not None
        has_pose = new["xyz"][0] is not None
        if had_pose and not has_pose:
            lost.append(frame)
        if has_pose and not had_pose:
            gained.append(frame)
        a, b = old["error_m"], new["error_m"]
        if had_pose and has_pose and (a is None) != (b is None):
            raise ValueError(f"The valid reference sets differ at frame {frame}")
        if a is not None and b is not None:
            paired.append((frame, a, b))

    improved = [frame for frame, a, b in paired if b < a]
    worsened = [frame for frame, a, b in paired if b > a]
    a = scores(a for _, a, _ in paired)
    b = scores(b for _, _, b in paired)
    threshold = len(before) - tail
    tail_before = scores(a for f, a, _ in paired if f >= threshold)
    tail_after = scores(b for f, _, b in paired if f >= threshold)
    same_unscored = all(old["xyz"] == new["xyz"] for old, new in zip(before, after)
                        if old["error_m"] is None and old["xyz"][0] is not None)
    return {
        "video_frames": len(before), "before": a, "after": b,
        "tail_before": tail_before, "tail_after": tail_after,
        "improved_frames": improved, "worsened_frames": worsened,
        "unchanged_frames": len(paired) - len(improved) - len(worsened),
        "gained_predictions": gained, "lost_predictions": lost,
        "same_unscored": same_unscored,
        "release_passes": bool(a and b and not worsened and not lost and same_unscored
                               and b["mse_m2"] < a["mse_m2"] and b["mae_m"] < a["mae_m"]
                               and b["max_m"] <= a["max_m"]),
        "passes": bool(a and b and not worsened and not lost
                       and all(b[key] < a[key] for key in ("mse_m2", "mae_m", "max_m"))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path, help="Baseline replay.json")
    parser.add_argument("--after", required=True, type=Path, help="Candidate replay.json")
    parser.add_argument("--tail", default=51, type=int)
    args = parser.parse_args()
    report = compare(read_run(args.before), read_run(args.after), args.tail)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
