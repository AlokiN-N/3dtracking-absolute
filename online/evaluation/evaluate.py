"""Score a completed run against valid reference rows without modifying them."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from core.dataset import read_transform_csv
from core.metrics import build_errors
from online.evaluation.plot import plot_comparison


def evaluate(output, reference):
    replay_path = output / "replay.json"
    data = json.loads(replay_path.read_text(encoding="utf-8"))
    gt, _ = read_transform_csv(reference)
    frames = data["frames"]
    valid = {r["frame"]: gt[r["frame"]].loc for r in frames
             if r["frame"] in gt and np.isfinite(gt[r["frame"]].loc).all()}
    prediction = {r["frame"]: np.asarray(r["xyz"]) for r in frames if r["xyz"][0] is not None}
    views = {r["frame"]: len(r["cams"]) for r in frames}
    table = build_errors(valid, prediction, views)
    metrics = table.summary()
    metrics.update(video_frames=len(frames), reference_frames=len(valid),
                   invalid_or_missing_reference=len(frames) - len(valid),
                   prediction_frames=len(prediction),
                   statuses=dict(Counter(r["status"] for r in frames)))
    if metrics["n"]:
        metrics["mse_m2"] = metrics["rmse_m"] ** 2
        metrics["mse_axis_m2"] = (np.mean(((np.array(table.pred) - np.array(table.gt))[table.valid] / 100) ** 2,
                                          axis=0)).tolist()
        metrics["error_gt_16m_frames"] = sum(e > 16 for e in table.errors if np.isfinite(e))
        metrics["error_lt_1m_frames"] = int(np.count_nonzero(np.array(table.errors) < 1))
        metrics["error_lt_1m_rate"] = metrics["error_lt_1m_frames"] / len(valid)
    errors = {f: (e if np.isfinite(e) else None) for f, e in zip(table.frames, table.errors)}
    for row in frames:
        row["error_m"] = errors.get(row["frame"])
    data["metrics"] = metrics
    replay_path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if table.frames:
        plot_comparison(output / "coordinates.svg", table, data["metadata"]["fps"])
        with (output / "errors.csv").open("w", newline="") as file:
            rows = table.as_rows()
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.out, args.reference), indent=2))


if __name__ == "__main__":
    main()
