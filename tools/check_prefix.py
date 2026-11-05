"""Repeat a prefix using the exact source snapshot and arguments of a full run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=Path)
parser.add_argument("--limit", required=True, type=int)
args = parser.parse_args()
experiment = args.experiment.resolve()
manifest_path = experiment / "manifest.json"
manifest = json.loads(manifest_path.read_text())
full = json.loads((experiment / "result/replay.json").read_text())["frames"]
if not 0 < args.limit < len(full):
    raise ValueError("The prefix must be shorter than the completed full run")
command = list(manifest["command"])
output = experiment / f"prefix-{args.limit}"
output.mkdir(exist_ok=False)
command[command.index("--out") + 1] = str(output)
if "--limit" in command:
    command[command.index("--limit") + 1] = str(args.limit)
else:
    command.extend(["--limit", str(args.limit)])
runtime = Path(command[command.index("--runtime") + 1])
configuration = json.loads(runtime.read_text())
lock_path = Path(configuration["cache"]) / "experiment-gpu.lock"
lock_path.parent.mkdir(parents=True, exist_ok=True)
with lock_path.open("a") as lock, (output / "run.log").open("w", buffering=1) as stream:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    environment = {**os.environ, **manifest.get("environment", {})}
    subprocess.run(command, cwd=experiment / "source", env=environment,
                   stdout=stream, stderr=subprocess.STDOUT, check=True)
prefix = json.loads((output / "replay.json").read_text())["frames"]
expected = [{key: value for key, value in row.items() if key != "error_m"} for row in full[:args.limit]]
report = {"frames_checked": args.limit, "matches": prefix == expected,
          "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
(experiment / "causality.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["matches"] else 1)
