"""Run an immutable experiment, then evaluate its completed output."""
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--runtime", required=True, type=Path)
parser.add_argument("--data", type=Path)
parser.add_argument("--cameras", type=Path)
parser.add_argument("--reference", required=True, type=Path)
parser.add_argument("--baseline", type=Path)
parser.add_argument("tracker_args", nargs=argparse.REMAINDER)
args = parser.parse_args()
if (args.data is None) == (args.cameras is None):
    parser.error("Specify either --data or --cameras")
repository = Path(__file__).resolve().parents[1]
args.out.mkdir(parents=True, exist_ok=False)
if args.baseline:
    shutil.copy2(args.baseline, args.out / "baseline.json")
reference = args.out / ("reference" + args.reference.suffix)
shutil.copy2(args.reference, reference)
snapshot = args.out / "source"
snapshot.mkdir()
for name in ("online", "core", "experiments", "tools"):
    shutil.copytree(repository/name, snapshot/name, ignore=shutil.ignore_patterns("__pycache__"))
(snapshot/"models").symlink_to((repository/"models").resolve(), target_is_directory=True)
hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in snapshot.rglob("*.py")}
configuration = json.loads(args.runtime.read_text())
(args.out/"runtime.json").write_text(json.dumps(configuration,indent=2))
tracker_args = args.tracker_args[1:] if args.tracker_args[:1] == ["--"] else args.tracker_args
result = args.out / "result"
command = [sys.executable, str(snapshot/"tools/run_online.py"),
           "--runtime", str(args.runtime.resolve()), "--out", str(result.resolve())]
command += ["--data", str(args.data.resolve())] if args.data else ["--cameras", str(args.cameras.resolve())]
command += tracker_args
weights = {}
sources = {}
for index, argument in enumerate(tracker_args[:-1]):
    value = Path(tracker_args[index + 1])
    if argument.endswith("-checkpoint") and value.is_file():
        with value.open("rb") as stream:
            weights[str(value.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
    if argument.endswith("-source") and (value/".git").exists():
        sources[str(value.resolve())] = subprocess.check_output(
            ["git", "-C", str(value), "rev-parse", "HEAD"], text=True).strip()
runtime_keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF")
manifest = {"command":command, "source_sha256":hashes, "weights_sha256":weights,
            "vendor_revisions":sources, "python":sys.version,
            "environment":{key: os.environ[key] for key in runtime_keys if key in os.environ},
            "reference_sha256":hashlib.sha256(reference.read_bytes()).hexdigest()}
model_lock = repository / "configs/models.lock.json"
if model_lock.exists():
    manifest["model_lock"] = json.loads(model_lock.read_text())
(args.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
environment = {**os.environ, "PYTHONPATH":str(snapshot)}
with (args.out/"run.log").open("w",buffering=1) as stream:
    lock_path = Path(configuration["cache"]) / "experiment-gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        subprocess.run(command,cwd=snapshot,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,"-m","online.evaluate","--out",str(result.resolve()),
                    "--reference",str(reference.resolve())],cwd=snapshot,env=environment,
                   stdout=stream,stderr=subprocess.STDOUT,check=True)
if args.baseline:
    sys.path.insert(0,str(snapshot))
    from online.evaluation.compare import compare,read_run
    before,after=read_run(args.out/"baseline.json"),read_run(result/"replay.json")
    if len(after) > len(before):
        raise ValueError("The baseline is shorter than this experiment")
    complete = len(after) == len(before)
    report=compare(before[:len(after)],after)
    report["complete_sequence"] = complete
    report["release_passes"] = report["release_passes"] and complete
    (args.out/"comparison.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
