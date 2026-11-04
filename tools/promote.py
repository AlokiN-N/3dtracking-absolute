"""Publish a validated replay by atomically exchanging Linux result directories."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repository = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository))
from online.evaluation.compare import compare, read_run
from online.evaluation.evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=Path)
parser.add_argument("--public", required=True, type=Path)
parser.add_argument("--reference", required=True, type=Path)
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args()
args.experiment = args.experiment.resolve()
args.public = args.public.resolve()
args.reference = args.reference.resolve()
result = args.experiment / "result"
baseline = args.experiment / "baseline.json"
manifest = json.loads((args.experiment / "manifest.json").read_text())
reference_hash = manifest.get("reference_sha256")
if reference_hash and hashlib.sha256(args.reference.read_bytes()).hexdigest() != reference_hash:
    raise ValueError("The reference differs from the one used for this experiment")
evaluate(result, args.reference)
before, after = read_run(baseline), read_run(result / "replay.json")
if json.loads(baseline.read_text()) != json.loads((args.public / "replay.json").read_text()):
    raise ValueError("The published replay changed since this experiment started")
report = compare(before, after)
if not report["release_passes"]:
    raise ValueError("Publication requires improved MSE and MAE with no per-frame regressions")
if args.check_only:
    print(json.dumps(report, indent=2))
    raise SystemExit(0)
subprocess.run([sys.executable, "-m", "online.smooth", "--out", str(result),
                "--lag", "5", "--reference", str(args.reference)], cwd=repository, check=True)
video = result / "control.mp4"
if not video.exists():
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(result / "control-raw.mp4"), "-c:v", "libx264", "-threads", "2",
                    "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
frames = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                  "-show_entries", "stream=nb_frames", "-of", "default=nk=1:nw=1",
                                  str(video)], text=True).strip()
if int(frames) != len(after):
    raise ValueError("The video and coordinate stream have different lengths")
staged = Path(tempfile.mkdtemp(prefix=".results-", dir=args.public.parent))
try:
    staged.chmod(0o755)
    for name in ("control.mp4", "replay.json", "metrics.json", "trajectory.csv", "lag.json", "trajectory-lag.csv"):
        destination = staged / name
        shutil.copy2(result / name, destination)
        destination.chmod(0o644)
    libc = ctypes.CDLL(None, use_errno=True)
    exchange = libc.renameat2
    exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if exchange(-100, os.fsencode(args.public), -100, os.fsencode(staged), 2):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
except BaseException:
    shutil.rmtree(staged, ignore_errors=True)
    raise
backup = args.experiment / "previous-public"
if backup.exists():
    backup = args.experiment / staged.name.removeprefix(".")
shutil.move(str(staged), str(backup))
print(f"Published {result}; previous result saved in {backup}")
