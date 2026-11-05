"""Fetch pinned upstream sources and verify model weights on a Linux data disk."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, type=Path)
parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "configs/models.lock.json")
parser.add_argument("--verify-only", action="store_true")
args = parser.parse_args()
root = args.root.resolve()
root.mkdir(parents=True, exist_ok=True)
for model in json.loads(args.lock.read_text()):
    source = root / "vendor" / model["name"]
    if not source.exists() and not args.verify_only:
        source.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(["git", "clone", "--no-checkout", model["repository"], str(source)],
                       env=environment, check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", model["commit"]],
                       env=environment, check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != model["commit"]:
        raise ValueError(f"Source revision differs from the lock for {model['name']}")
    weights = model.get("weights")
    if weights is None:
        continue
    target = (root / weights["file"]).resolve()
    if root not in target.parents:
        raise ValueError("Weight path must be inside the configured data root")
    if not target.is_file() or checksum(target) != weights["sha256"]:
        if args.verify_only:
            raise ValueError(f"Missing or mismatched checkpoint: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urlopen(weights["url"], timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                stream.flush()
                if checksum(temporary) != weights["sha256"]:
                    raise ValueError(f"Checkpoint digest mismatch for {model['name']}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    if weights.get("cache_file"):
        cached = (root / weights["cache_file"]).resolve()
        if root not in cached.parents:
            raise ValueError("Cache path must be inside the configured data root")
        if cached.exists():
            if checksum(cached) != weights["sha256"]:
                raise ValueError(f"Cached checkpoint differs from the lock: {cached}")
        elif not args.verify_only:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.symlink_to(target)
    print(f"Verified {model['name']}", flush=True)
