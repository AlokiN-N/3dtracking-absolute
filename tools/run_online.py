"""Launch the tracker with optional isolated Linux model dependencies."""
import argparse
import json
import os
from pathlib import Path
import runpy
import site
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--runtime", type=Path)
options, remaining = parser.parse_known_args()
repository = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository))
if options.runtime is not None:
    configuration = json.loads(options.runtime.read_text())
    for path in configuration.get("site_packages", []):
        directory = Path(path).resolve()
        if not directory.is_dir():
            raise ValueError(f"Runtime package directory is unavailable: {directory}")
        site.addsitedir(str(directory))
    cache = Path(configuration["cache"]).resolve()
    for key, folder in (("TORCH_HOME", "torch-cache"), ("HF_HOME", "huggingface"), ("TMPDIR", "tmp")):
        destination = cache / folder
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(destination)
sys.argv = ["online.run", *remaining]
runpy.run_module("online.run", run_name="__main__")
