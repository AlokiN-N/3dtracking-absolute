"""Encode BlenderProc RGB streams without exposing reference poses to inference."""
import argparse
import json
from pathlib import Path
import subprocess
parser=argparse.ArgumentParser()
parser.add_argument("scene",type=Path)
args=parser.parse_args()
scene=json.loads(args.scene.read_text())
for key in sorted({frame["camera"] for frame in scene["frames"]}):
    root=args.scene.parent
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate",str(scene["fps"]),
                    "-i",str(root/f"camera{key}"/"%06d.png"),"-c:v","libx264","-threads","2",
                    "-crf","18","-pix_fmt","yuv420p",str(root/f"camera{key}.mp4")],check=True)
