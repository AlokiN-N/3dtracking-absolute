"""Prepare a reproducible multi-camera scene with an arbitrary metric CAD."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.camera import Camera
from online.geometry import Rot
from online.target import load_target

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--target", type=Path)
parser.add_argument("--seed", type=int, default=437)
parser.add_argument("--frames", type=int, default=24)
parser.add_argument("--shape", choices=("assembly", "box", "cylinder"), default="assembly")
parser.add_argument("--occlusion", action="store_true")
parser.add_argument("--blackout-camera", type=int, choices=(3, 9, 14))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(args.seed)
if args.target:
    target = load_target(args.target)
    if target.triangles is None:
        raise ValueError("Validation rendering requires a triangular CAD mesh")
    mesh = trimesh.Trimesh(vertices=target.triangles.reshape(-1, 3),
                           faces=np.arange(target.triangles.size // 3).reshape(-1, 3),
                           process=False)
    mesh.visual.face_colors = np.c_[target.face_colors, np.full(len(target.face_colors), 255)]
else:
    if args.shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=.45, height=1.4, sections=12)
    elif args.shape == "box":
        mesh = trimesh.creation.box(extents=[2., .7, .9])
    else:
        parts = []
        for size, location in [([2.,.6,.5],[-.3,0,0]),([.5,1.4,.3],[.4,0,.3]),
                               ([.3,.4,.8],[-.8,.2,.4])]:
            part = trimesh.creation.box(extents=size)
            part.apply_translation(location)
            parts.append(part)
        mesh = trimesh.util.concatenate(parts)
    mesh.visual.face_colors = np.c_[rng.integers(65,235,(len(mesh.faces),3)),np.full(len(mesh.faces),255)]
mesh.export(args.out/"model.ply")
(args.out/"target.json").write_text(json.dumps(dict(mesh="model.ply", meters_per_unit=1,
                                                   name=args.shape, max_speed_mps=8)))
size = np.linalg.norm(np.ptp(mesh.vertices,axis=0))
offset = (mesh.vertices.min(0) + mesh.vertices.max(0))/2
intrinsic = [420, 405, 157, 121]
reflect = np.diag([1.,-1.,1.])
manifest={"cameras":[]}
scene={"model":"model.ply","frames":[],"width":320,"height":240,"intrinsic":intrinsic,
       "seed":args.seed,"fps":25,"occlusion":args.occlusion,
       "blackout_camera":args.blackout_camera,"frame_count":args.frames}
reference=[]
calibrations={key:[] for key in (3,9,14)}
for frame in range(args.frames):
    phase=frame/max(args.frames-1,1)
    center=np.array([size*3+.7*phase, -.4+.8*phase, size*.7+.3*np.sin(phase*3)])
    angles=np.array([.2+.5*phase,-.15+.3*phase,.12-.35*phase])
    rotation=Rot(*angles)
    origin=center-rotation@offset
    reference.append([frame,*(origin*100)])
    for i,key in enumerate(calibrations):
        location=np.array([0,(-1+i)*size*1.8,size*(.5+.4*i)])
        if key==14:location+=np.array([.3,-.4,.2])*phase
        direction=center-location
        orientation=np.rad2deg([.03 if key==14 else 0,np.arctan2(direction[2],np.linalg.norm(direction[:2])),
                               np.arctan2(direction[1],direction[0])])
        camera=Camera(key,location*100,orientation,0,(0,0),320,240,tuple(intrinsic))
        forward,right,down=camera.basis
        camera_matrix=np.eye(4)
        camera_matrix[:3,:3]=reflect@np.stack([right,-down,-forward],axis=1)
        camera_matrix[:3,3]=reflect@location
        object_matrix=np.eye(4)
        object_matrix[:3,:3]=reflect@rotation
        object_matrix[:3,3]=reflect@origin
        scene["frames"].append(dict(frame=frame,camera=key,camera_matrix=camera_matrix.tolist(),
                                    object_matrix=object_matrix.tolist(),
                                    energy=500 if frame<args.frames//2 else 160))
        calibrations[key].append([frame,*(location*100),*orientation])
for key,rows in calibrations.items():
    with (args.out/f"cal_{key}.csv").open("w") as stream:
        writer=csv.writer(stream,delimiter=";")
        writer.writerow(["Frame","Loc_X","Loc_Y","Loc_Z","Rot_Roll","Rot_Pitch","Rot_Yaw"])
        writer.writerows(rows)
    manifest["cameras"].append(dict(id=key,video=f"camera{key}.mp4",poses=f"cal_{key}.csv",intrinsics_px=intrinsic))
with (args.out/"reference.csv").open("w") as stream:
    writer=csv.writer(stream,delimiter=";")
    writer.writerow(["Frame","Loc_X","Loc_Y","Loc_Z"])
    writer.writerows(reference)
(args.out/"cameras.json").write_text(json.dumps(manifest,indent=2))
(args.out/"scene.json").write_text(json.dumps(scene))
