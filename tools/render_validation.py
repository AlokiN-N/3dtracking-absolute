import blenderproc as bproc
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import bpy

parser=argparse.ArgumentParser()
parser.add_argument("scene",type=Path)
args=parser.parse_args()
root=args.scene.parent
scene=json.loads(args.scene.read_text())
bproc.init()
bproc.renderer.set_render_devices(use_only_cpu=True)
objects=bproc.loader.load_obj(str(root/scene["model"]))
for obj in objects:
    obj.set_cp("category_id",1)
    material = bproc.material.create("target")
    colors = obj.blender_obj.data.color_attributes
    if len(colors):
        material.map_vertex_color(layer_name=colors[0].name)
    else:
        material.set_principled_shader_value("Base Color", [.7,.35,.12,1])
    obj.replace_materials(material)
floor=bproc.object.create_primitive("PLANE",scale=[100,100,1])
floor.set_cp("category_id",0)
material=bproc.material.create("ground")
material.set_principled_shader_value("Base Color",[.22,.25,.28,1])
floor.replace_materials(material)
light=bproc.types.Light()
light.set_type("SUN")
light.set_rotation_euler([.4,-.5,-.2])
light.set_location([3,0,12])
if scene.get("occlusion"):
    start = scene["frames"][0]
    target_position = np.array(start["object_matrix"])[:3, 3]
    camera_position = np.array(start["camera_matrix"])[:3, 3]
    blocker = bproc.object.create_primitive("CUBE", scale=[.25, .4, .7])
    blocker.set_location(.8 * target_position + .2 * camera_position)
    blocker.set_cp("category_id", 2)
fx,fy,cx,cy=scene["intrinsic"]
bproc.camera.set_intrinsics_from_K_matrix([[fx,0,cx],[0,fy,cy],[0,0,1]],
                                         scene["width"],scene["height"])
for index,frame in enumerate(scene["frames"]):
    for obj in objects:
        obj.set_local2world_mat(np.array(frame["object_matrix"]))
        for component in ("location", "rotation_euler", "scale"):
            obj.blender_obj.keyframe_insert(data_path=component, frame=index)
    light.set_energy(frame["energy"] / 250,frame=index)
    bproc.camera.add_camera_pose(np.array(frame["camera_matrix"]),frame=index)
bproc.renderer.set_max_amount_of_samples(8)
bpy.context.scene.render.threads_mode="FIXED"
bpy.context.scene.render.threads=3
data=bproc.renderer.render()
segmentation=bproc.renderer.render_segmap(map_by=["class"], default_values={"category_id": 0})
seeds=[]
for index,frame in enumerate(scene["frames"]):
    directory=root/f"camera{frame['camera']}"
    directory.mkdir(exist_ok=True)
    rgb=data["colors"][index]
    if (frame["camera"] == scene.get("blackout_camera")
            and scene["frame_count"] // 3 <= frame["frame"] < scene["frame_count"] // 2):
        rgb = np.zeros_like(rgb)
    Image.fromarray(rgb[:,:,:3]).save(directory/f"{frame['frame']:06d}.png")
    if frame["frame"]==0:
        mask=segmentation["class_segmaps"][index]==1
        y,x=np.where(mask)
        if len(x):
            seeds.extend(["--seed-box",f"{frame['camera']}:{x.min()},{y.min()},{x.max()},{y.max()}"])
(root/"seed.args").write_text(" ".join(seeds))
