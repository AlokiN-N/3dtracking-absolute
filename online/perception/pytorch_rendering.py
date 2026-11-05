"""Optional PyTorch3D rasterizer with calibrated image-space intrinsics."""
import numpy as np

from online.geometry import Rot
from online.perception.rendering import RenderedCrop, crop_transform


def render_crop(model, pose, view, box, size=280):
    import torch
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    transform = crop_transform(box, size)
    world = model.triangles @ Rot(*pose[3:]).T + pose[:3]
    camera_vertices = (world - view.camera.loc / 100) @ view.matrix.T
    vertices = torch.tensor(camera_vertices.reshape(-1, 3), dtype=torch.float32)
    faces = torch.arange(len(vertices), dtype=torch.int64).reshape(-1, 3)
    cam = view.camera
    intrinsic = transform @ np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.]])
    # PyTorch3D samples pixel centers at half-integer screen coordinates.
    intrinsic[:2, 2] += 0.5
    cameras = cameras_from_opencv_projection(
        R=torch.eye(3)[None], tvec=torch.zeros(1, 3),
        camera_matrix=torch.tensor(intrinsic, dtype=torch.float32)[None],
        image_size=torch.tensor([[size, size]]))
    settings = RasterizationSettings(image_size=size, faces_per_pixel=1,
                                     perspective_correct=True, cull_backfaces=False,
                                     z_clip_value=0.05)
    fragments = MeshRasterizer(cameras=cameras, raster_settings=settings)(
        Meshes(verts=[vertices], faces=[faces]))
    face = fragments.pix_to_face[0, ..., 0].numpy()
    bary = fragments.bary_coords[0, ..., 0, :].numpy()
    mask = face >= 0
    points = np.zeros((size, size, 3), np.float32)
    points[mask] = np.einsum("ni,nij->nj", bary[mask], model.triangles[face[mask]])
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[mask] = model.face_colors[face[mask]]
    return RenderedCrop(rgb, points, mask, transform)
