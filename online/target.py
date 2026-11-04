"""Metric geometry and optional tracking priors for a single rigid target."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class Target:
    name: str
    parts: tuple
    labels: tuple = ()
    speed: float = 42.0
    ground_vehicle: bool = False
    level_surface: bool = False
    edge_part: np.ndarray | None = None
    reference_part: np.ndarray | None = None
    triangles: np.ndarray | None = None
    face_colors: np.ndarray | None = None
    vertices: np.ndarray = field(init=False, repr=False)
    center: np.ndarray = field(init=False)

    def __post_init__(self):
        if not self.parts or any(
            part.ndim != 2 or part.shape[1] != 3 or len(part) < 3 or not np.isfinite(part).all()
            for part in self.parts
        ):
            raise ValueError("Target parts must contain finite XYZ vertices")
        vertices = np.vstack(self.parts)
        if np.linalg.matrix_rank(vertices - vertices.mean(axis=0)) < 2:
            raise ValueError("Target geometry must span at least two dimensions")
        if not np.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("Target speed must be positive in m/s")
        if any(type(label) is not int or label < 0 for label in self.labels):
            raise ValueError("Target labels must be nonnegative detector class IDs")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "center", (vertices.min(axis=0) + vertices.max(axis=0)) / 2)

    @property
    def size(self):
        return float(np.linalg.norm(np.ptp(self.vertices, axis=0)))


def way1_target():
    from experiments.sam2_model.model import HULLS

    parts = {name: vertices * 3.1 / 100 for name, vertices in HULLS.items()}
    return Target("CargoTruck", tuple(parts.values()), labels=(2, 5, 7),
                  ground_vehicle=True, level_surface=True,
                  edge_part=parts["Cargo"], reference_part=parts["Cab"])


def mesh_parts(path):
    try:
        import trimesh
    except ImportError as exc:
        raise ValueError("Install trimesh and Pillow to load CAD meshes") from exc

    if path.suffix.lower() == ".fbx":
        assimp = shutil.which(os.environ.get("PELENG_ASSIMP", "assimp"))
        if assimp is None:
            raise ValueError("FBX needs Assimp (assimp-utils on Debian or PELENG_ASSIMP) or conversion to GLB")
        with TemporaryDirectory(prefix="peleng-fbx-") as directory:
            converted = Path(directory) / "target.glb"
            result = subprocess.run([assimp, "export", str(path.resolve()), str(converted), "-f", "glb2"],
                                    capture_output=True, text=True, timeout=90)
            if result.returncode != 0 or not converted.is_file():
                raise ValueError(f"Could not convert FBX: {result.stderr.strip() or result.stdout.strip()}")
            return mesh_parts(converted)
    scene = trimesh.load_scene(path)
    parts, triangles, colors = [], [], []
    for node in scene.graph.nodes_geometry:
        transform, key = scene.graph[node]
        mesh = scene.geometry[key].copy()
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        mesh.apply_transform(transform)
        points = mesh.vertices[np.unique(mesh.faces)]
        if len(points) > 4:
            try:
                points = points[ConvexHull(points).vertices]
            except QhullError:
                pass
        parts.append(np.asarray(points))
        triangles.append(np.asarray(mesh.triangles))
        visual = mesh.visual
        if visual.kind == "texture":
            color = np.asarray(visual.material.to_color(visual.uv))
            if color.ndim == 2 and len(color) == len(mesh.vertices):
                face_color = color[mesh.faces, :3].mean(axis=1)
            else:
                base = np.asarray(visual.material.main_color)[:3]
                face_color = np.broadcast_to(base, (len(mesh.faces), 3))
        else:
            face_color = np.asarray(visual.face_colors)[:, :3]
        colors.append(face_color)
    if not triangles:
        raise ValueError(f"No triangular surface in {path}")
    return parts, np.concatenate(triangles), np.concatenate(colors)


def load_target(path):
    """Convert native CAD units and shift vertices to the output reference point."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    scale = float(spec["meters_per_unit"])
    origin = np.asarray(spec.get("origin", [0, 0, 0]), dtype=float)
    axes = spec.get("axes", [0, 1, 2])
    if not np.isfinite(scale) or scale <= 0 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Specify positive meters_per_unit and a finite XYZ origin")
    if sorted(axes) != [0, 1, 2] or any(type(axis) is not int for axis in axes):
        raise ValueError("axes must be a permutation of [0, 1, 2]")
    triangles = colors = None
    if ("mesh" in spec) == ("parts" in spec):
        raise ValueError("Specify either mesh or parts in the target JSON")
    if "mesh" in spec:
        parts, triangles, colors = mesh_parts(path.parent / spec["mesh"])
        triangles = ((triangles - origin) * scale)[..., axes]
    else:
        parts = [np.asarray(part, dtype=float) for part in spec["parts"]]
    return Target(str(spec.get("name", path.stem)), tuple(((part - origin) * scale)[:, axes] for part in parts),
                  labels=tuple(spec.get("labels", ())), speed=float(spec.get("max_speed_mps", 42)),
                  level_surface=bool(spec.get("level_surface", False)),
                  triangles=triangles, face_colors=colors)
