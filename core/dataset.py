"""Чтение датасета: сценарные CSV UE и пути к видео.

Источник данных — экспорт из Unreal Engine: `object_transform_<Actor>.csv`
(позиция/ориентация актора по кадрам) и `<Scene>.FinalImage.<Actor>.mp4`.

Формат CSV:
    Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35
    0;-24701.96;6623.57;1200.0;0.0;0.0;-70.0;;;матрица;23,76*13,365мм
    1;...

Разделитель `;`, кодировка CP1251 или UTF-8, единицы — сантиметры UE (X вперёд,
Y вправо, Z вверх), углы в градусах. Во второй строке камеры лежат
метаданные сенсора (`матрица;23,76*13,365мм`), в первой — фокусное.
В новом экспорте фокусное находится в столбце `Focal_Length_mm`, размер сенсора — в `matrix_size.txt`.

Данные в git не хранятся: `data/` в .gitignore, архив распаковывается
локально (см. README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

TRACK_ACTOR = "SM_AA_track"
CAMERA_ACTORS = tuple(f"CineCameraActor{i}" for i in range(1, 6))

VIDEO_GLOB = "Way1.FinalImage.{actor}.mp4"
CSV_GLOB = "object_transform_{actor}.csv"


@dataclass(frozen=True)
class Pose:
    """Позиция и ориентация актора в одном кадре."""

    frame: int
    loc: np.ndarray  # (3,) см
    rot: np.ndarray  # (3,) roll, pitch, yaw — градусы


@dataclass
class CameraTrack:
    """Временной ряд одной камеры."""

    index: int
    focal_mm: float
    sensor_mm: tuple[float, float]
    poses: dict[int, Pose]
    video: Path | None = None
    intrinsics_px: tuple[float, float, float, float] | None = None
    focal_by_frame: dict[int, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"CAM{self.index}"

    @property
    def is_static(self) -> bool:
        locs = np.array([p.loc for p in self.poses.values()])
        return bool(np.allclose(locs, locs[0], atol=1e-6))


@dataclass
class Scene:
    """Одна сцена: GT-трек цели + камеры + видео."""

    root: Path
    target: dict[int, Pose]
    cameras: dict[int, CameraTrack]
    width: int = 1920
    height: int = 1080
    fps: float = 30.0

    @property
    def frames(self) -> list[int]:
        return sorted(self.target)

    def video_path(self, cam: int) -> Path | None:
        return self.cameras[cam].video

    def gt_array(self) -> tuple[np.ndarray, list[int]]:
        frames = self.frames
        return np.stack([self.target[f].loc for f in frames]), frames


def _split(line: str) -> list[str]:
    # Не используем strip(): он съедает значимые пустые хвостовые поля.
    return line.rstrip("\r\n").split(";")


def _parse_float(text: str) -> float:
    return float(text.replace(",", ".").replace("\ufeff", "").strip())


def _read_text(path):
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1251", errors="replace")


def _sensor_dimensions(text):
    number = r"(\d+(?:[.,]\d+)?)"
    match = re.fullmatch(rf"\s*{number}\s*(?:mm|мм)?\s*[*×x]\s*{number}\s*(?:mm|мм)?\s*", text, re.I)
    if match:
        size = tuple(map(_parse_float, match.groups()))
        if min(size) > 0:
            return size
    return None


def read_sensor_size(root):
    path = root / "matrix_size.txt"
    if not path.is_file():
        return None
    size = _sensor_dimensions(_read_text(path))
    if size is None:
        raise ValueError(f"{path}: expected sensor width and height in millimetres")
    return size


def read_transform_csv(path: Path) -> tuple[dict[int, Pose], dict]:
    """Читает CSV трансформа. Возвращает (кадры, метаданные)."""
    poses: dict[int, Pose] = {}
    meta: dict = {"focal_mm": None, "sensor_mm": None, "focal_by_frame": {}}
    focal_column = None
    columns = None
    for lineno, raw in enumerate(_read_text(path).splitlines()):
        fields = _split(raw)
        if lineno == 0:
            # шапка: «...;Focal length;35» — ищем по подписи, а не по индексу:
            # число колонок зависит от наличия пустых полей в конце строки
            header = {name.strip().casefold(): pos for pos, name in enumerate(fields)}
            focal_column = header.get("focal_length_mm")
            names = ("frame", "loc_x", "loc_y", "loc_z", "rot_roll", "rot_pitch", "rot_yaw")
            if focal_column is not None and all(name in header for name in names):
                columns = [header[name] for name in names]
            focal_header = header.get("focal length")
            if focal_header is not None and focal_header + 1 < len(fields):
                meta["focal_mm"] = _parse_float(fields[focal_header + 1])
            continue
        for field in fields:
            size = _sensor_dimensions(field)
            if size is not None:
                meta["sensor_mm"] = size
        try:
            values_fields = fields[:7] if columns is None else [fields[pos] for pos in columns]
            frame = int(values_fields[0])
            values = [_parse_float(v) for v in values_fields[1:]]
            if focal_column is not None:
                focal = _parse_float(fields[focal_column])
                if not np.isfinite(focal) or focal <= 0:
                    continue
                meta["focal_by_frame"][frame] = focal
        except (ValueError, IndexError):
            continue
        if len(values) < 3:
            continue
        rot = list(values[3:6]) + [0.0] * (3 - len(values[3:6]))
        poses[frame] = Pose(
            frame=frame,
            loc=np.array(values[:3], dtype=np.float64),
            rot=np.array(rot[:3], dtype=np.float64),
        )
    if meta["focal_mm"] is None and meta["focal_by_frame"]:
        meta["focal_mm"] = next(iter(meta["focal_by_frame"].values()))
    return poses, meta


def load_scene(root: str | Path, width: int = 1920, height: int = 1080, fps: float = 30.0,
               require_gt: bool = True) -> Scene:
    """Загружает сцену из каталога с распакованным архивом.

    require_gt=False — для проверочного набора без трека объекта (только камеры и видео).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"нет каталога с данными: {root}")

    gt_path = root / CSV_GLOB.format(actor=TRACK_ACTOR)
    target = read_transform_csv(gt_path)[0] if gt_path.exists() else {}
    if not target and require_gt:
        raise RuntimeError(f"не найдены кадры GT в {root}")

    cameras: dict[int, CameraTrack] = {}
    for path in sorted(root.glob("object_transform_CineCameraActor*.csv")):
        match = re.fullmatch(r"object_transform_(CineCameraActor(\d+))\.csv", path.name)
        if match is None:
            continue
        actor, number = match.groups()
        idx = int(number)
        poses, meta = read_transform_csv(path)
        if not poses:
            continue
        sensor = meta["sensor_mm"] or read_sensor_size(root)
        if meta["focal_mm"] is None or sensor is None:
            raise RuntimeError(f"в {actor} не найдены фокус/сенсор")
        videos = sorted(root.glob(f"*{actor}.mp4"))
        if len(videos) > 1:
            raise ValueError(f"{actor}: found multiple matching videos in {root}")
        cameras[idx] = CameraTrack(
            index=idx,
            focal_mm=float(meta["focal_mm"]),
            sensor_mm=sensor,
            poses=poses,
            video=videos[0] if videos else None,
            focal_by_frame=meta["focal_by_frame"],
        )

    return Scene(root=root, target=target, cameras=cameras, width=width, height=height, fps=fps)
