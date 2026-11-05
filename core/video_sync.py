"""Map decoded video frames onto a shared calibration frame clock."""
from fractions import Fraction
import json
import subprocess


def frame_indices(timestamps, fps):
    times = sorted(Fraction(value) * fps for value in timestamps)
    frames = [round(value) for value in times]
    if (not frames or frames[0] < 0 or len(set(frames)) != len(frames)
            or any(abs(value - frame) > Fraction(1, 100) for value, frame in zip(times, frames))):
        raise ValueError("timestamps do not lie on a unique constant-rate frame clock")
    return frozenset(frames)


def video_timelines(videos, frame_counts):
    timelines = {}
    rate = None
    for key, video in videos.items():
        try:
            output = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
                "-show_entries", "stream=r_frame_rate,time_base:packet=pts,flags", "-of", "json", str(video),
            ], timeout=60)
            info = json.loads(output)
            stream, = info["streams"]
            fps = Fraction(stream["r_frame_rate"])
            time_base = Fraction(stream["time_base"])
            if fps <= 0 or time_base <= 0 or (rate is not None and rate != fps):
                raise ValueError("expected a common positive frame rate")
            times = [int(packet["pts"]) * time_base for packet in info["packets"]
                     if "D" not in packet.get("flags", "")]
            indices = frame_indices(times, fps)
            if frame_counts[key] > 0 and len(indices) != frame_counts[key]:
                raise ValueError("timestamp count differs from the video frame count")
        except FileNotFoundError as error:
            raise ValueError("Install FFmpeg for timestamp synchronization or select --sync index") from error
        except (KeyError, ValueError, subprocess.SubprocessError, ZeroDivisionError) as error:
            raise ValueError(f"{video}: could not synchronize timestamps: {error}") from error
        timelines[key] = indices
        rate = fps
    return float(rate), timelines
