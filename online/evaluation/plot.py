"""Write the local comparison plot without embedding reference points in the web replay."""
from pathlib import Path

import numpy as np


def plot_comparison(path: Path, table, fps: float) -> None:
    x0, x1 = 88, 1075
    rows = [(c, np.array(table.gt)[:, i] / 100, np.array(table.pred)[:, i] / 100)
            for i, c in enumerate("XYZ")]
    times = np.asarray(table.frames) / fps
    end = max(times[-1], 1 / fps)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800" role="img">',
        '<title>Координаты объекта по времени</title>',
        '<rect width="1100" height="800" fill="#11151b"/>',
        '<g fill="#eaf0f8" font-family="system-ui,sans-serif">',
        '<text x="34" y="44" font-size="25">Координаты объекта по времени</text>',
        '<text x="35" y="72" font-size="13" fill="#a5b3c2">Абсолютная система Unreal Engine · координаты в метрах</text>',
        '<text x="700" y="48" font-size="14" fill="#86f0b4">● Оценка</text>',
        '<text x="820" y="48" font-size="14" fill="#b5c5d9">● Эталон</text>',
    ]

    def line(a, b, c, d, color):
        svg.append(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}" stroke="{color}"/>')

    for i, (axis, truth, predicted) in enumerate(rows):
        top = 110 + i * 218
        bottom = top + 163
        finite = np.r_[truth, predicted[np.isfinite(predicted)]]
        lo, hi = finite.min(), finite.max()
        pad = max((hi - lo) * 0.08, 0.1)
        lo, hi = lo - pad, hi + pad
        svg.append(f'<text x="30" y="{top + 12}" font-size="15">{axis}, м</text>')
        for v in np.linspace(lo, hi, 5):
            y = round(bottom - (v - lo) / (hi - lo) * (bottom - top), 2)
            line(x0, y, x1, y, "#303b49")
            svg.append(f'<text x="{x0-8}" y="{y+4}" font-size="11" text-anchor="end">{v:.1f}</text>')
        for time in np.linspace(0, end, 7):
            x = round(x0 + time / end * (x1 - x0), 2)
            line(x, top, x, bottom, "#252e39")
            svg.append(f'<text x="{x}" y="{bottom+17}" font-size="11" text-anchor="middle">{time:.1f} с</text>')
        for series, color, width in ((truth, "#b5c5d9", 3), (predicted, "#86f0b4", 1.65)):
            segments = []
            previous = -2
            for frame, time, value in zip(table.frames, times, series):
                if not np.isfinite(value):
                    continue
                x = x0 + time / end * (x1 - x0)
                y = bottom - (value - lo) / (hi - lo) * (bottom - top)
                verb = "L" if frame == previous + 1 else "M"
                segments.append(f"{verb}{x:.2f} {y:.2f}")
                previous = frame
            path_data = " ".join(segments)
            svg.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="{width}"/>')
    svg.extend(["</g>", "</svg>"])
    path.write_text("\n".join(svg), encoding="utf-8")
