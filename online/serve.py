"""Serve the replay and calculate smoothing on request."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

if __package__:
    from online import smooth
else:
    import smooth


MAX_FRAMES = 120


def handler(assets, output, reference=None, way2=None):
    public = {"/": assets / "index.html"}
    for name in ("index.html", "style.css", "viewer.js"):
        public[f"/{name}"] = assets / name
    for name in ("control.mp4", "replay.json", "trajectory.csv", "metrics.json", "lag.json", "trajectory-lag.csv"):
        public[f"/{name}"] = output / name
    smoothing = {"/api/smooth": (output, reference)}
    if way2 is not None:
        public["/way2/"] = assets / "index.html"
        for name in ("control.mp4", "replay.json", "trajectory.csv", "report.json", "quality-samples.json"):
            public[f"/way2/{name}"] = way2 / name
        smoothing["/way2/api/smooth"] = (way2, None)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(True)

        def do_HEAD(self):
            self.respond(False)

        def respond(self, body):
            url = urlsplit(self.path)
            if url.path == "/way2" and way2 is not None:
                self.send_response(308)
                self.send_header("Location", "/way2/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if url.path in smoothing:
                self.respond_smoothing(url.query, body, *smoothing[url.path])
                return
            path = public.get(url.path)
            if path is None or not path.is_file():
                self.send_error(404)
                return
            length = path.stat().st_size
            start, end = 0, length - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if match and any(match.groups()):
                    lo, hi = match.groups()
                    start = int(lo) if lo else max(0, length - int(hi))
                    end = min(end, int(hi)) if lo and hi else end
                if not match or not any(match.groups()) or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{length}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if requested else 200)
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
            self.end_headers()
            if body:
                with path.open("rb") as stream:
                    stream.seek(start)
                    left = end - start + 1
                    try:
                        while left:
                            chunk = stream.read(min(left, 128 * 1024))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            left -= len(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        def respond_smoothing(self, query, body, replay_output, replay_reference):
            try:
                if len(query) > 100:
                    raise ValueError("Query is too long")
                params = parse_qs(query, keep_blank_values=True, max_num_fields=3, strict_parsing=True)
                if params.keys() - {"k", "history", "method"} or any(len(v) != 1 for v in params.values()):
                    raise ValueError("Unexpected or duplicate parameters")
                lag = int(params.get("k", ["5"])[0])
                history = int(params.get("history", [str(lag)])[0])
                method = params.get("method", ["linear"])[0]
                if not (0 <= lag <= MAX_FRAMES and 0 <= history <= MAX_FRAMES):
                    raise ValueError(f"Frame counts must be between 0 and {MAX_FRAMES}")
                smoother = smooth.FixedLag(lag, history, method)
            except ValueError as error:
                self.send_error(400, str(error))
                return

            try:
                frames = json.loads((replay_output / "replay.json").read_text(encoding="utf-8"))["frames"]
            except FileNotFoundError:
                self.send_error(404, "Replay is unavailable")
                return
            delayed = []
            for row in frames:
                point = smoother.push(row["frame"], row["xyz"])
                if point is not None:
                    delayed.append(point)
            payload = smooth.lag_report(smoother, frames, delayed, replay_reference)
            encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if body:
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reference", type=Path, help="Reference positions for interactive metrics (CSV or JSON)")
    parser.add_argument("--way2", type=Path, help="Optional directory containing the Way2 diagnostic replay")
    args = parser.parse_args()
    assets = Path(__file__).resolve().parent.parent / "web"
    reference = smooth.load_reference(args.reference) if args.reference else None
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler(assets, args.out, reference, args.way2)) as server:
        print(f"http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
