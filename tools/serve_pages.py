"""Serves the local pages in page/ on this machine only.

  python tools/serve_pages.py [--port 8777]

/long/NAME/ayahs/... serves the ayah clips made by data_tools/ayah_clips.py (page long.html).
/state.json is built on each request from data/page_state.json, with live progress
for jobs that have a "live" entry (lines counted in result files). The time left is worked
out from the rate over the last half hour of page refreshes.
"""
import argparse, collections, glob, http.server, json, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(ROOT, "page")
STATE = os.path.join(ROOT, "data", "page_state.json")
LONG = os.path.realpath(os.path.join(ROOT, "data", "long"))
HISTORY = collections.defaultdict(collections.deque)    # job id -> recent (time, done), for the rate


def count_lines(files):
    n = 0
    for f in files:
        with open(f, "rb") as fh:
            n += sum(1 for _ in fh)
    return n


def live(job, now):
    spec = job.pop("live")
    files = sorted(glob.glob(os.path.join(ROOT, spec["glob"])))
    done = count_lines(files)
    total = spec["total"]
    newest = max((os.path.getmtime(f) for f in files), default=0)
    job.update(done=done, total=total, unit=spec.get("unit", "items"))
    if done >= total * spec.get("done_at", 0.99):
        job["status"] = "done"
    elif now - newest < spec.get("stall_after", 600):
        job["status"] = "running"
        h = HISTORY[job["id"]]
        h.append((now, done))
        while len(h) > 2 and now - h[0][0] > 1800:      # rate over the last half hour
            h.popleft()
        if len(h) >= 2 and h[-1][1] > h[0][1]:
            rate = (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])
            job["eta_s"] = (total - done) / rate
    else:
        job["status"] = "stalled" if done else "queued"
    return job


def state():
    now = time.time()
    s = json.load(open(STATE, encoding="utf-8"))
    s["jobs"] = [live(j, now) if "live" in j else j for j in s.get("jobs", [])]
    s["served_at"] = int(now * 1000)
    return s


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PAGES, **kw)

    def do_GET(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self.send_error(403)
            return
        path = self.path.split("?")[0]
        if path.startswith("/long/"):              # ayah clips of a long recitation (local data)
            f = os.path.realpath(os.path.join(LONG, path[len("/long/"):]))
            if not f.startswith(LONG + os.sep) or not f.endswith((".wav", ".json")) or not os.path.isfile(f):
                self.send_error(404); return
            data = open(f, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav" if f.endswith(".wav") else "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/state.json":
            try:
                body = json.dumps(state(), ensure_ascii=False).encode()
                code = 200
            except Exception as e:
                body, code = json.dumps({"error": str(e)}).encode(), 500
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    print(f"http://127.0.0.1:{a.port}/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
