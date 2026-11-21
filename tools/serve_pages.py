"""Serves the local pages in page/ on this machine only.

  python tools/serve_pages.py [--port 8777]

/state.json is built on each request from data/page_state.json, with live progress
for jobs that have a "live" entry (lines counted in result files).
"""
import argparse, glob, http.server, json, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(ROOT, "page")
STATE = os.path.join(ROOT, "data", "page_state.json")


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
        start = spec.get("start")
        if start and done:
            rate = done / max(1, now - start)
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
        if self.path.split("?")[0] == "/state.json":
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
