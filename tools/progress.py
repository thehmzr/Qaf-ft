"""Progress reporting for jobs started by tools/jobs.py.

A job calls report(done, total, note=..., unit=...) as it works. The runner
reads jobs/<id>/progress.json to show a progress bar and a time estimate.
Outside the runner (no QAF_JOB_DIR) report() just prints occasionally.
"""
import json
import os
import time
from pathlib import Path

_window = []          # recent (time, done) pairs for the rate estimate
_last_write = 0.0
_last_print = 0.0


def report(done, total=None, note="", unit="items", force=False):
    global _last_write, _last_print
    now = time.time()
    # a long gap means the job was paused: restart the rate window
    if _window and now - _window[-1][0] > 90:
        _window.clear()
    _window.append((now, done))
    if len(_window) > 40:
        del _window[0]
    rate = None
    if len(_window) >= 2 and _window[-1][0] > _window[0][0]:
        rate = (_window[-1][1] - _window[0][1]) / (_window[-1][0] - _window[0][0])
    job_dir = os.environ.get("QAF_JOB_DIR")
    finished = total is not None and done >= total
    if job_dir and (force or finished or now - _last_write > 2):
        path = Path(job_dir) / "progress.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(done=done, total=total, note=note, unit=unit,
                                       rate=rate, updated=now), ensure_ascii=False))
        os.replace(tmp, path)
        _last_write = now
    if force or finished or now - _last_print > 30:
        tot = f"/{total}" if total else ""
        print(f"[progress] {done}{tot} {unit} {note}", flush=True)
        _last_print = now
