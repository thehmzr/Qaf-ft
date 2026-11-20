#!/usr/bin/env python3
"""Qaf-ft background job runner.

Every long task (benchmark, checker, training run) is a "job" with its own
directory under jobs/<id>/ holding meta.json, log.txt, progress.json and flag
files. Jobs run detached in their own process group so they can be paused
(SIGSTOP), resumed (SIGCONT) and stopped (SIGTERM) as a unit, and they keep
running between agent turns.

  jobs.py submit --id ID --name NAME --kind KIND [--queue] -- CMD ...
  jobs.py status            human-readable table
  jobs.py export            JSON for the control page
  jobs.py pause|resume|stop|restart|prioritize ID
  jobs.py wait ID [ID ...]  block until the jobs reach a final state
  jobs.py dispatcher        run the queue (started automatically)

Queued jobs start one at a time (jobs/config.json: max_parallel) because the
machine has 4 CPU cores and every job uses all of them.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "jobs"
FINAL = {"done", "failed", "stopped"}


def jdir(job_id: str) -> Path:
    return JOBS / job_id


def load(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0); check its state
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except Exception:
        return True


def config() -> dict:
    cfg = load(JOBS / "config.json", {}) or {}
    cfg.setdefault("max_parallel", 1)
    return cfg


def launch(job_id: str) -> None:
    d = jdir(job_id)
    meta = load(d / "meta.json")
    for flag in ("exit_code", "stopped", "paused", "queued"):
        (d / flag).unlink(missing_ok=True)
    cmd = " ".join(shlex.quote(c) for c in meta["cmd"])
    wrapper = (
        f"cd {shlex.quote(meta['cwd'])}; export QAF_JOB_DIR={shlex.quote(str(d))}; "
        f"export PYTHONUNBUFFERED=1; {cmd}; echo $? > {shlex.quote(str(d / 'exit_code'))}"
    )
    log = open(d / "log.txt", "a")
    log.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log.flush()
    p = subprocess.Popen(
        ["bash", "-c", wrapper], stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    meta.update(pid=p.pid, started_at=time.time(), ended_at=None,
                runs=meta.get("runs", 0) + 1)
    save(d / "meta.json", meta)


def state_of(job_id: str) -> str:
    d = jdir(job_id)
    meta = load(d / "meta.json") or {}
    if (d / "queued").exists():
        return "queued"
    if (d / "exit_code").exists():
        try:
            code = int((d / "exit_code").read_text().strip() or 1)
        except ValueError:
            code = 1
        if code == 0:
            return "done"
        return "stopped" if (d / "stopped").exists() else "failed"
    if alive(meta.get("pid")):
        return "paused" if (d / "paused").exists() else "running"
    return "stopped" if (d / "stopped").exists() else "failed"


def all_ids():
    if not JOBS.exists():
        return []
    ids = [p.name for p in JOBS.iterdir() if (p / "meta.json").exists()]
    return sorted(ids, key=lambda i: (load(jdir(i) / "meta.json") or {}).get("submitted_at", 0))


def ensure_dispatcher() -> None:
    pidf = JOBS / "dispatcher.pid"
    try:
        if alive(int(pidf.read_text())):
            return
    except Exception:
        pass
    p = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "dispatcher"],
        stdout=open(JOBS / "dispatcher.log", "a"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    pidf.write_text(str(p.pid))


def cmd_dispatcher(_):
    while True:
        ids = all_ids()
        states = {i: state_of(i) for i in ids}
        active = sum(1 for s in states.values() if s in ("running", "paused"))
        queued = [i for i in ids if states[i] == "queued"]
        queued.sort(key=lambda i: (load(jdir(i) / "meta.json") or {}).get("queue_rank", 0))
        if queued and active < config()["max_parallel"]:
            launch(queued[0])
        time.sleep(2)


def cmd_submit(a):
    JOBS.mkdir(exist_ok=True)
    d = jdir(a.id)
    if d.exists() and state_of(a.id) in ("running", "paused", "queued"):
        sys.exit(f"job {a.id} is already {state_of(a.id)}")
    d.mkdir(parents=True, exist_ok=True)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    old = load(d / "meta.json") or {}
    meta = dict(id=a.id, name=a.name or a.id, kind=a.kind, cmd=cmd,
                cwd=a.cwd or str(ROOT), submitted_at=time.time(),
                queue_rank=time.time(), runs=old.get("runs", 0),
                resumable=a.resumable)
    save(d / "meta.json", meta)
    if a.queue:
        for flag in ("exit_code", "stopped", "paused"):
            (d / flag).unlink(missing_ok=True)
        (d / "queued").write_text("1")
        ensure_dispatcher()
        print(f"queued {a.id}")
    else:
        launch(a.id)
        print(f"started {a.id} pid={load(d / 'meta.json')['pid']}")


def signal_group(job_id: str, sig) -> bool:
    meta = load(jdir(job_id) / "meta.json") or {}
    pid = meta.get("pid")
    if not alive(pid):
        return False
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def cmd_pause(a):
    st = state_of(a.id)
    if st != "running":
        sys.exit(f"cannot pause: job is {st}")
    signal_group(a.id, signal.SIGSTOP)
    (jdir(a.id) / "paused").write_text(str(time.time()))
    print(f"paused {a.id}")


def cmd_resume(a):
    st = state_of(a.id)
    if st != "paused":
        sys.exit(f"cannot resume: job is {st}")
    (jdir(a.id) / "paused").unlink(missing_ok=True)
    signal_group(a.id, signal.SIGCONT)
    print(f"resumed {a.id}")


def cmd_stop(a):
    d = jdir(a.id)
    st = state_of(a.id)
    if st in FINAL:
        sys.exit(f"nothing to stop: job is {st}")
    (d / "stopped").write_text(str(time.time()))
    if st == "queued":
        (d / "queued").unlink(missing_ok=True)
        (d / "exit_code").write_text("143")
        print(f"removed {a.id} from the queue")
        return
    signal_group(a.id, signal.SIGCONT)
    signal_group(a.id, signal.SIGTERM)
    for _ in range(20):
        if not alive((load(d / "meta.json") or {}).get("pid")):
            break
        time.sleep(0.25)
    else:
        signal_group(a.id, signal.SIGKILL)
    if not (d / "exit_code").exists():
        (d / "exit_code").write_text("143")
    (d / "paused").unlink(missing_ok=True)
    print(f"stopped {a.id}")


def cmd_restart(a):
    st = state_of(a.id)
    if st not in FINAL:
        sys.exit(f"cannot run again: job is {st}")
    d = jdir(a.id)
    for flag in ("exit_code", "stopped", "paused"):
        (d / flag).unlink(missing_ok=True)
    meta = load(d / "meta.json")
    meta["queue_rank"] = time.time()
    save(d / "meta.json", meta)
    (d / "queued").write_text("1")
    ensure_dispatcher()
    print(f"queued {a.id} to run again")


def cmd_prioritize(a):
    if state_of(a.id) != "queued":
        sys.exit(f"cannot move: job is {state_of(a.id)}")
    d = jdir(a.id)
    meta = load(d / "meta.json")
    meta["queue_rank"] = 0
    save(d / "meta.json", meta)
    print(f"{a.id} will run next")


def tail(path: Path, n=3, width=170):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 6000))
            text = f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return []
    lines = []
    for raw in text.splitlines():
        line = raw.split("\r")[-1].strip()
        if line and not line.startswith("=== start"):
            lines.append(line[:width])
    return lines[-n:]


def snapshot(job_id: str) -> dict:
    d = jdir(job_id)
    meta = load(d / "meta.json") or {}
    prog = load(d / "progress.json", {}) or {}
    st = state_of(job_id)
    can = {"running": ["pause", "stop"], "paused": ["resume", "stop"],
           "queued": ["prioritize", "stop"],
           "done": ["restart"], "failed": ["restart"], "stopped": ["restart"]}[st]
    ended = None
    if st in FINAL and (d / "exit_code").exists():
        ended = (d / "exit_code").stat().st_mtime
    eta = None
    if st == "running" and prog.get("rate") and prog.get("total"):
        remaining = max(0, prog["total"] - prog.get("done", 0))
        eta = remaining / prog["rate"] if prog["rate"] > 0 else None
    return dict(
        id=job_id, name=meta.get("name", job_id), kind=meta.get("kind", "job"),
        status=st, done=prog.get("done"), total=prog.get("total"),
        unit=prog.get("unit", ""), note=prog.get("note", ""),
        eta_s=round(eta) if eta is not None else None,
        started_at=int(meta["started_at"] * 1000) if meta.get("started_at") and st != "queued" else None,
        ended_at=int(ended * 1000) if ended else None,
        log=tail(d / "log.txt"), can=can, runs=meta.get("runs", 0),
    )


def cmd_export(_):
    print(json.dumps([snapshot(i) for i in all_ids()], ensure_ascii=False))


def cmd_status(_):
    rows = [snapshot(i) for i in all_ids()]
    if not rows:
        print("no jobs")
        return
    for r in rows:
        prog = ""
        if r["total"]:
            prog = f"{r['done']}/{r['total']} {r['unit']}".strip()
        eta = f"eta {r['eta_s'] // 60}m{r['eta_s'] % 60:02d}s" if r["eta_s"] is not None else ""
        print(f"{r['id']:<28} {r['status']:<8} {prog:<22} {eta:<12} {r['note'][:60]}")
        if r["status"] == "failed" and r["log"]:
            print(f"    last log: {r['log'][-1]}")


def cmd_wait(a):
    while True:
        states = {i: state_of(i) for i in a.ids}
        if all(s in FINAL for s in states.values()):
            for i, s in states.items():
                print(f"{i}: {s}")
            return
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--id", required=True)
    s.add_argument("--name")
    s.add_argument("--kind", default="job")
    s.add_argument("--cwd")
    s.add_argument("--queue", action="store_true")
    s.add_argument("--resumable", action="store_true")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_submit)
    for name, fn in (("pause", cmd_pause), ("resume", cmd_resume), ("stop", cmd_stop),
                     ("restart", cmd_restart), ("prioritize", cmd_prioritize)):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.set_defaults(fn=fn)
    w = sub.add_parser("wait")
    w.add_argument("ids", nargs="+")
    w.set_defaults(fn=cmd_wait)
    for name, fn in (("status", cmd_status), ("export", cmd_export), ("dispatcher", cmd_dispatcher)):
        sub.add_parser(name).set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
