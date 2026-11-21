"""Where each data label is fetched from.

The ids live in data/sources.json, kept with the source files and out of git:
  {"datasets": {"expert": "...", "phone": "...", "base": "...", ...}, ...}
Set QAF_SOURCES to use another file.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hub_id(label):
    path = os.environ.get("QAF_SOURCES") or os.path.join(ROOT, "data", "sources.json")
    try:
        with open(path) as f:
            return json.load(f)["datasets"][label]
    except (OSError, KeyError, ValueError) as e:
        raise SystemExit(f"no id for '{label}' in {path}: put the sources file there (see docs/handover.md)") from e
