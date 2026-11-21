#!/bin/bash
# Creates .venv with the pinned packages and checks that the model builds.
# usage: scripts/setup.sh [--no-check]
set -e
cd "$(dirname "$0")/.."

if command -v uv >/dev/null; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
elif command -v python3.12 >/dev/null; then
  python3.12 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
else
  echo "needs Python 3.12 (or uv)"; exit 1
fi

[ "$1" = "--no-check" ] && exit 0

# builds the network from the base model (downloads it once, about 460 MB)
# and loads models/full/best.pt if it is there
.venv/bin/python - <<'EOF'
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "train")
import torch
from pilot import QafModel
tokens = json.load(open("qafft/vocab.json"))["tokens"]
m = QafModel(len(tokens)).eval()
ck = "models/full/best.pt"
if os.path.exists(ck):
    m.load_state_dict(torch.load(ck, map_location="cpu")["model"])
    print("model loads, with", ck)
else:
    print("model builds; fetch", ck, "to load the trained weights (see docs/handover.md)")
with torch.no_grad():
    lp, hl = m(torch.zeros(1, 32000), torch.tensor([32000]))
print("forward ok:", tuple(lp.shape))
EOF
