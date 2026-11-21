#!/bin/bash
# Downloads phone-recording files one at a time, scores them, and deletes each file after.
# The next file downloads while the current one is scored.
# usage: fetch_and_score.sh CKPT BASE INDEX [INDEX ...]   (BASE can be "" to use the cached base model)
set -e
CKPT=$1; BASE=$2; shift 2
PY=.venv/bin/python
export HF_HUB_DISABLE_XET=1 PYTORCH_ENABLE_MPS_FALLBACK=1

name() { printf "data/clean-%05d.parquet" "$1"; }
done_already() {
  local s="data/phone_scores/clean-$(printf %05d "$1").jsonl"
  [ -s "$s" ] && [ "$(wc -l < "$s")" -ge 4990 ]
}
fetch() {
  $PY -c "from huggingface_hub import hf_hub_download; hf_hub_download('deepdml/tlog-clean-sf16k','$(name "$1")',repo_type='dataset',local_dir='data/tlog')" >/dev/null 2>&1
}

todo=()
for i in "$@"; do done_already "$i" || todo+=("$i"); done
[ ${#todo[@]} -eq 0 ] && { echo done; exit 0; }

fetch "${todo[0]}"
for k in "${!todo[@]}"; do
  i=${todo[$k]}; next=${todo[$((k + 1))]:-}
  pid=""
  if [ -n "$next" ]; then fetch "$next" & pid=$!; fi
  $PY data_tools/phone_filter.py --ckpt "$CKPT" ${BASE:+--base "$BASE"} --shards "data/tlog/$(name "$i")" 2>&1 | grep -E "progress|finished|Error|Traceback" || true
  [ "$i" != "0" ] && rm -f "data/tlog/$(name "$i")"
  [ -n "$pid" ] && wait "$pid"
done
echo done
