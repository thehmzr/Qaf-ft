#!/bin/bash
# Downloads phone-recording files one at a time, scores them, and deletes each file after.
# usage: fetch_and_score.sh CKPT BASE INDEX [INDEX ...]
set -e
CKPT=$1; BASE=$2; shift 2
export HF_HUB_DISABLE_XET=1
for i in "$@"; do
  f=$(printf "data/clean-%05d.parquet" "$i")
  [ -s "data/phone_scores/clean-$(printf %05d "$i").jsonl" ] && [ "$(wc -l < data/phone_scores/clean-$(printf %05d "$i").jsonl)" -ge 4990 ] && continue
  .venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('deepdml/tlog-clean-sf16k','$f',repo_type='dataset',local_dir='data/tlog')" >/dev/null 2>&1
  .venv-nemo/bin/python data_tools/phone_filter.py --ckpt "$CKPT" --base "$BASE" --shards "data/tlog/$f" 2>&1 | grep -E "progress|Error|Traceback" || true
  [ "$i" != "0" ] && rm -f "data/tlog/$f"
done
echo done
