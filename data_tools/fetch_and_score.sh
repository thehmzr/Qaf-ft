#!/bin/bash
# Downloads phone-recording files one at a time, scores them, and deletes each file after.
# The next file downloads while the current one is scored. A failed download or a failed
# scoring run skips that file (and keeps it on disk); run again to retry the missing ones.
# usage: fetch_and_score.sh CKPT BASE INDEX [INDEX ...]   (BASE can be "" to use the cached base model)
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

fetch "${todo[0]}" || echo "download failed: ${todo[0]}"
for k in "${!todo[@]}"; do
  i=${todo[$k]}; next=${todo[$((k + 1))]:-}
  pid=""
  if [ -n "$next" ]; then fetch "$next" & pid=$!; fi
  f="data/tlog/$(name "$i")"
  if [ -s "$f" ]; then
    $PY data_tools/phone_filter.py --ckpt "$CKPT" ${BASE:+--base "$BASE"} --shards "$f" 2>&1 | grep -E "progress|finished|Error|Traceback"
    if [ "${PIPESTATUS[0]}" -eq 0 ] && done_already "$i"; then
      [ "$i" != "0" ] && rm -f "$f"
    else
      echo "scoring failed: $i (file kept)"
    fi
  else
    echo "skipped $i: not downloaded"
  fi
  if [ -n "$pid" ]; then wait "$pid" || echo "download failed: $next"; fi
done
echo done
