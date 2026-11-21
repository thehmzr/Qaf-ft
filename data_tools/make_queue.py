"""Builds a labelling queue from scored phone clips.

  python data_tools/make_queue.py --scores data/phone_scores/clean-00000.jsonl \
      --shard data/phone_src/data/clean-00000.parquet --out data/queue_export --n 100

Picks clips whose length fits the ayah but whose sounds partly differ (real mistake or
model error: only a listener can tell), mixes in some close matches as hidden controls,
and writes 16 kHz WAV files plus manifest.json. The manifest does not show the model's
opinion to the labeller.
"""
import argparse, io, json, os, random

import pyarrow.parquet as pq
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True); ap.add_argument("--shard", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--controls", type=float, default=0.15)
a = ap.parse_args()
rng = random.Random(7)
rows = [json.loads(l) for l in open(a.scores)]
rows = [r for r in rows if not r.get("skip") and 2 <= r["dur"] <= 15]
unsure = [r for r in rows if 0.1 < r["err"] <= 0.5 and 0.7 <= r["ratio"] <= 1.3]
close = [r for r in rows if r["err"] <= 0.02]
n_ctl = int(a.n * a.controls)
rng.shuffle(unsure); rng.shuffle(close)
picked = [dict(r, group="unsure") for r in unsure[: a.n - n_ctl]] + [dict(r, group="control") for r in close[:n_ctl]]
rng.shuffle(picked)
want = {r["path"]: r for r in picked}
os.makedirs(a.out, exist_ok=True)
pf = pq.ParquetFile(a.shard)
for rg in range(pf.num_row_groups):
    for r in pf.read_row_group(rg, columns=["audio"]).to_pylist():
        p = r["audio"]["path"]
        if p in want:
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
            name = os.path.splitext(p)[0] + ".wav"
            sf.write(os.path.join(a.out, name), wav, sr, subtype="PCM_16")
            want[p]["file"] = name
items = [dict(file=r["file"], sura=r["sura"], aya=r["aya"]) for r in picked if "file" in r]
json.dump(dict(items=items), open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
# kept apart from the queue: what the model thought, for scoring afterwards
json.dump({r["file"]: dict(group=r["group"], err=r["err"], ratio=r["ratio"]) for r in picked if "file" in r},
          open(a.out.rstrip("/") + "_model_view.json", "w"), indent=1)
print(len(items), "clips,", sum(r["group"] == "control" for r in picked), "controls")
