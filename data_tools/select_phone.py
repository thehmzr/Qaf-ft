"""Picks phone clips for training from the scores written by phone_filter.py.

  python data_tools/select_phone.py [--max-err-basic 0.05] [--out data/phone_train.json]

Keeps clips whose free decoding is close to the claimed ayah on words and vowels.
Leaves out the benchmark holdout clips and every clip in the labelling queue.
A few whole shards are kept apart for phone validation.
Also writes the expected tokens of every ayah used, for the training notebook.
"""
import argparse, collections, glob, json, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default=os.path.join(ROOT, "data", "phone_scores"))
ap.add_argument("--max-err-basic", type=float, default=0.05)
ap.add_argument("--ratio", type=float, nargs=2, default=(0.8, 1.25), help="decoded / expected length")
ap.add_argument("--min-sec", type=float, default=1.0)
ap.add_argument("--max-sec", type=float, default=20.0)
ap.add_argument("--max-per-ayah", type=int, default=300, help="cap so a few ayahs do not dominate")
ap.add_argument("--val-shards", type=int, nargs="*", default=[10, 30, 50, 70])
ap.add_argument("--val-per-shard", type=int, default=250)
ap.add_argument("--out", default=os.path.join(ROOT, "data", "phone_train.json"))
ap.add_argument("--tokens-out", default=os.path.join(ROOT, "data", "phone_tokens.json"))
a = ap.parse_args()

drop = set()
hold = os.path.join(ROOT, "data", "tlog_holdout_ids.json")
if os.path.exists(hold):
    drop |= set(json.load(open(hold)))
else:
    print("warning: no holdout list at", hold)
drop |= {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ROOT, "data", "queue", "*.wav"))}

why = collections.Counter()
train, val = {}, {}
for f in sorted(glob.glob(os.path.join(a.scores, "clean-*.jsonl"))):
    idx = int(os.path.basename(f)[6:11])
    shard = f"data/clean-{idx:05d}.parquet"
    keep = []
    for line in open(f):
        r = json.loads(line)
        why["seen"] += 1
        if r.get("skip"):
            why["skipped when scoring"] += 1; continue
        if os.path.splitext(r["path"])[0] in drop:
            why["holdout or queue"] += 1; continue
        if not a.min_sec <= r["dur"] <= a.max_sec:
            why["length"] += 1; continue
        if not a.ratio[0] <= r["ratio"] <= a.ratio[1]:
            why["decoded length"] += 1; continue
        if r["err_basic"] > a.max_err_basic:
            why["err_basic"] += 1; continue
        keep.append([r["path"], f"{r['sura']}:{r['aya']}"])
    if idx in a.val_shards:
        val[shard] = keep[: a.val_per_shard]
    elif keep:
        train[shard] = keep

# cap clips per ayah, chosen at random across shards
rng = random.Random(0)
pool = [(s, c) for s, v in train.items() for c in v]
rng.shuffle(pool)
count, capped = collections.Counter(), collections.defaultdict(list)
for s, c in pool:
    count[c[1]] += 1
    if count[c[1]] <= a.max_per_ayah:
        capped[s].append(c)
    else:
        why["ayah cap"] += 1
train = {s: sorted(capped[s]) for s in sorted(capped)}

n_tr = sum(map(len, train.values())); n_va = sum(map(len, val.values()))
print("clips:", dict(why))
print(f"selected: train {n_tr} from {len(train)} shards, val {n_va} from {len(val)} shards")
ayahs = collections.Counter(k for v in train.values() for _, k in v)
print(f"train covers {len(ayahs)} ayahs; most repeated: {ayahs.most_common(3)}")

json.dump(dict(train=train, val=val, rule=vars(a)), open(a.out, "w"))
tokens = json.load(open(os.path.join(ROOT, "data", "ayah_tokens.json")))
used = {k for v in list(train.values()) + list(val.values()) for _, k in v}
json.dump({k: tokens[k] for k in sorted(used)}, open(a.tokens_out, "w"), ensure_ascii=False)
print("wrote", a.out, "and", a.tokens_out)
