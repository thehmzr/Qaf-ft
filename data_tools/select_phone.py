"""Picks phone clips for training from the scores written by phone_filter.py.

  python data_tools/select_phone.py [--max-err-basic 0.05]

Keeps clips whose free decoding is close to the claimed ayah on words and vowels.
Leaves out the benchmark holdout clips and every clip in the labelling queue.
Whole files are set apart before anything else: a few as phone-heldout (test, never trained
on, never used to choose anything) and a few for phone validation. Only fully scored files
are used.

Validation clips pass the same filter as training clips, so they are clips the scoring model
already decodes well. Phone validation checks that training did not break anything; whether
phone audio got better is judged on human-labelled clips.

Writes, under data/:
  phone_split.json     which files are heldout and which are validation (fixed by --seed)
  phone_train.json     {"train": {file: [[clip, "sura:aya"], ...]}, "val": {...}, "rule": {...}}
  phone_tokens.json    expected tokens of every ayah used
  phone_manifest.jsonl one line per picked clip with its scores
"""
import argparse, collections, glob, json, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_FILES = 79

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="data/phone_scores")
ap.add_argument("--max-err-basic", type=float, default=0.05)
ap.add_argument("--ratio", type=float, nargs=2, default=(0.8, 1.25), help="decoded / expected length")
ap.add_argument("--min-sec", type=float, default=1.0)
ap.add_argument("--max-sec", type=float, default=20.0)
ap.add_argument("--max-per-ayah", type=int, default=30, help="cap so a few ayahs do not dominate")
ap.add_argument("--n-heldout", type=int, default=4)
ap.add_argument("--n-val", type=int, default=4)
ap.add_argument("--val-per-file", type=int, default=250)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-dir", default="data")
a = ap.parse_args()
P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
rng = random.Random(a.seed)

# split by whole file first, with its own generator so reruns pick the same clips;
# file 0 holds the labelling queue, so it stays a training file
split_path = P(os.path.join(a.out_dir, "phone_split.json"))
pick = random.Random(a.seed).sample(range(1, N_FILES), a.n_heldout + a.n_val)
split = dict(heldout=sorted(pick[: a.n_heldout]), val=sorted(pick[a.n_heldout:]), seed=a.seed)
if os.path.exists(split_path):
    old = json.load(open(split_path))
    if old != split:
        raise SystemExit(f"{split_path} was made with other settings ({old}); the phone test files must not "
                         "change once chosen. Use the same --seed/--n-heldout/--n-val, or delete the file on purpose.")
else:
    json.dump(split, open(split_path, "w"), indent=1)
heldout, val_files = set(split["heldout"]), set(split["val"])

drop = set()
hold = P("data/tlog_holdout_ids.json")
if os.path.exists(hold):
    drop |= set(json.load(open(hold)))
else:
    raise SystemExit(f"missing {hold}: build it from the benchmark's tlog_holdout metadata first")
drop |= {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(P("data/queue/*.wav"))}

why = collections.Counter()
train, val, manifest = collections.defaultdict(list), {}, []
seen_files = []
def complete(f):
    return os.path.exists(f + ".done") or sum(1 for _ in open(f)) >= 4990


def rows(f):
    for line in open(f):
        try:
            yield json.loads(line)
        except ValueError:
            pass                            # half-written line from a stopped run


partial = []
for f in sorted(glob.glob(os.path.join(P(a.scores), "clean-*.jsonl"))):
    idx = int(os.path.basename(f)[6:11])
    if not complete(f):
        partial.append(idx); continue
    seen_files.append(idx)
    shard = f"data/clean-{idx:05d}.parquet"
    if idx in heldout:
        why["heldout file"] += sum(1 for _ in rows(f)); continue
    keep = []
    for r in rows(f):
        why["seen"] += 1
        if r.get("skip"):
            why["skipped when scoring"] += 1; continue
        if os.path.splitext(r["path"])[0] in drop:
            why["holdout or queue clip"] += 1; continue
        if not a.min_sec <= r["dur"] <= a.max_sec:
            why["length"] += 1; continue
        if not a.ratio[0] <= r["ratio"] <= a.ratio[1]:
            why["decoded length"] += 1; continue
        if r["err_basic"] > a.max_err_basic:
            why["err_basic"] += 1; continue
        keep.append(r)
    if idx in val_files:
        rng.shuffle(keep)
        val[shard] = [[r["path"], f"{r['sura']}:{r['aya']}"] for r in keep[: a.val_per_file]]
    else:
        for r in keep:
            train[r["sura"], r["aya"]].append((shard, r))

# cap clips per ayah, chosen at random
picked = collections.defaultdict(list)
for (s, ay), rows in sorted(train.items()):
    rng.shuffle(rows)
    why["ayah cap"] += max(0, len(rows) - a.max_per_ayah)
    for shard, r in rows[: a.max_per_ayah]:
        picked[shard].append([r["path"], f"{s}:{ay}"])
        manifest.append(dict(file=shard, clip=r["path"], sura=s, aya=ay, dur=r["dur"], err=r["err"],
                             err_basic=r["err_basic"], ratio=r["ratio"], lengths_unverified=r["err"] > a.max_err_basic))
train_sel = {k: sorted(v) for k, v in sorted(picked.items())}

n_tr = sum(map(len, train_sel.values())); n_va = sum(map(len, val.values()))
hours = sum(m["dur"] for m in manifest) / 3600
missing = sorted(set(range(N_FILES)) - set(seen_files))
print("clips:", dict(why))
print(f"files fully scored: {len(seen_files)} of {N_FILES}" + (f"; still being scored: {partial}" if partial else ""))
print(f"heldout files {sorted(heldout)}, val files {sorted(val_files)}")
print(f"selected: train {n_tr} clips ({hours:.1f} h) from {len(train_sel)} files covering {len(train)} ayahs; val {n_va}")
if not val:
    print("warning: no validation file has scores yet")
by_sura = collections.Counter(m["sura"] for m in manifest)
print("train clips by sura (top 10):", by_sura.most_common(10))
print("lengths unverified:", sum(m["lengths_unverified"] for m in manifest))

rule = {k: v for k, v in vars(a).items()}
json.dump(dict(train=train_sel, val=val, rule=rule, split=split),
          open(P(os.path.join(a.out_dir, "phone_train.json")), "w"))
tokens = json.load(open(P("data/ayah_tokens.json")))
used = {k for v in list(train_sel.values()) + list(val.values()) for _, k in v}
json.dump({k: tokens[k] for k in sorted(used)}, open(P(os.path.join(a.out_dir, "phone_tokens.json")), "w"), ensure_ascii=False)
with open(P(os.path.join(a.out_dir, "phone_manifest.jsonl")), "w") as fo:
    for m in manifest:
        fo.write(json.dumps(m) + "\n")
print("wrote phone_split.json, phone_train.json, phone_tokens.json, phone_manifest.jsonl in", a.out_dir)
