"""Per-mode word-score thresholds from a named set of known-correct recitation.

  python eval/thresholds.py --ckpt models/full/best.pt [--name studio-val]
         [--files data/heldout/moshaf_19.0/*.parquet] [--limit N] [--out FILE.json]

Every word in the set is taken as correct. For each scoring mode (full, and basic =
words and vowels via to_basic) the thresholds are the score quantiles that flag 1%
(flag), 0.1% (strong) and 5% (p5) of those words. Writes a small json:
{name, full: {flag, strong, p5}, basic: {...}, n_words, n_clips, ckpt}.
"""
import argparse, glob, io, json, os, sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
from qafft.phonetic import fold_with_groups
from qafft.verify import extra_sounds, group_scores, to_basic, token_scores

SETS = {"studio-val": "data/heldout/moshaf_19.0/*.parquet"}


def badness(lp, ids, grps, blank):
    """per word: the weakest expected sound, or a clear extra sound (as eval/long_recitation.py)"""
    ts = token_scores(lp, ids, blank)
    if ts is None:
        return None
    bad = {g: -v for g, v in group_scores(ts, grps).items()}
    for g, v in extra_sounds(lp, ids, grps, blank).items():
        bad[g] = max(bad.get(g, 0.0), v)
    return bad


def both_modes(lp, tokens, ids, grps, blank):
    full = badness(lp, ids, grps, blank)
    blp, bids, bgrps, k = to_basic(lp, tokens, ids, grps)
    basic = badness(blp, bids, bgrps, k)
    if full is None or basic is None:
        return None
    return dict(full=full, basic=basic)


def levels(v):
    v = np.asarray(v)
    return dict(flag=float(np.quantile(v, 0.99)), strong=float(np.quantile(v, 0.999)), p5=float(np.quantile(v, 0.95)))


def load_model(ckpt, base=None):
    from pilot import QafModel
    ck = torch.load(ckpt, map_location="cpu")
    tokens = ck["tokens"]
    model = QafModel(len(tokens), base); model.load_state_dict(ck["model"]); model.eval()

    @torch.no_grad()
    def logp(wav):
        w = torch.from_numpy(wav).unsqueeze(0)
        lp, hl = model(w, torch.tensor([w.shape[1]]))
        return lp[0, : hl[0]]
    return tokens, logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--name", default="studio-val")
    ap.add_argument("--files", nargs="*", default=None, help="parquet files (default: the named set)")
    ap.add_argument("--limit", type=int, default=0, help="max clips in all (0 = no limit)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    files = a.files or sorted(glob.glob(os.path.join(ROOT, SETS[a.name])))
    if not files:
        sys.exit(f"no files for set {a.name!r}: {SETS.get(a.name)}")
    tokens, logp = load_model(a.ckpt, a.base)
    vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    vals, n_clips = {"full": [], "basic": []}, 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=["audio", "phonemes", "start_span", "duration_seconds"]).to_pylist():
                if a.limit and n_clips >= a.limit:
                    break
                if not r["start_span"] or r["duration_seconds"] > 20:
                    continue
                toks, grps = fold_with_groups(r["phonemes"])
                if any(t not in vocab for t in toks):
                    continue
                wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                s = both_modes(logp(wav), tokens, [vocab[t] for t in toks], grps, blank)
                if s is None:
                    continue
                for m in vals:
                    vals[m] += list(s[m].values())
                n_clips += 1
                if n_clips % 50 == 0:
                    print(n_clips, "clips", flush=True)
    out = dict(name=a.name, full=levels(vals["full"]), basic=levels(vals["basic"]),
               n_words=len(vals["full"]), n_clips=n_clips, ckpt=a.ckpt)
    path = a.out or os.path.join(ROOT, "data", "eval", "thresholds", f"{a.name}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1)); print("wrote", path)


if __name__ == "__main__":
    main()
