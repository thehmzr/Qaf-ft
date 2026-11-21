"""Adds the model's suggested mistakes to a labelling batch.

  python data_tools/propose.py --ckpt best.pt --queue data/queue_export --clean FILE.parquet

For each clip, scores every word of the claimed ayah with the checker. Words scoring
worse than 99% of known-correct words (measured on --clean, reciters never trained on)
become suggestions. One clip in five is left blind (no suggestions shown) so the labels
also contain an unbiased sample.
"""
import argparse, io, json, os, random, sys

import pyarrow.parquet as pq
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
from qafft.phonetic import fold_with_groups
from qafft.wordmap import word_tokens
from qafft.verify import extra_sounds, group_scores, token_scores

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--base", default=None)
ap.add_argument("--queue", required=True); ap.add_argument("--clean", required=True)
ap.add_argument("--blind", type=float, default=0.2)
a = ap.parse_args()
from pilot import QafModel
torch.set_num_threads(4)
ck = torch.load(a.ckpt, map_location="cpu")
tokens = ck["tokens"]; vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
model = QafModel(len(tokens), a.base); model.load_state_dict(ck["model"]); model.eval()


def word_badness(wav, toks, grps):
    with torch.no_grad():
        w = torch.from_numpy(wav).unsqueeze(0)
        lp, hl = model(w, torch.tensor([w.shape[1]]))
    lp = lp[0, : hl[0]]
    ids = [vocab[t] for t in toks]
    ts = token_scores(lp, ids, blank)
    if ts is None:
        return None
    bad = {g: -v for g, v in group_scores(ts, grps).items()}
    for g, v in extra_sounds(lp, ids, grps, blank).items():
        bad[g] = max(bad.get(g, 0.0), v)
    return bad


# threshold from known-correct words
null, pf = [], pq.ParquetFile(a.clean)
for rg in range(pf.num_row_groups):
    for r in pf.read_row_group(rg, columns=["audio", "phonemes", "duration_seconds"]).to_pylist():
        if len(null) > 1500 or not r["phonemes"] or r["duration_seconds"] > 20:
            continue
        toks, grps = fold_with_groups(r["phonemes"])
        if any(t not in vocab for t in toks):
            continue
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        b = word_badness(wav, toks, grps)
        if b:
            null += list(b.values())
null.sort()
thr = null[int(0.99 * len(null))]; strong = null[int(0.999 * len(null))]
print(f"threshold from {len(null)} correct words: {thr:.2f} (1%), {strong:.2f} (0.1%)", flush=True)

mf = os.path.join(a.queue, "manifest.json")
items = json.load(open(mf))["items"]
rng = random.Random(11)
n_flag = n_words = 0
for it in items:
    # proposals are shown on display words, so score per display word
    toks, tw, _ = word_tokens(it["sura"], it["aya"])
    wav, sr = sf.read(os.path.join(a.queue, it["file"]), dtype="float32")
    b = word_badness(wav, toks, tw) or {}
    it["blind"] = rng.random() < a.blind
    it["proposals"] = [dict(word=g, strength="strong" if v > strong else "possible") for g, v in sorted(b.items()) if v > thr]
    n_flag += len(it["proposals"]); n_words += len(b)
json.dump(dict(items=items, threshold=thr), open(mf, "w"), indent=1)
print(f"{len(items)} clips, {n_words} words, {n_flag} suggested ({n_flag / max(1, n_words):.0%}), "
      f"{sum(bool(x['proposals']) for x in items)} clips with a suggestion, {sum(x['blind'] for x in items)} blind")
