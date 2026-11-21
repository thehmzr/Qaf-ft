"""Looks for the cause of unstable validation numbers: scores clips one at a time and in
padded batches of 8, and lists where the errors concentrate."""
import io, json, os, sys
import pyarrow.parquet as pq, soundfile as sf, torch
from rapidfuzz.distance import Levenshtein
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train")); sys.path.insert(0, os.path.join(ROOT, "tools"))
from pilot import QafModel
from qafft.phonetic import fold
from progress import report

ckpt, base, shard, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
torch.set_num_threads(4)
ck = torch.load(ckpt, map_location="cpu"); tokens = ck["tokens"]; vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
model = QafModel(len(tokens), base); model.load_state_dict(ck["model"]); model.eval()
clips = []
pf = pq.ParquetFile(shard)
for rg in range(pf.num_row_groups):
    for r in pf.read_row_group(rg, columns=["audio", "phonemes", "duration_seconds"]).to_pylist():
        toks = fold(r["phonemes"] or "")
        if not toks or r["duration_seconds"] > 20 or any(t not in vocab for t in toks):
            continue
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        clips.append((torch.from_numpy(wav), [vocab[t] for t in toks], r["duration_seconds"]))
        if len(clips) >= limit: break
    if len(clips) >= limit: break

def decode(lp, n):
    return [i for i in torch.unique_consecutive(lp[:n].argmax(-1)).tolist() if i != blank]

single, batched = [], []
with torch.no_grad():
    for k, (w, ids, d) in enumerate(clips):
        lp, hl = model(w.unsqueeze(0), torch.tensor([len(w)]))
        single.append(Levenshtein.distance(decode(lp[0], hl[0]), ids)); report(k + 1, 2 * len(clips), note="one at a time", unit="clips")
    for b in range(0, len(clips), 8):
        chunk = clips[b:b + 8]
        wl = torch.tensor([len(c[0]) for c in chunk])
        lp, hl = model(torch.nn.utils.rnn.pad_sequence([c[0] for c in chunk], batch_first=True), wl)
        for j, c in enumerate(chunk):
            batched.append(Levenshtein.distance(decode(lp[j], hl[j]), c[1]))
        report(len(clips) + b + len(chunk), 2 * len(clips), note="padded batches", unit="clips")
tot = sum(len(c[1]) for c in clips)
print(f"clips {len(clips)} tokens {tot}")
print(f"one at a time : {sum(single) / tot:.4f}   clips with any error: {sum(e > 0 for e in single)}")
print(f"padded batches: {sum(batched) / tot:.4f}   clips with any error: {sum(e > 0 for e in batched)}")
worst = sorted(range(len(clips)), key=lambda k: -batched[k])[:8]
print("worst in batches (errors batched / single / tokens / seconds / position in batch):")
for k in worst:
    print(f"  {batched[k]:>3} / {single[k]:>3} / {len(clips[k][1]):>3} / {clips[k][2]:5.1f} / {k % 8}")
top = sum(sorted(batched, reverse=True)[:max(1, len(clips) // 50)])
print(f"share of batched errors in the worst 2% of clips: {top / max(1, sum(batched)):.2f}")
