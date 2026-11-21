"""Scores phone recordings against the ayah their file name claims.

  python data_tools/phone_filter.py --ckpt best.pt --shards data/tlog/data/clean-00000.parquet [...]

For every clip: decode freely, compare with the expected sound tokens of that ayah.
Writes data/phone_scores/<shard>.jsonl with the token error and length ratio, so a
threshold can be chosen afterwards. Low error = the label can be trusted for training.
High error = wrong ayah, cut-off clip, noise, or a real mistake: candidates for labelling.
"""
import argparse, io, json, os, sys

import pyarrow.parquet as pq
import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train")); sys.path.insert(0, os.path.join(ROOT, "tools"))
from progress import report
from qafft.verify import basic_classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    from pilot import QafModel
    torch.set_num_threads(4)
    ck = torch.load(a.ckpt, map_location="cpu")
    tokens = ck["tokens"]; vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    model = QafModel(len(tokens), a.base); model.load_state_dict(ck["model"]); model.eval(); model.to(a.device)
    cls, _ = basic_classes(tokens)

    def basic(seq):
        """the same sequence judged on words and vowels only"""
        out = []
        for i in seq:
            c = cls[i]
            if c is not None and (not out or out[-1] != c):
                out.append(c)
        return out

    expected = json.load(open(os.path.join(ROOT, "data", "ayah_tokens.json")))
    out_dir = os.path.join(ROOT, "data", "phone_scores"); os.makedirs(out_dir, exist_ok=True)
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in a.shards)
    n = 0
    for f in a.shards:
        out = os.path.join(out_dir, os.path.basename(f).replace(".parquet", ".jsonl"))
        done = {json.loads(l)["path"] for l in open(out)} if os.path.exists(out) else set()
        pf = pq.ParquetFile(f)
        with open(out, "a") as fo:
            for rg in range(pf.num_row_groups):
                for r in pf.read_row_group(rg).to_pylist():
                    n += 1
                    path = r["audio"]["path"]
                    if path in done:
                        continue
                    s, ay = path.split("_")[:2]
                    ref = expected.get(f"{s}:{ay}")
                    rec = dict(path=path, sura=int(s), aya=int(ay), dur=round(r["duration"], 2))
                    if ref is None or r["duration"] > a.max_sec or any(t not in vocab for t in ref):
                        rec["skip"] = True
                    else:
                        try:
                            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                            if wav.ndim > 1: wav = wav.mean(1)
                            with torch.no_grad():
                                w = torch.from_numpy(wav).unsqueeze(0).to(a.device)
                                lp, hl = model(w, torch.tensor([w.shape[1]], device=a.device))
                            best = lp[0, : hl[0]].cpu().max(-1)
                            hyp = [i for i in torch.unique_consecutive(best.indices).tolist() if i != blank]
                            ids = [vocab[t] for t in ref]
                            rec.update(err=round(Levenshtein.distance(hyp, ids) / max(1, len(ids)), 4),
                                       ratio=round(len(hyp) / max(1, len(ids)), 3),
                                       conf=round(float(best.values.mean()), 4), n=len(ids),
                                       err_basic=round(Levenshtein.distance(basic(hyp), basic(ids)) / max(1, len(basic(ids))), 4))
                        except Exception as e:
                            rec["skip"] = True; rec["why"] = type(e).__name__
                    fo.write(json.dumps(rec) + "\n")
                    if n % 20 == 0:
                        fo.flush(); report(n, total, note="scoring phone clips", unit="clips")
    report(n, n, note="finished", unit="clips", force=True)


if __name__ == "__main__":
    main()
