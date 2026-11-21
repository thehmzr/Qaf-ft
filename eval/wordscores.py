"""Collects several per-word scores on planted mistakes (same protocol as
evaluate.py --shards) and writes them to a JSON file for comparison.

  python eval/wordscores.py --ckpt models/full/best.pt --shards F.parquet [...] \
      --limit 200 --first-sura 78 --out data/eval/wordscores/raw.json
"""
import argparse, io, json, os, random, sys

import pyarrow.parquet as pq
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
from qafft.phonetic import fold_with_groups
from qafft.mistakes import perturb
from qafft.verify import (extra_sounds, gop_sf_scores, group_scores, margin_scores,
                          to_basic, token_scores, token_spans)


def all_scores(lp, ids, grps, blank):
    ts = token_scores(lp, ids, blank)
    sp = token_spans(lp, ids, blank)
    if ts is None or sp is None:
        return None
    weak = {g: -v for g, v in group_scores(ts, grps).items()}
    extra = extra_sounds(lp, ids, grps, blank)
    cur = {g: max(v, extra.get(g, 0.0)) for g, v in weak.items()}
    gop = {g: -v for g, v in gop_sf_scores(lp, ids, grps, blank, spans=sp).items()}
    mar = {g: -v for g, v in margin_scores(lp, ids, grps, blank, spans=sp).items()}
    return {g: dict(current=cur[g], weakest=weak[g], extra=extra.get(g, 0.0),
                    gop=gop[g], margin=mar[g]) for g in cur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", nargs="+")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--first-sura", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pilot import QafModel
    torch.set_num_threads(4)
    ck = torch.load(a.ckpt, map_location="cpu")
    tokens = ck["tokens"]; vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    model = QafModel(len(tokens), None); model.load_state_dict(ck["model"]); model.eval()
    rng = random.Random(0)
    out = []
    for f in a.shards:
        n = 0
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            if n >= a.limit:
                break
            for r in pf.read_row_group(rg, columns=["audio", "phonemes", "start_span", "duration_seconds", "moshaf_id"]).to_pylist():
                if n >= a.limit:
                    break
                if not r["start_span"] or r["start_span"]["sura_idx"] < a.first_sura or r["duration_seconds"] > 20:
                    continue
                toks, grps = fold_with_groups(r["phonemes"])
                if any(t not in vocab for t in toks):
                    continue
                wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                with torch.no_grad():
                    w = torch.from_numpy(wav).unsqueeze(0)
                    lp, hl = model(w, torch.tensor([w.shape[1]]))
                    lp = lp[0, : hl[0]]
                ids = [vocab[t] for t in toks]
                p = perturb(toks, grps, rng, vocab)
                if p is None:
                    continue
                good = all_scores(lp, ids, grps, blank)
                pids = [vocab[t] for t in p[0]]
                bad = all_scores(lp, pids, p[1], blank)
                if good is None or bad is None or p[2]["group"] not in bad:
                    continue
                rec = dict(reciter=r["moshaf_id"], kind=p[2]["kind"], good=list(good.values()),
                           bad=bad[p[2]["group"]])
                # words-and-vowels mode
                blp, bids, bgr, bk = to_basic(lp, tokens, ids, grps)
                _, pbids, pbgr, _ = to_basic(lp, tokens, pids, p[1])
                if pbids != bids:
                    bg = all_scores(blp, bids, bgr, bk)
                    bb = all_scores(blp, pbids, pbgr, bk)
                    if bg is not None and bb is not None and p[2]["group"] in bb:
                        rec["basic_good"] = list(bg.values()); rec["basic_bad"] = bb[p[2]["group"]]
                out.append(rec)
                n += 1
                if n % 20 == 0:
                    print(f, n, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
