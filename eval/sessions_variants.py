"""Default reading vs best of legal reading choices on labelled phone clips judged correct.

  python eval/sessions_variants.py --ckpt models/full/best.pt --sessions data/sessions \
      --thresholds data/eval/variants/baqarah.json --out data/eval/variants/sessions.json

Every word of a correct clip is a correct word; the share above the long-recitation
thresholds (first-half, tajweed judged, each scoring with its own thresholds) is reported.
"""
import argparse, glob, json, os, sys

import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train")); sys.path.insert(0, os.path.join(ROOT, "eval"))
from quran_transcript import quran_phonetizer
from long_recitation import badness
from qafft.phonetic import ayah_uthmani, fold_with_groups
from qafft.variants import VARIANTS, moshaf, best_of
from qafft.wordmap import map_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--sessions", required=True)
    ap.add_argument("--thresholds", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pilot import QafModel
    torch.set_num_threads(4)
    ck = torch.load(a.ckpt, map_location="cpu")
    tokens = ck["tokens"]; vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    model = QafModel(len(tokens), None); model.load_state_dict(ck["model"]); model.eval()
    thr = json.load(open(a.thresholds))["results"]
    rows = []
    for f in sorted(glob.glob(os.path.join(a.sessions, "*.json"))):
        d = json.load(open(f))
        if d.get("clip_kind") != "usable" or d.get("verdict") != "correct":
            continue
        wav, _ = sf.read(f[:-5] + ".wav", dtype="float32")
        with torch.no_grad():
            w = torch.from_numpy(wav).unsqueeze(0)
            lp, hl = model(w, torch.tensor([w.shape[1]])); lp = lp[0, :hl[0]]
        words = d.get("words") or ayah_uthmani(d["sura"], d["aya"]).split()
        res, seen = [], set()
        for name in VARIANTS:
            ph = quran_phonetizer(ayah_uthmani(d["sura"], d["aya"]), moshaf(name), remove_spaces=False).phonemes
            t, g = fold_with_groups(ph)
            if tuple(t) in seen or any(x not in vocab for x in t):
                continue
            seen.add(tuple(t))
            tw = map_tokens(list(words), t, g)
            res.append(badness(lp, [vocab[x] for x in t], tw, blank))
        if not res or res[0] is None:
            continue
        best = best_of(res)
        rows += [dict(clip=d["id"], word=k, default=v, best=best[k], n_readings=len(res)) for k, v in res[0].items()]
    out = dict(words=len(rows), clips=len({r["clip"] for r in rows}))
    for mode in ("default", "best"):
        for lvl in ("1%", "0.1%"):
            t = thr[f"{mode}/{lvl}"]["threshold"]
            n = sum(r[mode] > t for r in rows)
            out[f"{mode}/{lvl}"] = dict(threshold=t, flagged=n, rate=n / len(rows))
            print(f"{mode:<8} {lvl:<5} thr {t:6.3f} flagged {n}/{len(rows)} = {n/len(rows):.2%}")
    json.dump(dict(results=out, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
