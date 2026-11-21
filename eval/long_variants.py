"""Default reading vs best of legal reading choices on a long correct recitation.

  python eval/long_variants.py data/long/NAME --out data/eval/variants/NAME.json

Same pieces, word spans, halves and thresholds as eval/long_recitation.py (tajweed judged
mode). Each piece is scored against every distinct variant reading of its words
(qafft/variants.py); a word keeps its lowest badness. Thresholds are set on the first half
of pieces with the same scoring, then applied to the second half.
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "data_tools")); sys.path.insert(0, os.path.join(ROOT, "eval"))
from rapidfuzz.distance import Levenshtein
from long_recitation import ayah_words, group_to_words, portion, badness
from cut_recitation import sura_stream
from qafft.variants import span_variants, VARIANTS


def main():
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.dir, "pieces.json")))
    tokens = json.load(open(os.path.join(ROOT, "qafft", "vocab.json")))["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    S = meta["sura"]
    _, ref_words = sura_stream(S)
    ayahs = sorted({w[0] for w in ref_words})
    order = [(ay, k) for ay in ayahs for k in range(len(ayah_words(S, ay)))]
    at = {w: i for i, w in enumerate(order)}

    def to_word(ay, g, side):
        gw = group_to_words(S, ay)
        ws = gw[g] if g < len(gw) and gw[g] else {0}
        return (ay, min(ws) if side == 0 else max(ws))

    def runs(i0, i1):
        k, out = i0, []
        while k <= i1:
            ay = order[k][0]; j = k
            while j + 1 <= i1 and order[j + 1][0] == ay:
                j += 1
            out.append((ay, order[k][1], order[j][1])); k = j + 1
        return out

    def expected(i0, i1, name=None):
        toks, wids = [], []
        for ay, w0, w1 in runs(i0, i1):
            if name is None:
                r = portion(S, ay, w0, w1)
                if r is None:
                    return None
                t, w = r
            else:
                words = ayah_words(S, ay)
                v = {n: (tt, ww) for n, tt, ww in span_variants(tuple(words[w0:w1 + 1]))}
                t, w = v.get(name, v["default"]); w = [w0 + x for x in w]
            toks += t; wids += [(ay, x) for x in w]
        return toks, wids

    from qafft.verify import basic_classes
    cls, _ = basic_classes(tokens)

    def basic_seq(ids):
        out = []
        for t in ids:
            c = cls[t]
            if c is not None and (not out or out[-1] != c):
                out.append(c)
        return out

    end = max(p["end"] for p in meta["pieces"])
    rows, used = [], {}
    for p in meta["pieces"]:
        if p["status"] != "matched" or p["how"] == "repeat":
            continue
        lp = torch.from_numpy(np.load(os.path.join(a.dir, "logp", f"{p['i']:04d}.npy")).astype(np.float32))
        hyp = basic_seq([i for i in torch.unique_consecutive(lp.argmax(-1)).tolist() if i != blank])
        i0 = at[to_word(*p["words"][0], 0)]; i1 = at[to_word(*p["words"][1], 1)]
        best = None
        for d0 in (0, -1, 1, -2, 2):
            for d1 in (0, -1, 1, -2, 2):
                b0, b1 = i0 + d0, i1 + d1
                if b0 < 0 or b1 >= len(order) or b1 < b0:
                    continue
                e = expected(b0, b1)
                if e is None or any(t not in vocab for t in e[0]):
                    continue
                dist = Levenshtein.distance(hyp, basic_seq([vocab[t] for t in e[0]])) / max(1, len(hyp))
                if best is None or dist < best[0] - 1e-9:
                    best = (dist, (b0, b1))
        if best is None:
            continue
        b0, b1 = best[1]
        per = {}
        seqs = set()
        for name in VARIANTS:
            toks, wids = expected(b0, b1, name)
            if tuple(toks) in seqs or any(t not in vocab for t in toks):
                continue
            seqs.add(tuple(toks))
            local = {w: j for j, w in enumerate(dict.fromkeys(wids))}
            bad = badness(lp, [vocab[t] for t in toks], [local[w] for w in wids], blank)
            if bad is None:
                if name == "default":
                    break
                continue
            per[name] = {w: bad.get(j, 0.0) for w, j in local.items()}
        if "default" not in per:
            continue
        half = "thr" if p["start"] < end / 2 else "test"
        for w, v in per["default"].items():
            if w[0] == 0:
                continue
            others = {n: s[w] for n, s in per.items() if w in s}
            bn = min(others, key=others.get)
            used[bn] = used.get(bn, 0) + 1
            rows.append(dict(piece=p["i"], half=half, ayah=w[0], word=w[1], default=v, best=others[bn], by=bn))
    thr = [r for r in rows if r["half"] == "thr"]; test = [r for r in rows if r["half"] == "test"]
    res = dict(words=len(rows), test_words=len(test), best_variant_counts=used)
    for mode in ("default", "best"):
        v = np.array([r[mode] for r in thr])
        for lvl, q in (("1%", 0.99), ("0.1%", 0.999)):
            t = float(np.quantile(v, q))
            n = sum(r[mode] > t for r in test)
            # also with the default-reading thresholds, to see the pure effect of best-of
            td = float(np.quantile([r["default"] for r in thr], q))
            nd = sum(r[mode] > td for r in test)
            res[f"{mode}/{lvl}"] = dict(threshold=t, flagged=int(n), rate=n / len(test),
                                        flagged_at_default_thr=int(nd), rate_at_default_thr=nd / len(test))
            print(f"{mode:<8} {lvl:<5} thr {t:6.3f} flagged {n}/{len(test)} = {n/len(test):.2%} | "
                  f"at default thr {td:6.3f}: {nd} = {nd/len(test):.2%}")
    print("variant giving the lowest badness:", used)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(results=res, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
