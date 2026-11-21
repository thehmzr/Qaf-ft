"""Summarises eval/wordscores.py output: AUC, catch at 1% / 0.1% false flags (thresholds
taken from the same negatives, so in-sample), and catch at 1% by kind of mistake.

  python eval/wordscores_report.py data/eval/wordscores/raw_*.json
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.evaluate import auc
from qafft.verify import rank_badness

BASE = ["current", "weakest", "extra", "gop", "margin"]
COMBOS = {"max(current,gop)": ["current", "gop"], "max(current,margin)": ["current", "margin"],
          "max(gop,extra)": ["gop", "extra"], "max(weakest,gop,extra)": ["weakest", "gop", "extra"],
          "max(current,gop,margin)": ["current", "gop", "margin"]}


def table(recs, gk, bk):
    recs = [r for r in recs if bk in r]
    neg = {s: [g[s] for r in recs for g in r[gk]] for s in BASE}
    pos = {s: [r[bk][s] for r in recs] for s in BASE}
    kinds = [r["kind"] for r in recs]
    for name, parts in COMBOS.items():
        rn = {s: rank_badness(neg[s], neg[s]) for s in parts}
        rp = {s: rank_badness(pos[s], neg[s]) for s in parts}
        neg[name] = [max(v) for v in zip(*rn.values())]
        pos[name] = [max(v) for v in zip(*rp.values())]
    ks = sorted(set(kinds))
    print(f"{len(neg['current'])} correct words, {len(recs)} wrong words")
    print(f"{'score':<26}{'AUC':>6}{'@1%':>7}{'@0.1%':>7}  " + " ".join(f"{k[:8]:>8}" for k in ks))
    res = {}
    for s in neg:
        ns = sorted(neg[s])
        t1 = ns[min(len(ns) - 1, int(0.99 * len(ns)))]
        t01 = ns[min(len(ns) - 1, int(0.999 * len(ns)))]
        c1 = sum(v > t1 for v in pos[s]) / len(pos[s])
        c01 = sum(v > t01 for v in pos[s]) / len(pos[s])
        bykind = {k: sum(v > t1 for v, kk in zip(pos[s], kinds) if kk == k) / kinds.count(k) for k in ks}
        res[s] = dict(auc=auc(pos[s], neg[s]), c1=c1, c01=c01, kind=bykind)
        print(f"{s:<26}{res[s]['auc']:>6.3f}{c1:>7.3f}{c01:>7.3f}  " + " ".join(f"{bykind[k]:>8.2f}" for k in ks))
    print("n by kind:", {k: kinds.count(k) for k in ks})
    return res


if __name__ == "__main__":
    recs = [r for f in sys.argv[1:] for r in json.load(open(f))]
    print("== full tokens"); table(recs, "good", "bad")
    print("== words-and-vowels (to_basic)"); table(recs, "basic_good", "basic_bad")
