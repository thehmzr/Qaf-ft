"""Score prediction files from bench_asr.py.

Baseline checker = "transcribe, compare letters with the expected ayah, flag if
they differ by more than a threshold". Reports, on real learner clips:
  - false-flag rate: share of clips judged correct by annotators that get flagged
  - catch rate: share of clips judged incorrect that get flagged
  - AUC, and catch rate at the threshold that keeps false flags <= 5%
Intervals are 95% bootstrap intervals resampling SPEAKERS (ISLP ch. 5), since
clips from one speaker are not independent.
"""
import json, random, re, sys
from rapidfuzz.distance import Levenshtein

DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")


def norm(s):
    s = re.sub(r"<\|[^|]*\|>", "", s)
    s = s.replace("ٱ", "ا").replace("ٰ", "")
    s = DIAC.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ی", "ي").replace("ک", "ك").replace("ة", "ه")
    s = re.sub(r"[^ء-ي ]", "", s)
    return " ".join(s.split())


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return Levenshtein.distance(r, h) / max(1, len(r))


def auc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def stats(rows):
    ok = [r["cer"] for r in rows if r["label"] == "correct"]
    bad = [r["cer"] for r in rows if r["label"] == "in_correct"]
    out = dict(n_ok=len(ok), n_bad=len(bad), mean_cer_ok=sum(ok) / max(1, len(ok)))
    out["false_flag_any"] = sum(c > 0 for c in ok) / max(1, len(ok))
    if bad:
        out["catch_any"] = sum(c > 0 for c in bad) / len(bad)
        out["auc"] = auc(bad, ok)
        thr = sorted(ok)[min(len(ok) - 1, int(0.95 * len(ok)))]
        out["thr_5pct"] = thr
        out["catch_at_5pct_ff"] = sum(c > thr for c in bad) / len(bad)
    return out


def main(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if not r["hyp"].startswith("__ERROR__")]
    for r in rows:
        r["cer"] = cer(r["ref"], r["hyp"])
    s = stats(rows)
    spk = {}
    for r in rows:
        spk.setdefault(r["spk"], []).append(r)
    keys = list(spk)
    random.seed(0)
    boots = []
    for _ in range(1000):
        samp = [x for k in random.choices(keys, k=len(keys)) for x in spk[k]]
        if any(x["label"] == "correct" for x in samp):
            boots.append(stats(samp))
    def ci(k):
        v = sorted(b[k] for b in boots if k in b)
        return [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v))], 3)] if v else None
    res = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
    res["ci"] = {k: ci(k) for k in ("false_flag_any", "catch_any", "auc", "catch_at_5pct_ff", "mean_cer_ok") if k in s}
    res["speakers"] = len(keys)
    print(path.split("/")[-1], json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
