"""False flags on a long correct recitation in a new voice, cut by data_tools/cut_recitation.py.

  python eval/long_recitation.py data/long/NAME [--thresholds FILE.json] [--out FILE.json]

Each piece is scored against its own words phonetized as said alone, so the word before a
pause gets its pause form and the first word its starting form (the joined form of the whole
sura would wrongly expect a final vowel the reciter drops). The word span found by the matcher
is checked up to two words either side and the candidate closest to the decoded sounds is kept.

Every matched piece is scored word by word in both modes (tajweed judged; words and vowels
only). Pieces are split by time: the first half sets thresholds, the second half is the test,
never both. Thresholds from another named set (for example studio-val, written by
eval/thresholds.py) can be given too and are reported beside them.

The recitation is taken as correct, so every flag on it is counted as a false flag. Intervals
come from resampling pieces; there is one voice, so they show spread within that voice only.
"""
import argparse, collections, json, os, random, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "data_tools"))
from functools import lru_cache
from rapidfuzz.distance import Levenshtein
from quran_transcript import quran_phonetizer
from qafft.phonetic import ayah_uthmani, default_moshaf, fold_with_groups
from qafft.verify import basic_classes, extra_sounds, group_scores, to_basic, token_scores
from qafft.wordmap import group_words, map_tokens
from cut_recitation import sura_stream


def ayah_text(sura, ayah):
    return ayah_uthmani(1, 1) if ayah == 0 else ayah_uthmani(sura, ayah)


@lru_cache(maxsize=None)
def ayah_words(sura, ayah):
    return tuple(ayah_text(sura, ayah).split())


@lru_cache(maxsize=None)
def group_to_words(sura, ayah):
    s, a = (1, 1) if ayah == 0 else (sura, ayah)
    return group_words(s, a, words=ayah_words(sura, ayah))


@lru_cache(maxsize=None)
def portion(sura, ayah, w0, w1):
    """Tokens of printed words w0..w1 of one ayah, said on their own (pause rules at both
    ends), and the printed word (within the ayah) of each token."""
    words = ayah_words(sura, ayah)
    if not 0 <= w0 <= w1 < len(words):
        return None
    part = list(words[w0:w1 + 1])
    ph = quran_phonetizer(" ".join(part), default_moshaf(), remove_spaces=False).phonemes
    toks, grps = fold_with_groups(ph)
    tw = map_tokens(part, toks, grps)
    return toks, [w0 + x for x in tw]


def badness(lp, ids, grps, blank):
    ts = token_scores(lp, ids, blank)
    if ts is None:
        return None
    bad = {g: -v for g, v in group_scores(ts, grps).items()}
    for g, v in extra_sounds(lp, ids, grps, blank).items():
        bad[g] = max(bad.get(g, 0.0), v)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--thresholds", default=None, help="json with per-mode thresholds from another set")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.dir, "pieces.json")))
    tokens = meta.get("tokens") or json.load(open(os.path.join(ROOT, "qafft", "vocab.json")))["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    ref_toks, ref_words = sura_stream(meta["sura"])
    first_at, last_at = {}, {}
    for i, w in enumerate(ref_words):
        first_at.setdefault(w, i); last_at[w] = i

    # every printed word of the sura in order, as (ayah, word); spans move in these units
    ayahs = sorted({w[0] for w in ref_words})
    order = [(ay, k) for ay in ayahs for k in range(len(ayah_words(meta["sura"], ay)))]
    at = {w: i for i, w in enumerate(order)}

    def to_word(ay, g, side):
        gw = group_to_words(meta["sura"], ay)
        ws = gw[g] if g < len(gw) and gw[g] else {0}
        return (ay, min(ws) if side == 0 else max(ws))
    cls, _ = basic_classes(tokens)

    def expected(i0, i1):
        """tokens and word ids for words order[i0..i1], phonetized per ayah portion"""
        toks, wids, k = [], [], i0
        while k <= i1:
            ay = order[k][0]; j = k
            while j + 1 <= i1 and order[j + 1][0] == ay:
                j += 1
            r = portion(meta["sura"], ay, order[k][1], order[j][1])
            if r is None:
                return None
            toks += r[0]; wids += [(ay, w) for w in r[1]]
            k = j + 1
        return toks, wids

    def basic_seq(ids):
        out = []
        for t in ids:
            c = cls[t]
            if c is not None and (not out or out[-1] != c):
                out.append(c)
        return out

    end = max(p["end"] for p in meta["pieces"])
    words, n_short, n_split, moved = [], 0, 0, 0   # one row per scored word
    oov = collections.Counter()                  # tokens the model has no output for
    for p in meta["pieces"]:
        if p["status"] != "matched" or p["how"] == "repeat":
            continue
        lp = torch.from_numpy(np.load(os.path.join(a.dir, "logp", f"{p['i']:04d}.npy")).astype(np.float32))
        hyp = basic_seq([i for i in torch.unique_consecutive(lp.argmax(-1)).tolist() if i != blank])
        i0 = at[to_word(*p["words"][0], 0)]
        i1 = at[to_word(*p["words"][1], 1)]
        best = None
        shifts = sorted(((d0, d1) for d0 in range(-2, 3) for d1 in range(-2, 3)), key=lambda d: abs(d[0]) + abs(d[1]))
        for d0, d1 in shifts:                  # smallest shift first, so it wins ties
            if True:
                a0, a1 = i0 + d0, i1 + d1
                if a0 < 0 or a1 >= len(order) or a1 < a0:
                    continue
                e = expected(a0, a1)
                if e is None:
                    continue
                if any(t not in vocab for t in e[0]):
                    oov.update(t for t in e[0] if t not in vocab); continue
                dist = Levenshtein.distance(hyp, basic_seq([vocab[t] for t in e[0]])) / max(1, len(hyp))
                if best is None or dist < best[0] - 1e-9:
                    best = (dist, e, (d0, d1))
        if best is None:
            n_split += 1; continue
        moved += best[2] != (0, 0)
        toks, wids = best[1]
        local = {w: j for j, w in enumerate(dict.fromkeys(wids))}
        ids, grps = [vocab[t] for t in toks], [local[w] for w in wids]
        full = badness(lp, ids, grps, blank)
        blp, bids, bgrps, k = to_basic(lp, tokens, ids, grps)
        basic = badness(blp, bids, bgrps, k)
        if full is None or basic is None:
            n_short += 1; continue
        half = "thr" if p["start"] < end / 2 else "test"
        inv = {j: w for w, j in local.items()}
        for j in local.values():
            words.append(dict(piece=p["i"], half=half, ayah=inv[j][0], group=inv[j][1],
                              full=full.get(j, 0.0), basic=basic.get(j, 0.0)))

    seen_w, real = set(), []
    for w in words:                                      # one row per printed word; basmala left out
        if w["ayah"] > 0 and (w["ayah"], w["group"]) not in seen_w:
            seen_w.add((w["ayah"], w["group"])); real.append(w)
    thr_rows = [w for w in real if w["half"] == "thr"]
    test_rows = [w for w in real if w["half"] == "test"]
    print(f"scored words: {len(real)} ({len(thr_rows)} threshold half, {len(test_rows)} test half); "
          f"pieces too short to align: {n_short}; skipped (no word span): {n_split}; "
          f"word span moved after checking: {moved}; candidate spans with tokens outside the vocabulary: {dict(oov)}")
    if len(thr_rows) < 100 or len(test_rows) < 100:
        raise SystemExit(f"too few scored words to split ({len(thr_rows)} / {len(test_rows)})")
    sets = {}
    for mode in ("full", "basic"):
        v = np.array([w[mode] for w in thr_rows])
        sets[f"own-half/{mode}"] = dict(flag=float(np.quantile(v, 0.99)), strong=float(np.quantile(v, 0.999)))
    if a.thresholds:
        other = json.load(open(a.thresholds))
        for mode in ("full", "basic"):
            sets[f"{other.get('name', 'other')}/{mode}"] = other[mode]

    rng = random.Random(0)
    pieces = sorted({w["piece"] for w in test_rows})
    by_piece = {p: [w for w in test_rows if w["piece"] == p] for p in pieces}
    # an ayah counts only if all its printed words were scored in the test half
    n_groups = {ay: len(ayah_words(meta["sura"], ay)) for ay in {w[0] for w in ref_words}}
    seen = {}
    for w in test_rows:
        seen.setdefault(w["ayah"], set()).add(w["group"])
    full_ayahs = [ay for ay, gs in seen.items() if len(gs) == n_groups[ay]]

    results = {}
    for name, t in sets.items():
        mode = name.split("/")[1]
        for level in ("flag", "strong"):
            flags = [w[mode] > t[level] for w in test_rows]
            rate = float(np.mean(flags))
            boots = []
            for _ in range(1000):
                s = [w for p in rng.choices(pieces, k=len(pieces)) for w in by_piece[p]]
                boots.append(np.mean([w[mode] > t[level] for w in s]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ay_flag = np.mean([any(w[mode] > t[level] for w in test_rows if w["ayah"] == ay) for ay in full_ayahs]) if full_ayahs else float("nan")
            n_fl = int(sum(flags))
            upper = 3 / len(test_rows) if n_fl == 0 else None
            results[f"{name}/{level}"] = dict(threshold=round(t[level], 3), words=len(test_rows), flagged=n_fl,
                                              rate=rate, ci=[float(lo), float(hi)], upper_if_zero=upper,
                                              ayahs=len(full_ayahs), ayah_rate=float(ay_flag))
            print(f"{name:<22} {level:<6} thr {t[level]:6.3f}  {n_fl:4d}/{len(test_rows)} words = {rate:.2%} "
                  f"(95% {lo:.2%}-{hi:.2%})  ayahs with a flag: {ay_flag:.1%} of {len(full_ayahs)}")
    out = a.out or os.path.join(a.dir, "false_flags.json")
    json.dump(dict(results=results, thresholds=sets, n_words=len(real)), open(out, "w"), indent=1)
    json.dump(words, open(os.path.join(a.dir, "word_scores.json"), "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
