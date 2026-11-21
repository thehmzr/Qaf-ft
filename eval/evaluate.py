"""Scores a trained checkpoint as a mistake checker.

  python eval/evaluate.py --ckpt best.pt --shards FILE.parquet [...] [--limit N]
  python eval/evaluate.py --ckpt best.pt --sessions data/sessions

--shards   : correct expert recitation. Each clip is scored twice: against its true
             text (every word should pass) and against a text with one planted
             mistake (that word should be flagged). Measures the checker with no
             human labels.
--sessions : clips labelled with the labelling tool. Words marked wrong should be
             flagged, the rest should pass.

Prints, with 95% intervals from resampling reciters: how well scores separate right
from wrong words (AUC) and the share of wrong words caught when false flags are held
to 5%, 1% and 0.1% of correct words.
"""
import argparse, glob, io, json, os, random, sys

import pyarrow.parquet as pq
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
from qafft.phonetic import fold_with_groups
from qafft.mistakes import perturb
from qafft.verify import extra_sounds, group_scores, token_scores


def auc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        rank += (i + j + 1) / 2 * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return (rank - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def summary(rows):
    """rows: (badness, is_mistake). Higher badness = more likely a mistake."""
    neg = sorted(b for b, y in rows if not y)
    pos = [b for b, y in rows if y]
    if not neg or not pos:
        return None
    out = dict(auc=auc(pos, neg))
    for name, f in (("catch_at_5pct", 0.05), ("catch_at_1pct", 0.01), ("catch_at_0.1pct", 0.001)):
        thr = neg[min(len(neg) - 1, int((1 - f) * len(neg)))]
        out[name] = sum(b > thr for b in pos) / len(pos)
    return out


def report(rows_by_reciter, title):
    rows = [r for v in rows_by_reciter.values() for r in v]
    s = summary(rows)
    if not s:
        print(title, ": not enough data"); return None
    keys, rng, boots = list(rows_by_reciter), random.Random(0), []
    for _ in range(300):
        b = summary([r for k in rng.choices(keys, k=len(keys)) for r in rows_by_reciter[k]])
        if b:
            boots.append(b)
    n_pos = sum(y for _, y in rows)
    print(f"{title}: {len(rows) - n_pos} correct words, {n_pos} wrong words, {len(keys)} reciters")
    for k, v in s.items():
        vs = sorted(b[k] for b in boots)
        lo, hi = (vs[int(.025 * len(vs))], vs[int(.975 * len(vs)) - 1]) if len(keys) > 1 and vs else (float("nan"),) * 2
        print(f"  {k:<16} {v:.3f}  ({lo:.3f} to {hi:.3f})")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--sessions", default=None)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--first-sura", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", action="store_true",
                    help="score every word against legal reading choices too and keep its best score")
    a = ap.parse_args()
    from pilot import QafModel
    torch.set_num_threads(4)
    ck = torch.load(a.ckpt, map_location="cpu")
    tokens = ck["tokens"]; vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    model = QafModel(len(tokens), a.base); model.load_state_dict(ck["model"]); model.eval()
    rng = random.Random(0)

    @torch.no_grad()
    def logp(wav):
        w = torch.from_numpy(wav).unsqueeze(0)
        lp, hl = model(w, torch.tensor([w.shape[1]]))
        return lp[0, : hl[0]]

    def score(lp, toks, grps):
        ids = [vocab[t] for t in toks]
        ts = token_scores(lp, ids, blank)
        if ts is None:
            return None
        # badness per word: the weakest expected sound, or a clear extra sound
        bad = {g: -v for g, v in group_scores(ts, grps).items()}
        for g, v in extra_sounds(lp, ids, grps, blank).items():
            bad[g] = max(bad.get(g, 0.0), v)
        return bad

    def readings(text):
        """token lists of the clip text under the default reading and each distinct variant"""
        from quran_transcript import quran_phonetizer
        from qafft.variants import VARIANTS, moshaf
        out, seen = [], set()
        for name in VARIANTS:
            try:
                t, _ = fold_with_groups(quran_phonetizer(text, moshaf(name), remove_spaces=False).phonemes)
            except Exception:
                continue
            if tuple(t) not in seen:
                seen.add(tuple(t)); out.append(t)
        return out

    def score_v(lp, toks, grps, reads):
        """best score per word over the clip's own reading and the variant edits applied to it"""
        if not reads:
            return score(lp, toks, grps)
        from qafft.variants import best_of, transfer
        alts = [(toks, grps)] + [transfer(toks, grps, reads[0], r) for r in reads[1:]]
        alts = [x for x in alts if all(t in vocab for t in x[0])]
        seen, res = set(), []
        for t, g in alts:
            if tuple(t) not in seen:
                seen.add(tuple(t)); res.append(score(lp, t, g))
        return None if res[0] is None else best_of(res)

    by, kinds, n = {}, {}, 0
    for f in a.shards:
        n = 0                                  # the limit applies to each file
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=["audio", "phonemes", "uthmani", "start_span", "duration_seconds", "moshaf_id"]).to_pylist():
                if n >= a.limit:
                    break
                if not r["start_span"] or r["start_span"]["sura_idx"] < a.first_sura or r["duration_seconds"] > 20:
                    continue
                toks, grps = fold_with_groups(r["phonemes"])
                if any(t not in vocab for t in toks):
                    continue
                wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                lp = logp(wav)
                reads = readings(r["uthmani"]) if a.variants else []
                good = score_v(lp, toks, grps, reads)
                p = perturb(toks, grps, rng, vocab)
                if good is None or p is None:
                    continue
                bad = score_v(lp, p[0], p[1], reads)
                if bad is None:
                    continue
                rows = by.setdefault(r["moshaf_id"], [])
                rows += [(v, 0) for v in good.values()]
                g = p[2]["group"]
                rows.append((bad.get(g, max(bad.values())), 1))
                kinds.setdefault(p[2]["kind"], []).append(bad.get(g, max(bad.values())))
                n += 1
    if a.shards:
        s = report(by, "planted mistakes on correct recitation")
        neg = sorted(b for v in by.values() for b, y in v if not y)
        thr = neg[int(0.99 * len(neg))]
        print("  caught at 1% false flags, by kind of mistake:")
        for k, v in sorted(kinds.items()):
            print(f"    {k:<10} {sum(x > thr for x in v) / len(v):.2f}  (n={len(v)})")
        if a.out:
            json.dump(dict(summary=s), open(a.out, "w"), indent=1)

    if a.sessions:
        by = {}
        for f in sorted(glob.glob(os.path.join(a.sessions, "*.json"))):
            d = json.load(open(f))
            if d.get("clip_kind", "usable") != "usable" or d.get("repeat_of"):
                continue
            from qafft.wordmap import word_tokens
            # marks are display-word indices: score per display word, not per spoken group
            toks, tw, _ = word_tokens(d["sura"], d["aya"], d.get("words"))
            wav, sr = sf.read(f[:-5] + ".wav", dtype="float32")
            sc = score(logp(wav), toks, tw)
            if sc is None:
                continue
            wrong = {m["word"] for m in d["marks"]}
            by.setdefault(d.get("reciter") or "unknown", []).extend((v, int(g in wrong)) for g, v in sc.items())
        report(by, "labelled clips")


if __name__ == "__main__":
    main()
