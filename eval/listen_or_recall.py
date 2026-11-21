"""Does the model listen, or recall the ayah from memory?

  python eval/listen_or_recall.py --ckpt models/full/best.pt [--limit 150] [--out-dir data/eval/listen]

Every ayah's text is in the training data, so the model may write the expected ayah
instead of what it hears. Here the mistake is put in the AUDIO: correct studio clips
are cut at word boundaries (data_tools/splice.py) to delete, repeat or swap one inner
word. The reference text stays the ORIGINAL ayah.

Per reciter and kind:
  recall rate: the free decode follows the text, not the audio
    deletion   - >= 60% of the removed word's words-and-vowels symbols still decoded there
    repetition - the repeated word is not decoded twice
    swap       - the original word is decoded (>= 60%) and fits better than the inserted one
  catch rate: the changed word is flagged at studio-val thresholds (1%, 0.1%), both modes
    (repetition: the repeated word or a neighbour, since the extra-sound charge can land
    on either side); false flags on the untouched words of the same clips
    (also on words two or more away from the change, "far")
  control: the same clips unspliced (false flags)
Cut check: the cut-out word (and the inserted word) is decoded on its own; recall is also
given for "clean" clips where it decodes as that word (>= 60%). Recalled clips are saved
under samples/recalled/ for listening.
Intervals: 95%, resampling clips.

Target words are inner words the model decodes in the unspliced clip (>= 60% matched)
whose first and last sounds do not merge with a neighbour's.
"""
import argparse, glob, io, json, os, random, sys, time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "train"), os.path.join(ROOT, "eval"), os.path.join(ROOT, "data_tools")):
    sys.path.insert(0, p)
from qafft.phonetic import fold_with_groups
from qafft.verify import basic_classes, token_spans
from splice import cuts_from_frames, deletion, repetition, swap, true_tokens
from thresholds import both_modes, load_model

SR = 16000
KINDS = ("deletion", "repetition", "swap")
# unseen-voice reciter first; 19.0 uses a shard the studio-val thresholds did not use
SETS = [("5.0", "data/heldout/moshaf_5.0/*.parquet"),
        ("13.0", "data/heldout/moshaf_13.0/*.parquet"),
        ("19.0", "data/heldout/moshaf_19.0/train-00007-*.parquet")]
HEARD = 0.6


def to_cls(ids, grps, cls):
    """words-and-vowels classes of a token sequence (dropped and repeated classes removed)"""
    out, g_out = [], []
    for i, g in zip(ids, grps):
        c = cls[i]
        if c is None or (out and out[-1] == c):
            continue
        out.append(c); g_out.append(g)
    return out, g_out


def match_frac(ref, ref_g, hyp):
    """share of each word's symbols matched in an edit-distance alignment of hyp to ref"""
    hit = [0] * len(ref)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag == "equal":
            for i in range(i1, i2):
                hit[i] = 1
    tot, got = {}, {}
    for h, g in zip(hit, ref_g):
        tot[g] = tot.get(g, 0) + 1; got[g] = got.get(g, 0) + h
    return {g: got[g] / tot[g] for g in tot}


def decode(lp, blank):
    return [i for i in torch.unique_consecutive(lp.argmax(-1)).tolist() if i != blank]


def ci(xs, n=1000, seed=0):
    xs = np.asarray(xs, dtype=float)
    if not len(xs):
        return [None, None, None]
    rng = np.random.default_rng(seed)
    b = rng.choice(xs, (n, len(xs))).mean(1)
    return [float(xs.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def ci_flags(per_clip, n=1000, seed=0):
    """word false-flag rate with clips resampled: per_clip = [(flags, words)]"""
    if not per_clip:
        return [None, None, None]
    a = np.asarray(per_clip, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    b = a[idx, 0].sum(1) / np.maximum(1, a[idx, 1].sum(1))
    return [float(a[:, 0].sum() / max(1, a[:, 1].sum())), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def rows(pattern, rng):
    """clips of 3-20 s with a known ayah, row groups in random order"""
    for f in sorted(glob.glob(os.path.join(ROOT, pattern))):
        pf = pq.ParquetFile(f)
        order = list(range(pf.num_row_groups)); rng.shuffle(order)
        for rg in order:
            t = pf.read_row_group(rg, columns=["audio", "phonemes", "start_span", "duration_seconds"]).to_pylist()
            rng.shuffle(t)
            for r in t:
                if r["start_span"] and 3 <= r["duration_seconds"] <= 20 and r["phonemes"]:
                    yield os.path.basename(f), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/full/best.pt"))
    ap.add_argument("--limit", type=int, default=150, help="clips per reciter (each gives one clip of every kind)")
    ap.add_argument("--thresholds", default=os.path.join(ROOT, "data/eval/thresholds/studio-val.json"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data/eval/listen"))
    ap.add_argument("--samples", type=int, default=10, help="saved wavs per kind per reciter")
    ap.add_argument("--reciters", nargs="*", default=None)
    a = ap.parse_args()
    torch.set_num_threads(6)
    os.makedirs(os.path.join(a.out_dir, "samples"), exist_ok=True)
    thr = json.load(open(a.thresholds))
    tokens, logp = load_model(a.ckpt)
    vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    cls, _ = basic_classes(tokens)
    rng = random.Random(0)
    clip_log = open(os.path.join(a.out_dir, "clips.jsonl"), "w")
    index = open(os.path.join(a.out_dir, "samples", "index.tsv"), "w")
    index.write("file\tkind\treciter\tsura:aya\tword\tjoins_s\tchanged word\tinserted word\n")
    result = dict(thresholds=a.thresholds, ckpt=a.ckpt, reciters={})

    for label, pattern in SETS:
        if a.reciters and label not in a.reciters:
            continue
        t0 = time.time()
        # pass 1: decode and time the words of each correct clip
        clips = []
        for fname, r in rows(pattern, rng):
            if len(clips) >= a.limit:
                break
            words = r["phonemes"].split()
            toks, grps = fold_with_groups(r["phonemes"])
            if len(words) < 4 or any(t not in vocab for t in toks) or sorted(set(grps)) != list(range(len(words))):
                continue
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if sr != SR or wav.ndim != 1:
                continue
            ids = [vocab[t] for t in toks]
            lp = logp(wav)
            spans = token_spans(lp, ids, blank)
            if spans is None:
                continue
            first, last = {}, {}
            for fr, g in zip(spans, grps):
                if fr:
                    first.setdefault(g, fr[0]); last[g] = fr[-1]
            if len(first) != len(words):
                continue
            cuts = cuts_from_frames([first[g] for g in range(len(words))], [last[g] for g in range(len(words))], len(wav))
            if any(b <= c for c, b in zip(cuts, cuts[1:])):
                continue
            ref, ref_g = to_cls(ids, grps, cls)
            hyp, _ = to_cls(decode(lp, blank), [0] * 10 ** 5, cls)
            heard = match_frac(ref, ref_g, hyp)
            ok = [k for k in range(1, len(words) - 1)
                  if heard.get(k, 0) >= HEARD and words[k - 1][-1] != words[k][0] and words[k][-1] != words[k + 1][0]]
            if not ok:
                continue
            sc = both_modes(lp, tokens, ids, grps, blank)
            if sc is None:
                continue
            ss = r["start_span"]
            clips.append(dict(file=fname, ayah=f"{ss['sura_idx']}:{ss['aya_idx']}", wav=wav, words=words, ids=ids,
                              grps=grps, ref=ref, ref_g=ref_g, cuts=cuts, ok=ok, control=sc,
                              tok_err=Levenshtein.distance(decode(lp, blank), ids) / len(ids)))
        print(f"{label}: {len(clips)} clips timed in {time.time() - t0:.0f}s", flush=True)

        # donor words for swaps: same reciter, same shard
        donors = [(di, k) for di, c in enumerate(clips) for k in c["ok"]]

        def basic_word(w):
            t, g = true_tokens([w])
            return to_cls([vocab[x] for x in t if x in vocab], g, cls)[0]

        res = {kind: [] for kind in KINDS}
        saved = {kind: 0 for kind in KINDS + ("recalled",)}

        def piece_heard(w, word):
            ref = basic_word(word)
            h = decode(logp(w), blank)
            return match_frac(ref, [0] * len(ref), to_cls(h, [0] * len(h), cls)[0]).get(0, 0.0)
        for n, c in enumerate(clips):
            for kind in KINDS:
                k = rng.choice(c["ok"])
                dur = c["cuts"][k + 1] - c["cuts"][k]
                extra = {}
                if kind == "deletion":
                    audio, tw, joins = deletion(c["wav"], c["words"], c["cuts"], k)
                elif kind == "repetition":
                    audio, tw, joins = repetition(c["wav"], c["words"], c["cuts"], k)
                else:
                    bw = basic_word(c["words"][k])
                    cand = []
                    for di, j in donors:
                        d = clips[di]
                        if di == n or d["file"] != c["file"]:
                            continue
                        dd = d["cuts"][j + 1] - d["cuts"][j]
                        if not 0.75 * dur <= dd <= 1.25 * dur:
                            continue
                        dw = d["words"][j]
                        if dw[0] in (c["words"][k - 1][-1],) or dw[-1] == c["words"][k + 1][0]:
                            continue          # keep the inserted word's sounds separate
                        if Levenshtein.normalized_distance(bw, basic_word(dw)) < 0.5:
                            continue          # different sounds, also in words-and-vowels
                        cand.append((di, j))
                    if not cand:
                        continue
                    di, j = rng.choice(cand)
                    d = clips[di]
                    audio, tw, joins = swap(c["wav"], c["words"], c["cuts"], k, d["wav"], d["words"], d["cuts"], j)
                    extra = dict(donor=d["ayah"], inserted=d["words"][j])
                    extra["piece_ins"] = piece_heard(d["wav"][d["cuts"][j]: d["cuts"][j + 1]], d["words"][j])
                t_toks, t_grps = true_tokens(tw)
                if any(t not in vocab for t in t_toks):
                    continue
                t_ids = [vocab[t] for t in t_toks]
                true, true_g = to_cls(t_ids, t_grps, cls)
                lp = logp(audio)
                hyp_ids = decode(lp, blank)
                hyp, _ = to_cls(hyp_ids, [0] * len(hyp_ids), cls)
                m_ref = match_frac(c["ref"], c["ref_g"], hyp)
                m_true = match_frac(true, true_g, hyp)
                if kind == "deletion":
                    recalled = m_ref.get(k, 0) >= HEARD
                    target = [k]
                elif kind == "repetition":
                    recalled = min(m_true.get(k, 0), m_true.get(k + 1, 0)) < HEARD
                    target = [k - 1, k, k + 1]
                else:
                    recalled = m_ref.get(k, 0) >= HEARD and m_ref.get(k, 0) > m_true.get(k, 0)
                    target = [k]
                sc = both_modes(lp, tokens, c["ids"], c["grps"], blank)
                if sc is None:
                    continue
                # does the cut-out word decode as that word on its own? (checks the cut points)
                piece = piece_heard(c["wav"][c["cuts"][k]: c["cuts"][k + 1]], c["words"][k])
                clean = piece >= HEARD and extra.get("piece_ins", 1.0) >= HEARD
                row = dict(reciter=label, kind=kind, ayah=c["ayah"], word=k, recalled=bool(recalled),
                           piece=piece, clean=bool(clean),
                           m_ref=m_ref.get(k, 0), m_true=[m_true.get(x, 0) for x in ([k, k + 1] if kind == "repetition" else [k])],
                           err_vs_true=Levenshtein.distance(hyp_ids, t_ids) / len(t_ids),
                           err_vs_ref=Levenshtein.distance(hyp_ids, c["ids"]) / len(c["ids"]), **extra)
                for mode in ("full", "basic"):
                    s = sc[mode]
                    row[mode] = dict(target=max(s.get(g, 0.0) for g in target),
                                     others=[v for g, v in s.items() if g not in target],
                                     far=[v for g, v in s.items() if abs(g - k) >= 2])
                res[kind].append(row)
                clip_log.write(json.dumps(row, ensure_ascii=False) + "\n")
                if recalled and saved["recalled"] < 30:
                    name = f"recalled/{kind}_{label}_{saved['recalled']:02d}.wav"
                    os.makedirs(os.path.join(a.out_dir, "samples", "recalled"), exist_ok=True)
                    sf.write(os.path.join(a.out_dir, "samples", name), audio, SR)
                    index.write("\t".join([name, kind, label, c["ayah"], str(k + 1),
                                           ",".join(f"{x / SR:.2f}" for x in joins),
                                           c["words"][k], extra.get("inserted", "")]) + "\n")
                    saved["recalled"] += 1
                if saved[kind] < a.samples:
                    name = f"{kind}/{label}_{saved[kind]:02d}.wav"
                    os.makedirs(os.path.join(a.out_dir, "samples", kind), exist_ok=True)
                    sf.write(os.path.join(a.out_dir, "samples", name), audio, SR)
                    index.write("\t".join([name, kind, label, c["ayah"], str(k + 1),
                                           ",".join(f"{x / SR:.2f}" for x in joins),
                                           c["words"][k], extra.get("inserted", "")]) + "\n")
                    saved[kind] += 1
            if (n + 1) % 25 == 0:
                print(f"  {label}: {n + 1}/{len(clips)} clips spliced, {time.time() - t0:.0f}s", flush=True)
            clip_log.flush(); index.flush()

        # summary
        out = dict(clips=len(clips), control={}, kinds={})
        for mode in ("full", "basic"):
            for level in ("flag", "strong"):
                t = thr[mode][level]
                out["control"][f"{mode}_{level}"] = ci_flags(
                    [(sum(v > t for v in c["control"][mode].values()), len(c["control"][mode])) for c in clips])
        out["control"]["token_error"] = float(np.mean([c["tok_err"] for c in clips])) if clips else None
        for kind, rs in res.items():
            o = dict(n=len(rs), recall=ci([r["recalled"] for r in rs]),
                     n_clean=sum(r["clean"] for r in rs), recall_clean=ci([r["recalled"] for r in rs if r["clean"]]),
                     err_vs_true=float(np.mean([r["err_vs_true"] for r in rs])) if rs else None,
                     err_vs_ref=float(np.mean([r["err_vs_ref"] for r in rs])) if rs else None)
            for mode in ("full", "basic"):
                for level in ("flag", "strong"):
                    t = thr[mode][level]
                    o[f"catch_{mode}_{level}"] = ci([r[mode]["target"] > t for r in rs])
                    o[f"ff_{mode}_{level}"] = ci_flags([(sum(v > t for v in r[mode]["others"]), len(r[mode]["others"])) for r in rs])
                    o[f"ff_far_{mode}_{level}"] = ci_flags([(sum(v > t for v in r[mode]["far"]), len(r[mode]["far"])) for r in rs])
            out["kinds"][kind] = o
        result["reciters"][label] = out
        print(json.dumps({label: out}, indent=1), flush=True)
        json.dump(result, open(os.path.join(a.out_dir, "result.json"), "w"), indent=1)
    print("wrote", os.path.join(a.out_dir, "result.json"))


if __name__ == "__main__":
    main()
