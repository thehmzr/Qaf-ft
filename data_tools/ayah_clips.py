"""Cuts a long recitation, already split and scored, into one clip per ayah for listening.

  python data_tools/ayah_clips.py data/long/NAME [--pad 0.15]

Needs pieces.json, logp/ and word_scores.json from data_tools/cut_recitation.py and
eval/long_recitation.py. Each piece's words are aligned to its audio to get word times; an
ayah runs from its first word's start to its last word's end. Writes in DIR/ayahs/:
  NNN.wav       one clip per ayah (000 is the opening)
  ayahs.json    per ayah: text words, clip times, word times within the clip, word scores
"""
import argparse, collections, json, os, sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval")); sys.path.insert(0, os.path.join(ROOT, "data_tools"))
import long_recitation as L

FPS = 25.0                                  # model output frames per second


def word_times(lp, ids, word_of, blank):
    """(start, end) in seconds within the piece for each word, from forced alignment."""
    al, _ = torchaudio.functional.forced_align(lp.unsqueeze(0), torch.tensor([ids]), blank=blank)
    al = al[0].tolist()
    span, pos, prev = {}, -1, blank
    for t, x in enumerate(al):
        if x != blank and (x != prev or (t and al[t - 1] == blank)):
            pos += 1
        if x != blank and 0 <= pos < len(ids):
            w = word_of[pos]
            s, e = span.get(w, (t, t))
            span[w] = (min(s, t), max(e, t + 1))
        prev = x
    return {w: (s / FPS, e / FPS) for w, (s, e) in span.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--pad", type=float, default=0.15)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.dir, "pieces.json")))
    sura = meta["sura"]
    tokens = meta.get("tokens") or json.load(open(os.path.join(ROOT, "qafft", "vocab.json")))["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    pieces = {p["i"]: p for p in meta["pieces"]}
    scores = json.load(open(os.path.join(a.dir, "word_scores.json")))
    by_piece = collections.defaultdict(list)
    for w in scores:
        by_piece[w["piece"]].append(w)
    ff = json.load(open(os.path.join(a.dir, "false_flags.json")))
    thr = ff.get("thresholds", {})

    times, score_of = {}, {}                # (ayah, word) -> absolute (start, end); -> scores
    for pid, ws in sorted(by_piece.items()):
        p = pieces[pid]
        words = sorted({(w["ayah"], w["group"]) for w in ws})
        toks, wids, k = [], [], 0
        while k < len(words):                 # same per-ayah phonetization the scores used
            ay = words[k][0]; j = k
            while j + 1 < len(words) and words[j + 1][0] == ay:
                j += 1
            r = L.portion(sura, ay, words[k][1], words[j][1])
            toks += r[0]; wids += [(ay, x) for x in r[1]]
            k = j + 1
        if any(t not in vocab for t in toks):
            continue
        lp = torch.from_numpy(np.load(os.path.join(a.dir, "logp", f"{pid:04d}.npy")).astype(np.float32))
        if lp.shape[0] < len(toks) + 1:
            continue
        for w, (s, e) in word_times(lp, [vocab[t] for t in toks], wids, blank).items():
            if w not in times:                # first time a word is heard wins (repeats come later)
                times[w] = (p["start"] + s, p["start"] + e)
        for w in ws:
            score_of.setdefault((w["ayah"], w["group"]), dict(full=round(w["full"], 2), basic=round(w["basic"], 2)))

    wav, sr = sf.read(os.path.join(a.dir, "audio16k.wav"), dtype="float32")
    out = os.path.join(a.dir, "ayahs"); os.makedirs(out, exist_ok=True)
    ayahs = sorted({ay for ay, _ in times})
    rows = []
    for ay in ayahs:
        text = list(L.ayah_words(sura, ay))
        have = {k: v for (x, k), v in times.items() if x == ay}
        if not have:
            continue
        s0 = max(0.0, min(v[0] for v in have.values()) - a.pad)
        e0 = min(len(wav) / sr, max(v[1] for v in have.values()) + a.pad)
        clip = wav[int(s0 * sr): int(e0 * sr)]
        sf.write(os.path.join(out, f"{ay:03d}.wav"), clip, sr, subtype="PCM_16")
        rows.append(dict(ayah=ay, start=round(s0, 2), end=round(e0, 2), words=[
            dict(text=t, start=round(have[k][0] - s0, 2) if k in have else None,
                 end=round(have[k][1] - s0, 2) if k in have else None, **score_of.get((ay, k), {}))
            for k, t in enumerate(text)], missing=len(text) - len(have)))
    json.dump(dict(sura=sura, thresholds=thr, ayahs=rows), open(os.path.join(out, "ayahs.json"), "w"), ensure_ascii=False)
    full = sum(1 for r in rows if r["missing"] == 0)
    print(f"{len(rows)} ayah clips ({full} with every word timed), {sum(r['end'] - r['start'] for r in rows) / 60:.1f} min")


if __name__ == "__main__":
    main()
