"""Cuts a long, correct recitation of one sura into pieces and finds which words each piece holds.

  python data_tools/cut_recitation.py AUDIO --sura 2 --out data/long/NAME [--ckpt models/full/best.pt]

Steps: 16 kHz mono copy (the original is left alone); split at pauses; decode every piece
with the model; match each piece to the sura's sound tokens with a moving pointer (on words
and vowels only, so tajweed lengths do not get in the way). Writes in --out:
  audio16k.wav     the converted audio
  pieces.json      every piece: start/end, decoded tokens, matched words or why not, mismatch
  logp/NNNN.npy    the model's frame log-probabilities per piece (for scoring later)
  coverage.txt     share of the sura's words covered once, gaps, repeats
"""
import argparse, json, os, sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
from qafft.phonetic import ayah_uthmani, default_moshaf, fold_with_groups
from qafft.verify import basic_classes


def load_16k(path, out_wav):
    if os.path.exists(out_wav):
        return sf.read(out_wav, dtype="float32")[0]
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.numpy()
    sf.write(out_wav, wav, 16000, subtype="PCM_16")
    return wav


def split_at_pauses(wav, min_sil=0.3, min_piece=1.0, max_piece=25.0, hop=0.01, win=0.025):
    """Cut points in the middle of pauses. The silence level is set from this recording:
    a third of the way from its quiet floor to its speech level, in dB."""
    h, w = int(hop * 16000), int(win * 16000)
    n = 1 + max(0, (len(wav) - w) // h)
    frames = np.lib.stride_tricks.sliding_window_view(wav, w)[::h][:n]
    db = 10 * np.log10((frames ** 2).mean(1) + 1e-10)
    floor, speech = np.percentile(db, 10), np.percentile(db, 90)
    quiet = db < floor + (speech - floor) / 3

    def pauses(lo, hi, min_len):
        out, i = [], lo
        while i < hi:
            if quiet[i]:
                j = i
                while j < hi and quiet[j]:
                    j += 1
                if (j - i) * hop >= min_len:
                    out.append((i, j))
                i = j
            else:
                i += 1
        return out

    cuts = [0] + [(i + j) // 2 for i, j in pauses(0, n, min_sil)] + [n]
    pieces = [[a, b] for a, b in zip(cuts, cuts[1:]) if b > a]
    merged = []                                  # short pieces join the previous one
    for p in pieces:
        if merged and (p[1] - p[0]) * hop < min_piece:
            merged[-1][1] = p[1]
        else:
            merged.append(p)
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) * hop < min_piece:
        merged[1][0] = merged[0][0]; merged.pop(0)
    final = []
    stack = merged[::-1]
    while stack:                                 # long pieces: cut at their longest inner pause
        a, b = stack.pop()
        if (b - a) * hop <= max_piece:
            final.append((a, b)); continue
        inner = pauses(a + int(min_piece / hop), b - int(min_piece / hop), 0.05)
        if inner:
            i, j = max(inner, key=lambda x: x[1] - x[0])
            m = (i + j) // 2
        else:
            m = (a + b) // 2                    # no pause at all: cut in the middle
        stack += [(m, b), (a, m)]
    return [(a * h, min(len(wav), b * h + w)) for a, b in final], dict(floor_db=float(floor), speech_db=float(speech))


def sura_stream(sura, with_basmala=True):
    """The sura's tokens in order, with the (ayah, group) each starts in. Ayah 0 is the basmala."""
    from quran_transcript import Aya
    n = Aya(sura, 1).get().num_ayat_in_sura
    ayahs = ([(0, ayah_uthmani(1, 1))] if with_basmala and sura not in (1, 9) else [])
    ayahs += [(a, ayah_uthmani(sura, a)) for a in range(1, n + 1)]
    from quran_transcript import quran_phonetizer
    toks, words = [], []
    for a, text in ayahs:
        ph = quran_phonetizer(text, default_moshaf(), remove_spaces=False).phonemes
        t, g = fold_with_groups(ph)
        toks += t; words += [(a, x) for x in g]
    return toks, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--sura", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/full/best.pt"))
    ap.add_argument("--accept", type=float, default=0.15, help="largest mismatch to accept a match")
    ap.add_argument("--ahead", type=int, default=150, help="tokens searched beyond the expected position")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logp"), exist_ok=True)

    wav = load_16k(a.audio, os.path.join(a.out, "audio16k.wav"))
    spans, levels = split_at_pauses(wav)
    secs = np.array([(e - s) / 16000 for s, e in spans])
    print(f"{len(wav) / 16000 / 3600:.2f} h, {len(spans)} pieces; length median {np.median(secs):.1f} s, "
          f"10-90% {np.percentile(secs, 10):.1f}-{np.percentile(secs, 90):.1f} s, max {secs.max():.1f} s", flush=True)

    from pilot import QafModel
    torch.set_num_threads(8)
    ck = torch.load(a.ckpt, map_location="cpu")
    tokens = ck["tokens"]; vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    model = QafModel(len(tokens)).eval().to(a.device)
    model.load_state_dict(ck["model"])
    cls, _ = basic_classes(tokens)

    ref_toks, ref_words = sura_stream(a.sura)
    ref_ids = [vocab[t] for t in ref_toks]
    # words-and-vowels view of the reference, and where each of its symbols came from
    ref_b, ref_pos = [], []
    for i, t in enumerate(ref_ids):
        c = cls[t]
        if c is None or (ref_b and ref_b[-1] == c and ref_pos[-1] == i - 1):
            continue
        ref_b.append(c); ref_pos.append(i)
    as_text = lambda seq: "".join(chr(0x4E00 + c) for c in seq)
    ref_s = as_text(ref_b)
    # word boundaries in the reference symbols: index of the first symbol of each word
    word_of = [ref_words[p] for p in ref_pos]
    starts = {i for i in range(len(word_of)) if i == 0 or word_of[i] != word_of[i - 1]}

    def best_span(hyp_s, lo, hi):
        """Closest stretch of the reference (between lo and hi) to the decoded piece."""
        lo, hi = max(0, lo), min(len(ref_s), hi)
        if hi - lo < 1 or not hyp_s:
            return None
        al = fuzz.partial_ratio_alignment(hyp_s, ref_s[lo:hi])
        if al is None:
            return None
        s0, e0 = lo + al.dest_start, lo + al.dest_end
        best = None
        for s in range(max(lo, s0 - 12), min(hi, s0 + 12) + 1):
            for e in range(max(s + 1, e0 - 12), min(hi, e0 + 12) + 1):
                d = Levenshtein.distance(hyp_s, ref_s[s:e]) / len(hyp_s)
                if best is None or d < best[0]:
                    best = (d, s, e)
        return best

    def snap(s, e):
        """Move the ends to word boundaries; returns the moved ends and how far they moved."""
        bs = sorted(starts | {len(word_of)})
        ns = min(bs, key=lambda b: abs(b - s)); ne = min(bs, key=lambda b: abs(b - e))
        return ns, ne, abs(ns - s) + abs(ne - e)

    pieces, ptr, fails = [], 0, 0
    with torch.no_grad():
        for k, (s, e) in enumerate(spans):
            x = torch.from_numpy(wav[s:e]).unsqueeze(0).to(a.device)
            lp, hl = model(x, torch.tensor([x.shape[1]], device=a.device))
            lp = lp[0, : hl[0]].float().cpu()
            np.save(os.path.join(a.out, "logp", f"{k:04d}.npy"), lp.numpy().astype(np.float16))
            hyp = [i for i in torch.unique_consecutive(lp.argmax(-1)).tolist() if i != blank]
            hb = []
            for t in hyp:
                c = cls[t]
                if c is not None and (not hb or hb[-1] != c):
                    hb.append(c)
            p = dict(i=k, start=s / 16000, end=e / 16000, n_hyp=len(hb))
            if len(hb) < 4:
                p.update(status="too short or silent"); pieces.append(p); continue
            hs = as_text(hb)
            m = best_span(hs, ptr - 5, ptr + len(hb) + a.ahead)
            how = "ahead"
            if m is None or m[0] > a.accept:
                back = best_span(hs, ptr - 3 * len(hb) - a.ahead, ptr + 5)
                if back and back[0] <= a.accept:
                    m, how = back, "repeat"
                elif fails >= 2:
                    anywhere = best_span(hs, 0, len(ref_s))
                    if anywhere and anywhere[0] <= a.accept:
                        m, how = anywhere, "re-anchored"
            if m is None or m[0] > a.accept:
                fails += 1
                p.update(status="no match", mismatch=None if m is None else round(m[0], 3)); pieces.append(p); continue
            fails = 0
            d, ms, me = m
            ns, ne, moved = snap(ms, me)
            if ne <= ns:
                p.update(status="no whole word", mismatch=round(d, 3)); pieces.append(p); continue
            first, last = word_of[ns], word_of[ne - 1]
            p.update(status="matched", how=how, mismatch=round(d, 3), snap_moved=moved,
                     words=[list(first), list(last)], ref=[ns, ne])
            if how != "repeat":
                ptr = max(ptr, ne)
            pieces.append(p)
            if k % 50 == 0:
                print(f"piece {k}/{len(spans)} ptr {ptr}/{len(ref_s)}", flush=True)

    # coverage over the sura's words (basmala and ayah numbers included as words)
    all_words = sorted(set(word_of))
    count = {w: 0 for w in all_words}
    for p in pieces:
        if p["status"] == "matched" and p["how"] != "repeat":
            for w in set(word_of[p["ref"][0]: p["ref"][1]]):
                count[w] += 1
    once = sum(1 for w in all_words if w[0] > 0 and count[w] == 1)
    real = [w for w in all_words if w[0] > 0]
    gaps = [w for w in real if count[w] == 0]
    multi = [w for w in real if count[w] > 1]
    status = {}
    for p in pieces:
        status[p["status"]] = status.get(p["status"], 0) + 1
    rep = (f"pieces: {status}\nwords covered exactly once: {once} of {len(real)} ({once / len(real):.1%})\n"
           f"gaps: {len(gaps)} words, e.g. {gaps[:15]}\ncovered more than once: {len(multi)}, e.g. {multi[:15]}\n"
           f"repeats: {sum(1 for p in pieces if p.get('how') == 'repeat')}, "
           f"re-anchored: {sum(1 for p in pieces if p.get('how') == 're-anchored')}\nlevels: {levels}\n")
    print(rep, flush=True)
    open(os.path.join(a.out, "coverage.txt"), "w").write(rep)
    json.dump(dict(sura=a.sura, ckpt=os.path.basename(a.ckpt), pieces=pieces, ref_len=len(ref_s)),
              open(os.path.join(a.out, "pieces.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
