"""Checks every ayah clip of a long recitation by decoding it with the model.

  python data_tools/ayah_check.py data/long/NAME/ayahs_v2 [--audio WAV] [--ckpt CKPT] [--write]

For each clip SSS_AAA.wav listed in ayahs.json the decoded sounds (words and vowels only) are
compared with the ayah's own sounds:
  dist       edit distance to the whole ayah, divided by the ayah's length; a larger dist is
             accepted only when the extra sounds are a phrase of the same ayah said again
             ("again") and few of the ayah's own sounds are unheard ("lost")
  starts_ok  the clip begins with the ayah's first sounds
  ends_ok    the clip ends with the ayah's last sounds
  extra      "prev" / "next" if sounds of the neighbouring ayah sit before / after the ayah,
             "start" / "end" for other extra sounds there
  quiet_ok   both ends of the clip lie in a pause (not inside a word)
A clip passes when all of these hold. --write stores the result in ayahs.json as "check".
"""
import argparse, json, os, sys

import numpy as np
import soundfile as sf
import torch
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval")); sys.path.insert(0, os.path.join(ROOT, "train"))
from qafft.verify import basic_classes

SR, HOP, WIN = 16000, 0.01, 0.025
EDGE = 6            # sounds compared at each end
MAX_DIST = 0.2
MAX_LOST = 0.1      # share of the ayah's sounds that may be unheard when a phrase is said again
AMIN = ["ء", "َ", "ا2", "م", "ِ", "ۦ2", "ن"]      # said after sura 1, not part of the text


def frame_db(wav):
    h, w = int(HOP * SR), int(WIN * SR)
    if len(wav) < w:
        return np.zeros(0)
    fr = np.lib.stride_tricks.sliding_window_view(wav, w)[::h]
    return 10 * np.log10((fr ** 2).mean(1) + 1e-10)


def quiet_level(db):
    """Same rule as the pause splitter: a third of the way from the quiet floor to speech."""
    floor, speech = np.percentile(db, 10), np.percentile(db, 90)
    return float(floor + (speech - floor) / 3)


def quiet_runs(quiet, lo, hi, min_len):
    """Runs of quiet frames inside [lo, hi) at least min_len seconds long, as (first, end)."""
    out, i = [], lo
    while i < hi:
        if quiet[i]:
            j = i
            while j < hi and quiet[j]:
                j += 1
            if (j - i) * HOP >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def chunks(quiet, max_len=20.0, pad=2.0):
    """Frame spans to decode one at a time. A clip is kept whole when it can be; a long one
    is cut at its longest pause (short quiet stretches can sit inside a word)."""
    out, stack = [], [(0, len(quiet))]
    while stack:
        a, b = stack.pop()
        if (b - a) * HOP <= max_len:
            out.append((a, b)); continue
        k = int(pad / HOP)
        inner = quiet_runs(quiet, a + k, b - k, 0.05)
        m = sum(max(inner, key=lambda r: r[1] - r[0])) // 2 if inner else (a + b) // 2
        stack += [(m, b), (a, m)]
    return out


class Decoder:
    def __init__(self, ckpt, threads=6):
        from pilot import QafModel
        torch.set_num_threads(threads)
        ck = torch.load(ckpt, map_location="cpu")
        self.tokens = ck["tokens"]; self.blank = len(self.tokens)
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.cls, _ = basic_classes(self.tokens)
        self.model = QafModel(len(self.tokens)).eval()
        self.model.load_state_dict(ck["model"])

    def basic(self, ids):
        out = []
        for t in ids:
            c = self.cls[t]
            if c is not None and (not out or out[-1] != c):
                out.append(c)
        return out

    @torch.no_grad()
    def frames(self, wav):
        """Best token per frame (25 per second)."""
        if len(wav) < 1600:
            return []
        x = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
        lp, hl = self.model(x, torch.tensor([x.shape[1]]))
        return lp[0, : hl[0]].argmax(-1).tolist()

    def sounds(self, wav, level):
        """Decoded words-and-vowels sounds of a clip, pause by pause."""
        quiet = frame_db(wav) < level
        if len(quiet) == 0:
            return []
        ids = []
        for a, b in chunks(quiet):
            am = self.frames(wav[a * 160: min(len(wav), b * 160 + 400)])
            ids += [i for i in torch.unique_consecutive(torch.tensor(am)).tolist() if i != self.blank] if am else []
        return self.basic(ids)

    def amin(self):
        return self.basic([self.vocab[t] for t in AMIN if t in self.vocab])

    def expected(self, key):
        """Sounds of one whole ayah said alone; None if a sound is outside the model's tokens."""
        import long_recitation as L
        toks, _ = L.portion(key, 0, len(L.ayah_words(key)) - 1)
        if any(t not in self.vocab for t in toks):
            return None
        return self.basic([self.vocab[t] for t in toks])


def _lead(hyp, ref):
    """How many decoded sounds come before the ayah's first sounds, and the distance there."""
    head, best = ref[:EDGE], None
    for s in range(0, min(len(hyp), 40) + 1):
        d = min(Levenshtein.distance(hyp[s:s + n], head) for n in range(max(1, len(head) - 2), len(head) + 3))
        if best is None or d < best[0]:
            best = (d, s)
    return best[1], best[0]


def check(hyp, ref, prev=None, nxt=None, quiet_ok=True):
    """hyp: decoded sounds of the clip. ref: the ayah's sounds. prev / nxt: neighbours' sounds."""
    if not ref:
        return dict(dist=None, lost=None, again=False, starts_ok=False, ends_ok=False, extra=None, quiet_ok=bool(quiet_ok), **{"pass": False})
    dist = Levenshtein.distance(hyp, ref) / len(ref)
    # sounds of the ayah not heard, and whether the extra sounds are a phrase of it said again
    text = lambda seq: "".join(chr(0x4E00 + c) for c in seq)
    lost, added, known = 0, 0, 0
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag in ("delete", "replace"):
            lost += i2 - i1
        if tag == "insert" and j2 - j1 >= 4:
            added += j2 - j1
            if fuzz.partial_ratio(text(hyp[j1:j2]), text(ref)) >= 75:
                known += j2 - j1
            elif j2 - j1 > 8:
                added += 10 ** 6                  # a long stretch that is not from this ayah
    again = added > 0 and known >= 0.6 * added
    whole = dist <= MAX_DIST or (again and lost / len(ref) <= MAX_LOST)
    lead, d0 = _lead(hyp, ref)
    trail, d1 = _lead(hyp[::-1], ref[::-1])
    tol = 1 if len(ref) >= EDGE else 0
    starts = lead <= 1 and d0 <= tol
    ends = trail <= 1 and d1 <= tol
    extra = None

    def like(part, other, slack=0):           # the extra sounds are the start of `other`
        return bool(other) and len(part) >= 2 and Levenshtein.distance(part, other[:len(part) + slack]) <= max(slack, len(part) // 3)
    if lead >= 2 and d0 <= tol:
        part = hyp[:lead]
        if like(part[::-1], (prev or [])[::-1]):
            extra = "prev"
        elif like(part, ref, 1):              # the ayah's own first words said twice
            starts = again = True
        else:
            extra = "start"
    if trail >= 2 and d1 <= tol:
        part = hyp[len(hyp) - trail:]
        if like(part, nxt or []):
            extra = "next"
        elif like(part[::-1], ref[::-1], 1):  # the ayah's own last words said twice
            ends = again = True
        else:
            extra = "end"
    ok = starts and ends and extra is None and quiet_ok and whole
    return dict(dist=round(dist, 3), lost=round(lost / len(ref), 3), again=again, starts_ok=bool(starts),
                ends_ok=bool(ends), extra=extra, quiet_ok=bool(quiet_ok), **{"pass": bool(ok)})


def edges_quiet(wav, level, span=0.03):
    """True if the first and last frames of the clip are below the quiet level."""
    db = frame_db(wav)
    k = max(1, int(span / HOP))
    return len(db) > 2 * k and bool((db[:k] < level).all() and (db[-k:] < level).all())


def summary(rows):
    n = len(rows)
    part = lambda f: f"{sum(1 for r in rows if f(r['check']))}/{n}"
    return (f"clips {n}: pass {part(lambda c: c['pass'])}; dist<={MAX_DIST} {part(lambda c: c['dist'] is not None and c['dist'] <= MAX_DIST)} "
            f"(+{sum(1 for r in rows if r['check']['dist'] is not None and r['check']['dist'] > MAX_DIST and r['check']['again'] and r['check']['lost'] <= MAX_LOST)} with a phrase said again); "
            f"starts {part(lambda c: c['starts_ok'])}; ends {part(lambda c: c['ends_ok'])}; "
            f"no extra {part(lambda c: c['extra'] is None)}; quiet ends {part(lambda c: c['quiet_ok'])}")


def reason(c):
    why = []
    if c["dist"] is None or (c["dist"] > MAX_DIST and not (c["again"] and c["lost"] <= MAX_LOST)):
        why.append(f"dist {c['dist']}")
    if not c["starts_ok"]:
        why.append("start")
    if not c["ends_ok"]:
        why.append("end")
    if c["extra"]:
        why.append("extra " + c["extra"])
    if not c["quiet_ok"]:
        why.append("no pause at an end")
    return ", ".join(why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="folder with ayahs.json and the clips")
    ap.add_argument("--audio", default=None, help="whole recording, for the quiet level (default DIR/../audio16k.wav)")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/full/best.pt"))
    ap.add_argument("--write", action="store_true", help="store the result in ayahs.json")
    a = ap.parse_args()
    path = os.path.join(a.dir, "ayahs.json")
    meta = json.load(open(path))
    level = meta.get("quiet_db")
    if level is None:
        whole = sf.read(a.audio or os.path.join(a.dir, "..", "audio16k.wav"), dtype="float32")[0]
        level = quiet_level(frame_db(whole))
    dec = Decoder(a.ckpt)
    rows = meta["ayahs"]
    refs = [dec.amin() if r.get("kind") == "amin" else dec.expected(r["sura"] * 1000 + r["ayah"]) for r in rows]
    for k, r in enumerate(rows):
        wav = sf.read(os.path.join(a.dir, r.get("file") or f"{r['sura']:03d}_{r['ayah']:03d}.wav"), dtype="float32")[0]
        r["check"] = check(dec.sounds(wav, level), refs[k], refs[k - 1] if k else None,
                           refs[k + 1] if k + 1 < len(rows) else None, edges_quiet(wav, level))
    print(summary(rows))
    for r in rows:
        if not r["check"]["pass"]:
            print(f"  {r['sura']}:{r['ayah']}{' amin' if r.get('kind') == 'amin' else ''}  {reason(r['check'])}")
    if a.write:
        json.dump(meta, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
