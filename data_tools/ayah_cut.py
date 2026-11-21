"""Cuts a long recitation into one clip per ayah, at the pauses between ayahs, and checks each clip.

  python data_tools/ayah_cut.py data/long/NAME [--out DIR] [--audio WAV] [--ckpt CKPT]

Needs pieces.json, logp/ and word_scores.json (data_tools/cut_recitation.py, eval/long_recitation.py).

How: the sounds decoded from all pieces are lined up with the text of the suras in one pass, so
every ayah gets the place of its first and last sound even where a piece was never matched. The
cut between two ayahs is then the middle of the pause between them (the pause the pieces were
split at, or the longest quiet stretch if both ayahs sit in one piece). Clips touch: an ayah
ends exactly where the next one starts. An ayah with too few sounds found is left out (for
example a basmala that was not recited). "آمين" said after sura 1 becomes its own segment.

Every clip is decoded again and checked (data_tools/ayah_check.py). Where an end fails, the
pauses around the cut are tried and the best one kept.

Writes in --out (default DIR/ayahs_v2):
  SSS_AAA.wav   one clip per ayah (SSS_AAA_amin.wav for آمين)
  ayahs.json    per ayah: start/end in the recording, words with times and scores, the check
"""
import argparse, collections, json, os, sys

import numpy as np
import soundfile as sf
import torch
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "eval")); sys.path.insert(0, os.path.join(ROOT, "data_tools"))
import long_recitation as L
import ayah_check as C
from ayah_clips import word_times
from cut_recitation import sura_stream

FPS, SR = 25.0, 16000


def heard(folder, pieces, dec):
    """Decoded sounds of every piece in order: sound, time in the recording, piece."""
    sym, when, piece = [], [], []
    for p in pieces:
        am = np.load(os.path.join(folder, "logp", f"{p['i']:04d}.npy")).argmax(-1)
        prev, last = dec.blank, None
        for t, x in enumerate(am.tolist()):
            if x != dec.blank and x != prev:
                c = dec.cls[x]
                if c is not None and c != last:
                    sym.append(c); when.append(p["start"] + t / FPS); piece.append(p["i"])
                if c is not None:
                    last = c
            prev = x
    return sym, when, piece


def line_up(ref, hyp):
    """For each expected sound the index of the decoded sound it pairs with (None if lost),
    and whether the pair is equal."""
    to, same, i, j = [None] * len(ref), [False] * len(ref), 0, 0
    for op in list(Levenshtein.editops(ref, hyp)) + [None]:
        si, dj = (len(ref), len(hyp)) if op is None else (op.src_pos, op.dest_pos)
        while i < si and j < dj:
            to[i], same[i] = j, True; i += 1; j += 1
        if op is None:
            break
        if op.tag == "insert":
            j += 1
        elif op.tag == "delete":
            i += 1
        else:
            to[i] = j; i += 1; j += 1
    return to, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/full/best.pt"))
    ap.add_argument("--thresholds", default=None, help="ayahs.json or false_flags.json to copy thresholds from")
    ap.add_argument("--rounds", type=int, default=3, help="passes that try other pauses for failing cuts")
    ap.add_argument("--min-found", type=float, default=0.5, help="share of an ayah's sounds that must be found")
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "ayahs_v2"); os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(a.dir, "pieces.json")))
    pieces = meta["pieces"]; start_of = {p["i"]: p["start"] for p in pieces}
    suras = meta.get("suras") or [meta.get("sura")]
    wav = sf.read(a.audio or os.path.join(a.dir, "audio16k.wav"), dtype="float32")[0]
    total = len(wav) / SR
    db = C.frame_db(wav); level = C.quiet_level(db); quiet = db < level
    dec = C.Decoder(a.ckpt)
    as_text = lambda seq: "".join(chr(0x4E00 + c) for c in seq)

    # 1. line the decoded sounds up with the text
    sym, when, piece = heard(a.dir, pieces, dec)
    keys = list(dict.fromkeys(k for k, _ in sura_stream(suras)[1]))
    refs = {k: dec.expected(k) for k in keys}
    keys = [k for k in keys if refs[k]]
    while True:
        flat = [c for k in keys for c in refs[k]]
        at, n = {}, 0
        for k in keys:
            at[k] = (n, n + len(refs[k])); n += len(refs[k])
        to, same = line_up(flat, sym)
        weak = [k for k in keys if sum(same[at[k][0]:at[k][1]]) < a.min_found * len(refs[k])]
        if not weak:
            break
        print("not recited (too few sounds found):", ", ".join("%d:%d" % L.split_key(k) for k in weak))
        keys = [k for k in keys if k not in weak]
    found = {k: [to[i] for i in range(*at[k]) if same[i]] for k in keys}

    def run_at(t, min_len=0.05):
        """The quiet run holding time t, as seconds, or None."""
        f = min(len(quiet) - 1, int(t / C.HOP))
        for g in (f, f - 1, f + 1, f - 2, f + 2):
            if 0 <= g < len(quiet) and quiet[g]:
                i = j = g
                while i > 0 and quiet[i - 1]:
                    i -= 1
                while j < len(quiet) and quiet[j]:
                    j += 1
                return (i * C.HOP, j * C.HOP) if (j - i) * C.HOP >= min_len else None
        return None

    def mid(t):
        r = run_at(t)
        return (r[0] + r[1]) / 2 if r else t

    def pauses(t0, t1, min_len):
        """Middles and lengths of the quiet runs between two times, widest run kept whole."""
        lo, hi = max(0, int(t0 / C.HOP)), min(len(quiet), int(t1 / C.HOP))
        res = []
        for i, j in C.quiet_runs(quiet, lo, hi, 0.0):
            while i > 0 and quiet[i - 1]:
                i -= 1
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * C.HOP >= min_len:
                res.append(((i + j) / 2 * C.HOP, (j - i) * C.HOP))
        return res

    by_piece = collections.defaultdict(list)
    for s, p in zip(sym, piece):
        by_piece[p].append(s)

    # 2. first cut between each pair of ayahs
    cuts, notes = [], collections.defaultdict(list)
    for ka, kb in zip(keys, keys[1:]):
        ha, hb = found[ka][-1], found[kb][0]
        p, q = piece[ha], piece[hb]
        if q > p:
            c = q
            for j in range(q - 1, p, -1):         # pieces in between: said-again end of a, or false start of b
                s = as_text(by_piece.get(j, []))
                if not s:
                    continue
                if fuzz.partial_ratio(s, as_text(refs[kb])) > fuzz.partial_ratio(s, as_text(refs[ka])):
                    c = j
                else:
                    break
            if q - p > 1:
                notes[ka if c == q else kb].append("extra sounds between the ayahs")
            cuts.append(mid(start_of[c]))
        else:                                     # both ayahs in one piece: longest quiet stretch between them
            ps = pauses(when[ha] - 0.3, when[hb] + 1.0, 0.05)
            if ps:
                cuts.append(max(ps, key=lambda x: x[1])[0])
            else:
                lo, hi = int(when[ha] / C.HOP), max(int(when[ha] / C.HOP) + 2, int((when[hb] + 0.4) / C.HOP))
                cuts.append((lo + int(np.argmin(db[lo:hi]))) * C.HOP)
                notes[ka].append("no pause before the next ayah")
    last_piece = piece[found[keys[-1]][-1]]
    end = mid(start_of[last_piece + 1]) if last_piece + 1 in start_of else total
    segs = [dict(key=k, kind="ayah", start=s, end=e) for k, s, e in zip(keys, [0.0] + cuts, cuts + [end])]

    # 3. "آمين" after sura 1 is not part of the text: give it its own segment
    amin = dec.amin()
    for n, g in enumerate(segs[:-1]):
        if g["key"] != 1007:
            continue
        f, f2 = found[1007], found[segs[n + 1]["key"]]      # middle of one ayah to middle of the next
        ps = [m for m, ln in pauses(when[f[len(f) // 2]], when[f2[len(f2) // 2]], 0.3)]
        for c0, c1 in [(x, y) for i, x in enumerate(ps) for y in ps[i + 1:]]:
            got = dec.sounds(wav[int(c0 * SR): int(c1 * SR)], level)
            if Levenshtein.distance(got, amin) <= 2:
                g["end"] = c0; segs[n + 1]["start"] = c1
                segs.insert(n + 1, dict(key=1007, kind="amin", start=c0, end=c1))
                break
        break

    # 4. check every clip; where an end fails, try the pauses around the cut
    def ref_of(g):
        return amin if g["kind"] == "amin" else refs[g["key"]]

    def clip(g):
        return wav[int(round(g["start"] * SR)): int(round(g["end"] * SR))]

    def run_check(n):
        g = segs[n]; w = clip(g)
        g["heard"] = dec.sounds(w, level)
        g["check"] = C.check(g["heard"], ref_of(g), ref_of(segs[n - 1]) if n else None,
                             ref_of(segs[n + 1]) if n + 1 < len(segs) else None, C.edges_quiet(w, level))

    def joint(ga, gb, cur, reach=8.0):
        """Cuts to try for two ayahs said without a pause: around where the decoded sounds
        change from the end of one to the start of the next."""
        t0, t1 = max(ga["start"], cur - reach), min(gb["end"], cur + reach)
        am = dec.frames(wav[int(t0 * SR): int(t1 * SR)])
        h, ht, prev, last = [], [], dec.blank, None
        for t, x in enumerate(am):
            if x != dec.blank and x != prev:
                c = dec.cls[x]
                if c is not None and c != last:
                    h.append(c); ht.append(t0 + t / FPS)
                if c is not None:
                    last = c
            prev = x
        ra, rb, best = ref_of(ga)[-8:], ref_of(gb)[:8], None
        for k in range(2, len(h) - 1):
            cost = (min(Levenshtein.distance(h[max(0, k - n):k], ra) for n in range(6, 11))
                    + min(Levenshtein.distance(h[k:k + n], rb) for n in range(6, 11)))
            if best is None or cost < best[0]:
                best = (cost, k)
        if best is None or best[0] > 4:
            return []
        k = best[1]                               # sounds come out a little early: look past the change too
        out = []
        for c in np.arange((ht[k - 1] + ht[k]) / 2, ht[k] + 0.24, 0.04):
            lo = int((c - 0.04) / C.HOP)          # the quietest frame near each step
            out.append((lo + int(np.argmin(db[lo:lo + 8]))) * C.HOP)
        return sorted(set(out))

    def bad_end(c):
        return not c["ends_ok"] or c["extra"] in ("next", "end")

    def bad_start(c):
        return not c["starts_ok"] or c["extra"] in ("prev", "start")

    for n in range(len(segs)):
        run_check(n)
    print("first cut:  ", C.summary(segs), flush=True)
    for g in segs:
        if not g["check"]["pass"]:
            print("   %d:%d %s  %s" % (*L.split_key(g["key"]), g["kind"], C.reason(g["check"])), flush=True)
    for rnd in range(a.rounds):
        moved = 0
        for n in range(len(segs) - 1):
            ga, gb = segs[n], segs[n + 1]
            if not (bad_end(ga["check"]) or bad_start(gb["check"]) or not ga["check"]["quiet_ok"]):
                continue
            cur, reach = ga["end"], 4.0 * (rnd + 1)
            lo, hi = max(ga["start"] + 0.2, cur - reach), min(gb["end"] - 0.2, cur + reach)
            cands = [m for m, ln in pauses(lo, hi, 0.15) if lo < m < hi]
            cands += [c for c in joint(ga, gb, cur) if lo < c < hi]
            best = None
            for c in sorted(set(cands + [cur]), key=lambda x: abs(x - cur)):
                ta = dec.sounds(wav[int(max(ga["start"], c - 12) * SR): int(c * SR)], level)
                hb_ = dec.sounds(wav[int(c * SR): int(min(gb["end"], c + 12) * SR)], level)
                t, d1 = C._lead(ta[::-1], ref_of(ga)[::-1]); l, d0 = C._lead(hb_, ref_of(gb))
                cost = t + d1 + l + d0 + (0 if run_at(c) else 2)
                rank = (cost, float(db[int(c / C.HOP)]))       # ties: the quietest point
                if best is None or rank < best[0]:
                    best = (rank, c)
            if best and abs(best[1] - cur) > 1e-6:
                old = (ga["check"], gb["check"], ga["heard"], gb["heard"])
                ga["end"] = gb["start"] = best[1]
                run_check(n); run_check(n + 1)
                fails = lambda x, y: (not x["pass"]) + (not y["pass"])
                if fails(ga["check"], gb["check"]) > fails(old[0], old[1]):
                    ga["end"] = gb["start"] = cur            # no better: put it back
                    ga["check"], gb["check"], ga["heard"], gb["heard"] = old
                else:
                    moved += 1
        print(f"round {rnd + 1}: moved {moved} cuts;", C.summary(segs), flush=True)
        if not moved:
            break

    # 5. word times: forced alignment inside each matched piece, else the lined-up sounds
    scores = json.load(open(os.path.join(a.dir, "word_scores.json")))
    rows_by_piece, score_of = collections.defaultdict(list), {}
    for w in scores:
        rows_by_piece[w["piece"]].append(w)
        score_of.setdefault((w["key"], w["group"]), dict(full=round(w["full"], 2), basic=round(w["basic"], 2)))
    times = {}
    for pid, ws in sorted(rows_by_piece.items()):
        words = sorted({(w["key"], w["group"]) for w in ws})
        toks, wids, k = [], [], 0
        while k < len(words):
            ay, j = words[k][0], k
            while j + 1 < len(words) and words[j + 1][0] == ay:
                j += 1
            r = L.portion(ay, words[k][1], words[j][1])
            toks += r[0]; wids += [(ay, x) for x in r[1]]
            k = j + 1
        if any(t not in dec.vocab for t in toks):
            continue
        lp = torch.from_numpy(np.load(os.path.join(a.dir, "logp", f"{pid:04d}.npy")).astype(np.float32))
        if lp.shape[0] < len(toks) + 1:
            continue
        for w, (s, e) in word_times(lp, [dec.vocab[t] for t in toks], wids, dec.blank).items():
            times.setdefault(w, (start_of[pid] + s, start_of[pid] + e))
    rough = {}
    for k in keys:
        toks, wd = L.portion(k, 0, len(L.ayah_words(k)) - 1)
        seq, last = [], None                      # the word of each expected sound, same folding as the text
        for t, w in zip(toks, wd):
            c = dec.cls[dec.vocab[t]]
            if c is None or c == last:
                continue
            last = c; seq.append(w)
        for i, w in zip(range(*at[k]), seq):
            if same[i]:
                t = when[to[i]]
                s, e = rough.get((k, w), (t, t + 0.08))
                rough[(k, w)] = (min(s, t), max(e, t + 0.08))

    thr_path = a.thresholds or os.path.join(a.dir, "ayahs", "ayahs.json")
    thr = json.load(open(thr_path)).get("thresholds", {}) if os.path.exists(thr_path) else {}
    rows = []
    for g in segs:
        sura, ayah = L.split_key(g["key"])
        s0, e0 = round(g["start"], 3), round(g["end"], 3)
        name = f"{sura:03d}_{ayah:03d}" + ("_amin" if g["kind"] == "amin" else "") + ".wav"
        sf.write(os.path.join(out, name), wav[int(round(s0 * SR)): int(round(e0 * SR))], SR, subtype="PCM_16")
        row = dict(sura=sura, ayah=ayah, kind=g["kind"], start=s0, end=e0)
        if g["kind"] == "amin":
            row.update(label="آمين", file=name, words=[dict(text="آمين", start=0.0, end=round(e0 - s0, 2), abs_start=s0, abs_end=e0)], missing=0)
        else:
            ws, missing = [], 0
            for k, t in enumerate(L.ayah_words(g["key"])):
                se = None
                for cand in (times.get((g["key"], k)), rough.get((g["key"], k))):
                    if cand and cand[1] > s0 and cand[0] < e0:
                        se = (max(s0, cand[0]), min(e0, cand[1])); break
                missing += se is None
                ws.append(dict(text=t, start=round(se[0] - s0, 2) if se else None, end=round(se[1] - s0, 2) if se else None,
                               abs_start=round(se[0], 2) if se else None, abs_end=round(se[1], 2) if se else None,
                               **score_of.get((g["key"], k), {})))
            row.update(words=ws, missing=missing)
        row["check"] = g["check"]
        note = list(notes.get(g["key"], [])) if g["kind"] == "ayah" else []
        if g["check"]["again"]:
            note.append("a phrase is said again inside")
        if note:
            row["note"] = "; ".join(note)
        rows.append(row)
    json.dump(dict(suras=suras, thresholds=thr, quiet_db=round(level, 2), ayahs=rows),
              open(os.path.join(out, "ayahs.json"), "w"), ensure_ascii=False)
    gaps = sum(1 for x, y in zip(rows, rows[1:]) if x["end"] != y["start"])
    print(f"{len(rows)} clips, {rows[0]['start']:.2f}-{rows[-1]['end']:.2f} s of {total:.2f}; cuts that do not touch: {gaps}; "
          f"words not timed: {sum(r['missing'] for r in rows)}")
    print("final:      ", C.summary(rows))
    for r in rows:
        if not r["check"]["pass"] or r.get("note"):
            print(f"  {r['sura']}:{r['ayah']}{' amin' if r['kind'] == 'amin' else ''}  "
                  f"{C.reason(r['check']) or 'pass'}{'  [' + r['note'] + ']' if r.get('note') else ''}")


if __name__ == "__main__":
    main()
