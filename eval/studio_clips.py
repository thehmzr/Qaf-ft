"""False flags per printed word on correct studio ayah clips from held-out voices.

  python eval/studio_clips.py heldout [--limit N]          # thresholds from a known-correct set
  python eval/studio_clips.py score r06 --suras 2 18 36    # score every printed word
  python eval/studio_clips.py report r06 r10               # rates per voice and mode

Each ayah clip (cut by data_tools/studio_clips.py) is scored against the whole ayah
phonetized as one ayah: an ayah ends at a pause, so the ayah form is the one the reciter says.
Every printed word gets a badness in both modes (tajweed judged; words and vowels only).
A planted mistake per clip (same perturbations as eval/evaluate.py) gives a catch rate.
Where the timing table has word times, the forced alignment word edges are compared to them.

Thresholds: 99th and 99.9th percentile of word badness on the held-out known-correct set
(moshaf 19.0 shards if present, else 5.0/13.0), and separately from the voice's own first half
of clips with the second half as test. Every clip is taken as correct, so every flag is false.
"""
import argparse, glob, io, json, os, random, sys, time

import numpy as np
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
sys.path.insert(0, os.path.join(ROOT, "data_tools")); sys.path.insert(0, os.path.join(ROOT, "eval"))
from qafft.verify import extra_sounds, group_scores, to_basic, token_scores

OUT = os.path.join(ROOT, "data", "eval", "studioclips")
MAX_SEC = 45.0


def load_model(ckpt):
    from pilot import QafModel
    torch.set_num_threads(4)
    ck = torch.load(ckpt, map_location="cpu")
    tokens = ck["tokens"]
    model = QafModel(len(tokens)); model.load_state_dict(ck["model"]); model.eval()
    return model, tokens


def badness(lp, ids, grps, blank):
    ts = token_scores(lp, ids, blank)
    if ts is None:
        return None
    bad = {g: -v for g, v in group_scores(ts, grps).items()}
    for g, v in extra_sounds(lp, ids, grps, blank).items():
        bad[g] = max(bad.get(g, 0.0), v)
    return bad


def both(lp, tokens, ids, grps, blank):
    full = badness(lp, ids, grps, blank)
    blp, bids, bgrps, k = to_basic(lp, tokens, ids, grps)
    basic = badness(blp, bids, bgrps, k)
    return full, basic


def token_frames(lp, ids, blank):
    """(first, last) frame of each expected token on the forced alignment path."""
    al, _ = torchaudio.functional.forced_align(lp.unsqueeze(0), torch.tensor([ids]), blank=blank)
    al = al[0].tolist()
    span, pos, prev = [None] * len(ids), -1, blank
    for t, a in enumerate(al):
        if a != blank and (a != prev or (t and al[t - 1] == blank)):
            pos += 1
        if a != blank and 0 <= pos < len(ids):
            span[pos] = (span[pos][0], t) if span[pos] else (t, t)
        prev = a
    return span


def cmd_heldout(a):
    import pyarrow.parquet as pq, soundfile as sf
    from qafft.phonetic import fold_with_groups
    files = sorted(glob.glob(os.path.join(ROOT, "data/heldout/moshaf_19.0/*.parquet")))
    name = "moshaf 19.0"
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, "data/heldout/moshaf_5.0/*.parquet")) +
                       glob.glob(os.path.join(ROOT, "data/heldout/moshaf_13.0/*.parquet")))
        name = "moshaf 5.0/13.0"
    model, tokens = load_model(a.ckpt)
    vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    vals = {"full": [], "basic": []}
    per_file = max(1, a.limit // len(files))
    for f in files:
        pf, n = pq.ParquetFile(f), 0
        for rg in range(pf.num_row_groups):
            if n >= per_file:
                break
            for r in pf.read_row_group(rg, columns=["audio", "phonemes", "start_span", "duration_seconds"]).to_pylist():
                if n >= per_file:
                    break
                if not r["start_span"] or r["duration_seconds"] > 20:
                    continue
                toks, grps = fold_with_groups(r["phonemes"])
                if any(t not in vocab for t in toks):
                    continue
                wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                with torch.no_grad():
                    w = torch.from_numpy(wav).unsqueeze(0)
                    lp, hl = model(w, torch.tensor([w.shape[1]]))
                    lp = lp[0, : hl[0]].float()
                full, basic = both(lp, tokens, [vocab[t] for t in toks], grps, blank)
                if full is None or basic is None:
                    continue
                vals["full"] += list(full.values()); vals["basic"] += list(basic.values())
                n += 1
    out = dict(name=name, clips=a.limit, words=len(vals["full"]))
    for m in ("full", "basic"):
        v = np.array(vals[m])
        out[m] = dict(flag=float(np.quantile(v, 0.99)), strong=float(np.quantile(v, 0.999)))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "heldout_thresholds.json"), "w"), indent=1)
    print(json.dumps(out))


def cmd_score(a):
    import importlib.util
    spec = importlib.util.spec_from_file_location("studio_clips_data", os.path.join(ROOT, "data_tools", "studio_clips.py"))
    sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    clips = sc.clips
    from qafft.wordmap import word_tokens
    from evaluate import perturb
    model, tokens = load_model(a.ckpt)
    vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    rng = random.Random(0)
    caps = {2: a.max_ayah_2}
    rows, planted, bounds, skipped = [], [], [], dict(long=0, oov=0, short=0)
    manifest = []
    t0 = time.time()
    for k, (m, wav) in enumerate(clips(a.voice, a.suras, caps)):
        if time.time() - t0 > a.budget:
            print("time budget reached"); break
        sec = len(wav) / 16000
        if sec > MAX_SEC:
            skipped["long"] += 1; continue
        toks, tw, n_words = word_tokens(m["sura"], m["ayah"])
        if any(t not in vocab for t in toks):
            skipped["oov"] += 1; continue
        ids = [vocab[t] for t in toks]
        with torch.no_grad():
            w = torch.from_numpy(wav).unsqueeze(0)
            lp, hl = model(w, torch.tensor([w.shape[1]]))
            lp = lp[0, : hl[0]].float()
        full, basic = both(lp, tokens, ids, tw, blank)
        if full is None or basic is None:
            skipped["short"] += 1; continue
        manifest.append(m)
        for g in range(n_words):
            if g in full:
                rows.append(dict(i=len(manifest) - 1, sura=m["sura"], ayah=m["ayah"], word=g,
                                 full=full[g], basic=basic.get(g, 0.0)))
        p = perturb(toks, tw, rng, vocab)
        if p is not None:
            pf, pb = both(lp, tokens, [vocab[t] for t in p[0]], p[1], blank)
            if pf is not None and pb is not None:
                g = p[2]["group"]
                planted.append(dict(i=len(manifest) - 1, kind=p[2]["kind"],
                                    full=pf.get(g, max(pf.values())), basic=pb.get(g, max(pb.values()))))
        if m["words"] and len(m["words"]) == n_words:
            fs = sec / lp.shape[0]
            span = token_frames(lp, ids, blank)
            for g, (n, ws, we) in enumerate(m["words"]):
                fr = [span[j] for j in range(len(ids)) if tw[j] == g and span[j]]
                if not fr:
                    continue
                s = m["start"] + min(f[0] for f in fr) * fs
                e = m["start"] + (max(f[1] for f in fr) + 1) * fs
                bounds.append(dict(sura=m["sura"], ayah=m["ayah"], word=g, start_err=s - ws, end_err=e - we,
                                   first=g == 0, last=g == n_words - 1))
        if k % 25 == 0:
            print(f"{k} clips, {len(rows)} words, {time.time() - t0:.0f} s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "studio_c_clips"), exist_ok=True)
    json.dump(manifest, open(os.path.join(ROOT, "data", "studio_c_clips", f"{a.voice}_scored.json"), "w"))
    json.dump(dict(voice=a.voice, suras=a.suras, skipped=skipped, rows=rows, planted=planted, bounds=bounds),
              open(os.path.join(OUT, f"{a.voice}.json"), "w"))
    print(f"{a.voice}: {len(manifest)} clips, {len(rows)} words, skipped {skipped}, {time.time() - t0:.0f} s")


def rates(rows, planted, thr, mode):
    out = {}
    for level in ("flag", "strong"):
        t = thr[level]
        flags = np.array([r[mode] > t for r in rows])
        by = {}
        for r in rows:
            by.setdefault(r["i"], []).append(r[mode] > t)
        out[level] = dict(threshold=round(t, 3), word_rate=float(flags.mean()), flagged=int(flags.sum()),
                          words=len(rows), ayah_rate=float(np.mean([any(v) for v in by.values()])), ayahs=len(by),
                          catch=float(np.mean([p[mode] > t for p in planted])) if planted else None)
    return out


def cmd_report(a):
    hp = os.path.join(OUT, "heldout_thresholds.json")
    held = json.load(open(hp)) if os.path.exists(hp) else None
    summary = {}
    for v in a.voices:
        d = json.load(open(os.path.join(OUT, f"{v}.json")))
        rows, planted = d["rows"], d["planted"]
        n = max(r["i"] for r in rows) + 1
        thr_rows = [r for r in rows if r["i"] < n // 2]
        test_rows = [r for r in rows if r["i"] >= n // 2]
        test_pl = [p for p in planted if p["i"] >= n // 2]
        res = dict(clips=n, words=len(rows), skipped=d["skipped"], modes={})
        print(f"\n{v}: {n} ayah clips, {len(rows)} printed words, planted {len(planted)}, skipped {d['skipped']}")
        for mode in ("full", "basic"):
            q = np.quantile([r[mode] for r in rows], [0.5, 0.9, 0.99, 0.999]).round(3).tolist()
            mres = dict(quantiles_50_90_99_999=q)
            if held:
                mres["heldout"] = rates(rows, planted, held[mode], mode)
            own = {lv: float(np.quantile([r[mode] for r in thr_rows], qq)) for lv, qq in (("flag", .99), ("strong", .999))}
            mres["own_half"] = rates(test_rows, test_pl, own, mode)
            res["modes"][mode] = mres
            print(f"  {mode:<5} word badness quantiles 50/90/99/99.9%: {q}")
            for src in ("heldout", "own_half"):
                if src not in mres:
                    continue
                for lv, r in mres[src].items():
                    print(f"    {src:<8} {lv:<6} thr {r['threshold']:6.3f}  words flagged {r['word_rate']:.2%} "
                          f"({r['flagged']}/{r['words']})  ayahs with a flag {r['ayah_rate']:.1%} of {r['ayahs']}  "
                          f"planted caught {r['catch']:.1%}")
        b = d["bounds"]
        if b:
            se = np.abs([x["start_err"] for x in b]); ee = np.abs([x["end_err"] for x in b])
            mid = [x for x in b if not x["first"] and not x["last"]]
            ms = np.abs([x["start_err"] for x in mid]); me = np.abs([x["end_err"] for x in mid])
            res["bounds"] = dict(words=len(b), start_med=float(np.median(se)), start_p90=float(np.percentile(se, 90)),
                                 end_med=float(np.median(ee)), end_p90=float(np.percentile(ee, 90)),
                                 inner_words=len(mid), inner_start_med=float(np.median(ms)) if mid else None,
                                 inner_start_p90=float(np.percentile(ms, 90)) if mid else None,
                                 inner_end_med=float(np.median(me)) if mid else None,
                                 inner_end_p90=float(np.percentile(me, 90)) if mid else None,
                                 start_bias_med=float(np.median([x["start_err"] for x in b])),
                                 end_bias_med=float(np.median([x["end_err"] for x in b])))
            print("  word edges vs timing table (s):", {k: round(x, 3) if isinstance(x, float) else x for k, x in res["bounds"].items()})
        summary[v] = res
    json.dump(dict(thresholds=held, voices=summary), open(os.path.join(OUT, "report.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["heldout", "score", "report"])
    ap.add_argument("voices", nargs="*")
    ap.add_argument("--suras", type=int, nargs="+", default=[2, 18, 36, 55, 67] + list(range(78, 115)))
    ap.add_argument("--max-ayah-2", type=int, default=100)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--budget", type=float, default=1e9, help="stop scoring after this many seconds")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/full/best.pt"))
    a = ap.parse_args()
    if a.cmd == "heldout":
        cmd_heldout(a)
    elif a.cmd == "score":
        for v in a.voices:
            a.voice = v; cmd_score(a)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
