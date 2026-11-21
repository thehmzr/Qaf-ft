"""Scores the checker on real recitation mistakes.

  python eval/real_mistakes.py scores SET     # SET: mdd_c, mdd_d, learner_b, studio_val
  python eval/real_mistakes.py summary        # needs scores for studio_val and the sets

Every word of the expected ayah gets a badness score in two modes: tajweed judged
("full") and words and vowels ("basic"). Labels per word: 0 correct, 1 word/vowel
mistake, 2 tajweed-only mistake. Thresholds for 1% and 0.1% false flags come from
correct words of the studio validation reciter (studio_val), never from the test set.
"""
import argparse, glob, io, json, os, random, re, sys, unicodedata

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
from qafft.phonetic import fold, fold_with_groups
from qafft.verify import basic_classes, extra_sounds, group_scores, to_basic, token_scores
from qafft.wordmap import ayah_phonemes, display_words, word_tokens

OUT = os.path.join(ROOT, "data/eval/realmistakes")
WV, TJ = 1, 2


def plain(w):
    """Arabic word without marks, for matching label words to display words."""
    w = unicodedata.normalize("NFKD", w)
    w = "".join(c for c in w if unicodedata.category(c)[0] == "L")
    return w.replace("ٱ", "ا").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")


def find_word(w, words):
    from rapidfuzz import fuzz
    p = plain(w)
    s = [fuzz.ratio(p, plain(x)) for x in words]
    i = int(np.argmax(s))
    return i if s[i] >= 60 else None


class Checker:
    def __init__(self, ckpt):
        from pilot import QafModel
        torch.set_num_threads(4)
        ck = torch.load(ckpt, map_location="cpu")
        self.tokens = ck["tokens"]; self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.blank = len(self.tokens)
        self.model = QafModel(len(self.tokens), None); self.model.load_state_dict(ck["model"]); self.model.eval()

    @torch.no_grad()
    def logp(self, wav):
        w = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        lp, hl = self.model(w, torch.tensor([w.shape[1]]))
        return lp[0, : hl[0]]

    def _score(self, lp, ids, grps, blank):
        ts = token_scores(lp, ids, blank)
        if ts is None:
            return None
        # badness per word: the weakest expected sound, or a clear extra sound
        bad = {g: -v for g, v in group_scores(ts, grps).items()}
        for g, v in extra_sounds(lp, ids, grps, blank).items():
            bad[g] = max(bad.get(g, 0.0), v)
        return bad

    def both(self, lp, toks, grps):
        if any(t not in self.vocab for t in toks):
            return None
        ids = [self.vocab[t] for t in toks]
        full = self._score(lp, ids, grps, self.blank)
        blp, bids, bgrps, k = to_basic(lp, self.tokens, ids, grps)
        basic = self._score(blp, bids, bgrps, k)
        if full is None or basic is None:
            return None
        return {"full": full, "basic": basic}


def load_wav(path_or_bytes):
    wav, sr = sf.read(path_or_bytes, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import torchaudio.functional as F
        wav = F.resample(torch.from_numpy(wav), sr, 16000).numpy()
    return wav


# ---- label readers: each yields (clip_id, group_key, sura, aya, audio, {word_index: label})

def read_mdd_c():
    import pandas as pd
    d = os.path.join(ROOT, "data/mdd_c")
    m = pd.read_csv(os.path.join(d, "metadata.csv"))
    for r in m.itertuples():
        f = os.path.join(d, r.file_name)
        if not os.path.exists(f):
            continue
        words = display_words(112, int(r.verse_number))
        lab = {}
        if r.label_name == "error":
            locs = [x.strip() for x in re.split("[,،]", str(r.error_location)) if x.strip() and x.strip() != "nan"]
            kinds = [x.strip() for x in re.split("[,،]", str(r.error_type)) if x.strip()]
            if not locs:
                continue                      # error with no word given: unusable
            for j, w in enumerate(locs):
                i = find_word(w, words)
                if i is None:
                    continue
                # vowel errors are written "حركات"; everything else here is tajweed
                k = kinds[j] if len(kinds) == len(locs) else " ".join(kinds)
                lab[i] = WV if ("حركات" in k or lab.get(i) == WV) else TJ
            if not lab:
                continue
        spk = re.match(r"ID(\d+)", os.path.basename(f)).group(1)
        yield os.path.basename(f), spk, 112, int(r.verse_number), f, lab


def read_mdd_d():
    d = os.path.join(ROOT, "data/mdd_d")
    for line in open(os.path.join(d, "metadata.jsonl")):
        x = json.loads(line)
        if x["riwayah"] != "Hafs":
            continue
        sura, aya = int(x["surah"]), int(x["ayah"])
        words = display_words(sura, aya)
        if abs(len(x["text"].split()) - len(words)) > 1:
            continue                          # chunk text is not the whole ayah
        lab = {}
        for w, kinds in x["errors"].items():
            i = find_word(w, words)
            if i is None:
                lab = None; break
            # wording, letter and vowel-mark errors are word/vowel mistakes
            lab[i] = WV if set(kinds) - {"Tajweed"} else TJ
        if lab is None:
            continue
        yield x["file_name"], x["recording_id"], sura, aya, os.path.join(d, x["file_name"]), lab


def basic_keys(phon_word):
    toks = fold(phon_word)
    vocab = sorted(set(toks))
    cls, _ = basic_classes(vocab)
    names = {}
    out = []
    for t in toks:
        c = cls[vocab.index(t)]
        if c is None:
            continue
        # class names, not indices, so two words compare
        ch, n = t[0], (int(t[1:]) if len(t) > 1 else 1)
        key = ch if ch in "اۥۦ" else ("ن2" if ch == "ں" or (ch == "ن" and n >= 2) else
               "م2" if ch == "۾" or (ch == "م" and n >= 2) else t)
        if not out or out[-1] != key:
            out.append(key)
    return out


def read_learner_b():
    f = glob.glob(os.path.join(ROOT, "data/learner_b/data/*.parquet"))[0]
    sura, aya = 5, 109
    # the clips cover only the last words of the ayah; one spoken group per word there
    said0 = pq.read_table(f, columns=["phonetic_transcript"]).column(0)[0].as_py().split()
    canon = ayah_phonemes(sura, aya).split()[-len(said0):]
    toks, grps = fold_with_groups(" ".join(canon))
    for r in pq.read_table(f).to_pylist():
        said = r["phonetic_transcript"].split()
        if len(said) != len(canon):
            continue
        lab = {}
        for i, (s, c) in enumerate(zip(said, canon)):
            ks, kc = basic_keys(s), basic_keys(c)
            # noon/meem held or hidden differently is a ghunnah rule in this set: tajweed
            soft = lambda k: [x[0] if x in ("ن2", "م2") else x for x in k]
            if soft(ks) != soft(kc):
                lab[i] = WV
            elif fold(s) != fold(c):
                lab[i] = TJ
        yield r["id"], r["original_id"].split("_")[0], (toks, grps), aya, io.BytesIO(r["audio"]["bytes"]), lab


def run_scores(name, ckpt, limit):
    os.makedirs(OUT, exist_ok=True)
    ch = Checker(ckpt)
    out = []
    if name == "studio_val":
        files = sorted(glob.glob(os.path.join(ROOT, "data/heldout/moshaf_19.0/*.parquet")))
        for f in files:
            for r in pq.read_table(f, columns=["audio", "phonemes", "start_span", "duration_seconds"]).to_pylist():
                if len(out) >= limit:
                    break
                if not r["start_span"] or r["duration_seconds"] > 20:
                    continue
                toks, grps = fold_with_groups(r["phonemes"])
                sc = ch.both(ch.logp(load_wav(io.BytesIO(r["audio"]["bytes"]))), toks, grps)
                if sc:
                    out.append(dict(clip=len(out), words={m: {str(g): v for g, v in s.items()} for m, s in sc.items()}))
    else:
        reader = {"mdd_c": read_mdd_c, "mdd_d": read_mdd_d, "learner_b": read_learner_b}[name]
        for cid, grp, sura, aya, audio, lab in reader():
            toks, tw = sura if isinstance(sura, tuple) else word_tokens(sura, aya)[:2]
            try:
                sc = ch.both(ch.logp(load_wav(audio)), toks, tw)
            except Exception as e:
                print("skip", cid, e); continue
            if not sc:
                continue
            out.append(dict(clip=cid, group=grp, labels={str(k): v for k, v in lab.items()},
                            words={m: {str(g): v for g, v in s.items()} for m, s in sc.items()}))
            if len(out) % 50 == 0:
                print(name, len(out), flush=True)
    json.dump(out, open(os.path.join(OUT, f"scores_{name}.json"), "w"))
    print(name, len(out), "clips scored")


# ---- summary

def auc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rows_of(clips, mode):
    """Per clip: list of (badness, label)."""
    return [[(v, c["labels"].get(g, 0)) for g, v in c["words"][mode].items()] for c in clips]


def stats(per_clip, thr, prec_targets=(0.9, 0.95)):
    rows = [r for c in per_clip for r in c]
    b = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    neg, pos, tj = b[y == 0], b[y == WV], b[y == TJ]
    s = dict(n_correct=int(len(neg)), n_wordvowel=int(len(pos)), n_tajweed_only=int(len(tj)))
    s["auc"] = auc(pos, neg) if len(pos) and len(neg) else None
    for k, t in thr.items():
        fp, tp = int((neg > t).sum()), int((pos > t).sum())
        s[k] = dict(threshold=t, catch=tp / len(pos) if len(pos) else None,
                    false_flags=fp / len(neg) if len(neg) else None,
                    precision=tp / (tp + fp) if tp + fp else None,
                    tajweed_only_flagged=float((tj > t).mean()) if len(tj) else None)
    # chosen on the test set (optimistic): lowest threshold that reaches the precision target
    for p in prec_targets:
        best = None
        for t in np.unique(np.r_[pos, neg]):
            tp, fp = (pos >= t).sum(), (neg >= t).sum()
            if tp and tp / (tp + fp) >= p:
                best = dict(threshold=float(t), catch=float(tp / len(pos)), false_flags=float(fp / len(neg)))
                break
        s[f"test_chosen_precision_{int(p * 100)}"] = best
    return s


def boot(per_clip, thr, draws=1000):
    rng = random.Random(0); n = len(per_clip); keys = {}
    for _ in range(draws):
        s = stats([per_clip[rng.randrange(n)] for _ in range(n)], thr, ())
        for k in ("auc",) + tuple(f"{t}.{m}" for t in thr for m in ("catch", "false_flags", "precision")):
            v = s[k] if "." not in k else s[k.split(".")[0]][k.split(".")[1]]
            if v is not None:
                keys.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in keys.items()}


def run_summary():
    val = json.load(open(os.path.join(OUT, "scores_studio_val.json")))
    res = dict(threshold_source=f"studio validation reciter (moshaf_19.0), {len(val)} segments up to 20 s")
    thrs = {}
    for mode in ("full", "basic"):
        neg = np.sort([v for c in val for v in c["words"][mode].values()])
        thrs[mode] = {"at_1pct": float(neg[int(0.99 * len(neg))]), "at_0p1pct": float(neg[min(len(neg) - 1, int(0.999 * len(neg)))])}
        res[f"studio_correct_words_{mode}"] = int(len(neg))
    res["thresholds"] = thrs
    for name in ("mdd_c", "mdd_d", "learner_b"):
        p = os.path.join(OUT, f"scores_{name}.json")
        if not os.path.exists(p):
            continue
        clips = json.load(open(p))
        res[name] = dict(clips=len(clips), clips_all_correct=sum(not c["labels"] for c in clips),
                         clips_with_wordvowel=sum(WV in c["labels"].values() for c in clips))
        for mode in ("full", "basic"):
            pc = rows_of(clips, mode)
            s = stats(pc, thrs[mode]); s["intervals_95"] = boot(pc, thrs[mode])
            res[name][mode] = s
            print(name, mode, json.dumps({k: s[k] for k in ("n_correct", "n_wordvowel", "n_tajweed_only", "auc")}),
                  {k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in s[k].items()} for k in thrs[mode]},
                  s["test_chosen_precision_90"], s["test_chosen_precision_95"])
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["scores", "summary"])
    ap.add_argument("set", nargs="?")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/full/best.pt"))
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    run_scores(a.set, a.ckpt, a.limit) if a.what == "scores" else run_summary()
