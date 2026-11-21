"""Cut full-sura studio recordings into ayah clips using the ayah timing tables.

  python data_tools/studio_clips.py r06 --suras 78 79 80     # writes a small manifest only

data/gapless/<voice>/NNN.mp3 is one sura; data/timings_b/<voice>.db has rows
timings(sura, ayah, time[, words]) where time is the ms at which the ayah starts (ayah 0 is
the opening before ayah 1, ayah 999 marks the end). Ayah n spans time[n] up to the next row.
`words` (some voices only) is "n:start_ms:end_ms,..." per printed word, n from 1.
Audio is not saved; clips() yields 16 kHz mono arrays for scoring.
"""
import argparse, json, os, sqlite3

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SR = 16000


def timing_rows(voice, sura):
    db = sqlite3.connect(f"file:{os.path.join(DATA, 'timings_b', voice + '.db')}?mode=ro", uri=True)
    cols = [r[1] for r in db.execute("pragma table_info(timings)")]
    q = "select ayah, time" + (", words" if "words" in cols else "") + " from timings where sura=? order by time, ayah"
    rows = db.execute(q, (sura,)).fetchall()
    db.close()
    return rows


def parse_words(s):
    if not s:
        return None
    out = []
    for part in s.split(","):
        n, a, b = part.split(":")
        out.append((int(n), int(a) / 1000, int(b) / 1000))
    return out


def load_sura(voice, sura):
    wav, sr = sf.read(os.path.join(DATA, "gapless", voice, f"{sura:03d}.mp3"), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.numpy()


def clip_meta(voice, sura, max_ayah=None):
    """Clip rows (no audio) for one sura: ayah, start/end seconds, word times if any."""
    rows = timing_rows(voice, sura)
    out = []
    for i, r in enumerate(rows):
        ayah, t = r[0], r[1]
        if ayah in (0, 999) or i + 1 >= len(rows) or (max_ayah and ayah > max_ayah):
            continue
        end = rows[i + 1][1]
        if end <= t:
            continue
        words = parse_words(r[2]) if len(r) > 2 else None
        out.append(dict(voice=voice, sura=sura, ayah=ayah, start=t / 1000, end=end / 1000, words=words))
    return out


def clips(voice, suras, max_ayah=None):
    """Yield (meta, wav16k) for every ayah of the given suras. max_ayah: {sura: n} cap."""
    max_ayah = max_ayah or {}
    for sura in suras:
        path = os.path.join(DATA, "gapless", voice, f"{sura:03d}.mp3")
        if not os.path.exists(path):
            continue
        metas = clip_meta(voice, sura, max_ayah.get(sura))
        if not metas:
            continue
        wav = load_sura(voice, sura)
        for m in metas:
            a, b = int(m["start"] * SR), min(len(wav), int(m["end"] * SR))
            if b - a < SR // 4:
                continue
            yield m, wav[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice")
    ap.add_argument("--suras", type=int, nargs="+", required=True)
    a = ap.parse_args()
    rows = [m for s in a.suras for m in clip_meta(a.voice, s)]
    os.makedirs(os.path.join(DATA, "studio_c_clips"), exist_ok=True)
    out = os.path.join(DATA, "studio_c_clips", f"{a.voice}.json")
    json.dump(rows, open(out, "w"))
    secs = np.array([m["end"] - m["start"] for m in rows])
    print(f"{a.voice}: {len(rows)} ayah clips, {secs.sum() / 3600:.2f} h, median {np.median(secs):.1f} s -> {out}")


if __name__ == "__main__":
    main()
