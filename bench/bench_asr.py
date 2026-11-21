"""Transcribe a test set with an existing model and save predictions (resumable).

  python bench/bench_asr.py --model whisper|nemo --set learner_a|studio_b [--limit N]

learner_a    : the 905 learner clips labelled correct / in_correct (real phone audio)
studio_b : professional reciters (one shard of the public test split)
Output: bench/out/<model>_<set>.jsonl with one {id, label, ref, hyp, dur} per clip.
"""
import argparse, glob, io, json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from progress import report


def load_audio(b):
    wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def items(name, limit):
    if name == "learner_a":
        n = 0
        for f in sorted(glob.glob(f"{ROOT}/data/learner_a/data/*.parquet")):
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["audio", "Aya", "final_label", "reciter_id", "Surah"]).to_pylist()
                for r in t:
                    if r["final_label"] not in ("correct", "in_correct"):
                        continue
                    yield dict(id=r["audio"]["path"], label=r["final_label"], ref=r["Aya"],
                               spk=r["reciter_id"], surah=r["Surah"], audio=r["audio"]["bytes"])
                    n += 1
                    if limit and n >= limit:
                        return
    else:
        f = glob.glob(f"{ROOT}/data/studio_b_test/data/*.parquet")[0]
        pf = pq.ParquetFile(f)
        n = 0
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                if r["duration"] > 25:
                    continue
                yield dict(id=f"{r['reciter']}_{rg}_{n}", label="correct", ref=r["text"],
                           spk=r["reciter"], surah="", audio=r["audio"]["bytes"])
                n += 1
                if limit and n >= limit:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(f"{ROOT}/bench/out", exist_ok=True)
    out = f"{ROOT}/bench/out/{a.model}_{a.set}.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    total = a.limit or (905 if a.set == "learner_a" else 1700)

    import torch
    torch.set_num_threads(4)
    if a.model == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        p = f"{ROOT}/models/model_d"
        proc = WhisperProcessor.from_pretrained(p)
        model = WhisperForConditionalGeneration.from_pretrained(p).eval()

        def run(wav):
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=200)
            return proc.batch_decode(ids, skip_special_tokens=True)[0]
    else:
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.restore_from(
            f"{ROOT}/models/base/base.nemo", map_location="cpu").eval()
        if a.model == "nemo_ctc":
            model.change_decoding_strategy(decoder_type="ctc")

        def run(wav):
            with torch.no_grad():
                r = model.transcribe([wav], batch_size=1, verbose=False)
            r = r[0] if not isinstance(r, tuple) else r[0][0]
            return r.text if hasattr(r, "text") else str(r)

    n = len(done)
    with open(out, "a") as fo:
        for it in items(a.set, a.limit):
            if it["id"] in done:
                continue
            wav = load_audio(it.pop("audio"))
            t0 = time.time()
            try:
                hyp = run(wav)
            except Exception as e:  # keep going; record the failure
                hyp = f"__ERROR__ {type(e).__name__}"
            it.update(hyp=hyp, dur=round(len(wav) / 16000, 2), rt=round(time.time() - t0, 2))
            fo.write(json.dumps(it, ensure_ascii=False) + "\n")
            fo.flush()
            n += 1
            report(n, total, note=f"{a.model} on {a.set}", unit="clips")
    report(n, n, note="finished", unit="clips", force=True)


if __name__ == "__main__":
    main()
