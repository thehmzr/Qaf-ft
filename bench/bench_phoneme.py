"""Qaf-ft verifier prototype on real learner clips.

Uses an existing open phoneme model (model_a, benchmarking only) to test the
design: instead of transcribing and diffing text, score the EXPECTED ayah's sound
sequence against the audio.
  per  : edit distance between free phoneme decode and expected phonemes / length
  llr  : (log P_ctc(expected) - log P(best path)) per frame; 0 = audio fits the ayah perfectly
  minp : lowest aligned per-phoneme log posterior (the single weakest sound)
Output: bench/out/phoneme_learner_a.jsonl
"""
import glob, io, json, os, re, sys
import numpy as np, torch, soundfile as sf
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "bench"))
from progress import report
from score import norm
from rapidfuzz.distance import Levenshtein
import quran_transcript as qt
from quran_transcript import Aya, quran_phonetizer, MoshafAttributes

MD = f"{ROOT}/models/model_a"


def load_model():
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel, AutoFeatureExtractor
    from safetensors.torch import load_file
    cfg = json.load(open(f"{MD}/config.json"))
    keep = Wav2Vec2BertConfig().to_dict().keys()
    enc = Wav2Vec2BertModel(Wavcfg := Wav2Vec2BertConfig(**{k: v for k, v in cfg.items() if k in keep}))
    sd = load_file(f"{MD}/model.safetensors")
    pref = sorted({k.split(".")[0] for k in sd})
    enc_pref = [p for p in pref if p != "level_to_lm_head"][0] + "."
    missing, unexpected = enc.load_state_dict({k[len(enc_pref):]: v for k, v in sd.items() if k.startswith(enc_pref)}, strict=False)
    print("encoder prefix", enc_pref, "missing", len(missing), "unexpected", len(unexpected), flush=True)
    head = torch.nn.Linear(cfg["hidden_size"], 43)
    head.load_state_dict({"weight": sd["level_to_lm_head.phonemes.weight"], "bias": sd["level_to_lm_head.phonemes.bias"]})
    fe = AutoFeatureExtractor.from_pretrained(MD)
    vocab = json.load(open(f"{MD}/vocab.json"))["phonemes"]
    return enc.eval(), head.eval(), fe, vocab


def ayah_index():
    idx = {}
    a = Aya(1, 1)
    for s in range(1, 115):
        n = Aya(s, 1).get().num_ayat_in_sura
        for i in range(1, n + 1):
            u = Aya(s, i).get().uthmani
            idx.setdefault(norm(u), u)
    return idx


def main(limit=0):
    torch.set_num_threads(4)
    enc, head, fe, vocab = load_model()
    idx = ayah_index()
    print("ayahs indexed", len(idx), flush=True)
    mos = MoshafAttributes(rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=4)
    out = f"{ROOT}/bench/out/phoneme_learner_a.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    n, skipped = len(done), 0
    with open(out, "a") as fo:
        for f in sorted(glob.glob(f"{ROOT}/data/learner_a/data/*.parquet")):
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                for r in pf.read_row_group(rg, columns=["audio", "Aya", "final_label", "reciter_id"]).to_pylist():
                    if r["final_label"] not in ("correct", "in_correct") or r["audio"]["path"] in done:
                        continue
                    key = norm(r["Aya"])
                    if key not in idx:
                        skipped += 1
                        continue
                    ref = quran_phonetizer(idx[key], mos, remove_spaces=True).phonemes
                    ids = [vocab[c] for c in ref if c in vocab]
                    wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    if wav.ndim > 1: wav = wav.mean(1)
                    if sr != 16000:
                        import librosa; wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    with torch.no_grad():
                        x = fe(wav, sampling_rate=16000, return_tensors="pt")
                        lp = torch.log_softmax(head(enc(**x).last_hidden_state)[0], -1)  # (T, 43)
                    T = lp.shape[0]
                    best = lp.max(-1)
                    greedy = [int(i) for i in torch.unique_consecutive(best.indices) if int(i) != 0]
                    per = Levenshtein.distance(greedy, ids) / max(1, len(ids))
                    rec = dict(id=r["audio"]["path"], label=r["final_label"], spk=r["reciter_id"], per=round(per, 4), T=T, L=len(ids))
                    if T >= 2 * len(ids) + 1 or T > len(ids):
                        try:
                            nll = torch.nn.functional.ctc_loss(lp.unsqueeze(1), torch.tensor([ids]), torch.tensor([T]), torch.tensor([len(ids)]), blank=0, reduction="sum", zero_infinity=False)
                            rec["llr"] = round(float((-nll - best.values.sum()) / T), 5)
                            import torchaudio
                            al, sc = torchaudio.functional.forced_align(lp.unsqueeze(0), torch.tensor([ids]), blank=0)
                            al, sc = al[0], sc[0]
                            tok = {}
                            pos, prev = -1, 0
                            for t in range(T):
                                a_ = int(al[t])
                                if a_ != 0 and a_ != prev or (a_ != 0 and t > 0 and int(al[t - 1]) == 0):
                                    pos += 1
                                if a_ != 0:
                                    tok.setdefault(pos, []).append(float(sc[t]))
                                prev = a_
                            means = [float(np.mean(v)) for v in tok.values()]
                            rec["minp"] = round(min(means), 4) if means else None
                            rec["meanp"] = round(float(np.mean(means)), 4) if means else None
                        except Exception as e:
                            rec["err"] = type(e).__name__
                    fo.write(json.dumps(rec) + "\n"); fo.flush()
                    n += 1
                    report(n, 850, note=f"phoneme verifier, skipped {skipped} non-ayah clips", unit="clips")
                    if limit and n >= limit:
                        return
    report(n, n, note=f"finished, skipped {skipped}", unit="clips", force=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
