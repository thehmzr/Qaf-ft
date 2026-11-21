"""Juz 'Amma pilot: fine-tune the Arabic FastConformer encoder to emit sound tokens.

Runs on one GPU (Kaggle T4/P100) or, for a smoke test, on CPU.

  python train/pilot.py --data DIR --out DIR [--steps N] [--smoke]

--data holds parquet shards of the expert-recitation set (one folder per moshaf).
Only segments from suras 78-114 are used. Whole reciters are held out for
validation and test. Training resumes from --out/last.pt if it exists.
"""
import argparse, glob, io, json, math, os, random, sys, time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qafft.phonetic import fold

BASE = "base"                       # label in data/sources.json, looked up when needed
TEST_MOSHAF = {"5.0", "13.0"}      # never trained on
VAL_MOSHAF = {"19.0"}
FIRST_SURA = 78


class Segments(torch.utils.data.Dataset):
    """Reads audio bytes and phoneme strings straight from parquet row groups."""

    def __init__(self, files, vocab, max_sec=20.0, first_sura=FIRST_SURA):
        self.vocab, self.items = vocab, []
        for f in files:
            pf = pq.ParquetFile(f)
            if "start_span" not in pf.schema_arrow.names or "phonemes" not in pf.schema_arrow.names:
                continue
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["start_span", "duration_seconds", "phonemes"]).to_pylist()
                for i, r in enumerate(t):
                    if not r["start_span"] or r["start_span"]["sura_idx"] < first_sura:
                        continue
                    if r["duration_seconds"] > max_sec or not r["phonemes"]:
                        continue
                    toks = fold(r["phonemes"])
                    if any(x not in vocab for x in toks):
                        continue
                    # 12.5 frames/s doubled by the head = 25 frames/s
                    if len(toks) > 25 * r["duration_seconds"] - 2:
                        continue
                    self.items.append((f, rg, i, r["duration_seconds"], [vocab[x] for x in toks]))
        self._cache = (None, None)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        f, rg, i, dur, ids = self.items[k]
        if self._cache[0] != (f, rg):
            self._cache = ((f, rg), pq.ParquetFile(f).read_row_group(rg, columns=["audio"]).column("audio").to_pylist())
        wav, sr = sf.read(io.BytesIO(self._cache[1][i]["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, sr
        return torch.from_numpy(wav), torch.tensor(ids)


def collate(batch):
    wavs, ids = zip(*batch)
    wl = torch.tensor([len(w) for w in wavs]); il = torch.tensor([len(i) for i in ids])
    return (nn.utils.rnn.pad_sequence(wavs, batch_first=True), wl,
            nn.utils.rnn.pad_sequence(ids, batch_first=True), il)


def augment(wav, wl):
    """Light, label-safe augmentation: gain and additive noise. No time-warping,
    because stretching changes the madd counts written in the labels."""
    gain = 10 ** (torch.empty(wav.size(0), 1).uniform_(-6, 6) / 20)
    wav = wav * gain
    snr = torch.empty(wav.size(0), 1).uniform_(8, 35)
    rms = wav.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-4)
    mask = (torch.rand(wav.size(0), 1) < 0.5).float()
    return wav + mask * torch.randn_like(wav) * rms / (10 ** (snr / 20))


class QafModel(nn.Module):
    def __init__(self, n_tokens, base_path=None):
        super().__init__()
        import nemo.collections.asr as nemo_asr
        m = (nemo_asr.models.ASRModel.restore_from(base_path, map_location="cpu") if base_path
             else nemo_asr.models.ASRModel.from_pretrained(__import__("qafft.sources", fromlist=["hub_id"]).hub_id(BASE), map_location="cpu"))
        self.pre, self.enc, self.spec = m.preprocessor, m.encoder, m.spec_augmentation
        d = self.enc.d_model if hasattr(self.enc, "d_model") else 512
        self.up = nn.ConvTranspose1d(d, d, kernel_size=2, stride=2)   # 12.5 -> 25 frames/s
        self.out = nn.Linear(d, n_tokens + 1)                          # index n_tokens = blank
        del m

    def forward(self, wav, wl, train=False):
        feats, fl = self.pre(input_signal=wav, length=wl)
        if train and self.spec is not None:
            feats = self.spec(input_spec=feats, length=fl)
        h, hl = self.enc(audio_signal=feats, length=fl)             # (B, D, T)
        h = torch.relu(self.up(h)).transpose(1, 2)
        return self.out(h).log_softmax(-1), hl * 2


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, loader, blank, dev, max_batches=0):
    model.eval()
    err = tot = 0
    for b, (wav, wl, ids, il) in enumerate(loader):
        lp, hl = model(wav.to(dev), wl.to(dev))
        best = lp.argmax(-1).cpu()
        for k in range(best.size(0)):
            seq = torch.unique_consecutive(best[k, : hl[k]]).tolist()
            hyp = [x for x in seq if x != blank]
            ref = ids[k, : il[k]].tolist()
            err += edit_distance(hyp, ref); tot += len(ref)
        if max_batches and b + 1 >= max_batches:
            break
    model.train()
    return err / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="local .nemo file; default downloads the base model")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--amp", action="store_true", help="mixed precision; off by default, it produced NaN losses")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = json.load(open(os.path.join(ROOT, "qafft", "vocab.json")))["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}
    blank = len(tokens)

    def files(pred):
        out = []
        for d in sorted(glob.glob(os.path.join(a.data, "moshaf_*"))):
            mid = os.path.basename(d).split("_", 1)[1]
            if pred(mid):
                out += sorted(glob.glob(os.path.join(d, "*.parquet")))
        return out

    tr = Segments(files(lambda m: m not in TEST_MOSHAF | VAL_MOSHAF), vocab)
    va = Segments(files(lambda m: m in VAL_MOSHAF), vocab)
    te = Segments(files(lambda m: m in TEST_MOSHAF), vocab)
    if a.smoke and not len(va):
        va = tr
    print(f"segments: train {len(tr)} val {len(va)} test {len(te)}", flush=True)
    kw = dict(collate_fn=collate, num_workers=0 if a.smoke else 2)
    # Shuffle row groups, not single rows, so each parquet row group is read once.
    order = sorted(range(len(tr)), key=lambda k: (hash(tr.items[k][:2]) , k))
    tl = torch.utils.data.DataLoader(torch.utils.data.Subset(tr, order), batch_size=a.batch, shuffle=False, **kw)
    vl = torch.utils.data.DataLoader(va, batch_size=a.batch, **kw)

    model = QafModel(len(tokens), a.base).to(dev)
    head = list(model.up.parameters()) + list(model.out.parameters())
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": head, "lr": a.head_lr}], weight_decay=1e-3)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    use_amp = a.amp and dev == "cuda"
    scaler = torch.amp.GradScaler(dev, enabled=use_amp)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    step, best = 0, 1e9
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
        print("resumed at step", step, flush=True)

    model.train()
    t0, run, bad = time.time(), 0.0, 0
    while step < a.steps:
        for n, (wav, wl, ids, il) in enumerate(tl):
            wav = augment(wav, wl)
            with torch.autocast(dev, enabled=use_amp):
                lp, hl = model(wav.to(dev), wl.to(dev), train=True)
            loss = ctc(lp.float().transpose(0, 1), ids.to(dev), hl, il.to(dev)) / a.accum
            if not torch.isfinite(loss):
                bad += 1
                opt.zero_grad(set_to_none=True)
                if bad > 50:
                    raise SystemExit("loss is not finite for 50 batches; stopping")
                continue
            scaler.scale(loss).backward()
            run += loss.item() * a.accum
            if (n + 1) % a.accum:
                continue
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            if step % 20 == 0 or a.smoke:
                print(f"step {step} loss {run / (a.accum * (20 if not a.smoke else 1)):.3f} "
                      f"{(time.time() - t0) / step:.1f}s/step", flush=True)
                run = 0.0
            if step % a.eval_every == 0 or step >= a.steps:
                per = evaluate(model, vl, blank, dev, max_batches=2 if a.smoke else 60)
                print(f"step {step} val token error rate {per:.4f}", flush=True)
                ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                          step=step, best=min(best, per), tokens=tokens)
                torch.save(ck, last)
                if per < best:
                    best = per
                    torch.save(dict(model=model.state_dict(), tokens=tokens, step=step, per=per),
                               os.path.join(a.out, "best.pt"))
            if step >= a.steps:
                break
    if len(te):
        per = evaluate(model, torch.utils.data.DataLoader(te, batch_size=a.batch, **kw), blank, dev)
        print(f"held-out reciters token error rate {per:.4f}", flush=True)
        json.dump(dict(test_token_error_rate=per, step=step), open(os.path.join(a.out, "result.json"), "w"))


if __name__ == "__main__":
    main()
