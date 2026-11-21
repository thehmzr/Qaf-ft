"""Full-Quran training run: streams shards of the expert set one at a time.

  python train/full.py --out DIR [--steps N]
  python train/full.py --out DIR --init best.pt --phone phone_train.json --phone-tokens ayah_tokens.json \
      --steps 8000 --lr 3e-5 --head-lr 3e-5 --lr-floor 0 --avg-last 5

Each shard is downloaded, used once per pass and deleted, so disk use stays small.
Whole reciters are held out for validation and test (same split as the pilot).
With --phone, batches from selected phone clips are mixed in (--phone-frac of them).
"""
import argparse, collections, io, json, os, random, sys, threading, queue, time

import pyarrow.parquet as pq
import requests
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
from pilot import Segments, collate, augment, QafModel, evaluate, TEST_MOSHAF, VAL_MOSHAF

REPO = "obadx/muaalem-annotated-v3"
PHONE_REPO = "deepdml/tlog-clean-sf16k"


def shard_lists():
    files = [s["rfilename"] for s in requests.get(f"https://huggingface.co/api/datasets/{REPO}").json()["siblings"]]
    by = collections.defaultdict(list)
    for f in files:
        if f.startswith("moshaf_") and f.endswith(".parquet") and "metadata" not in f:
            by[f.split("/")[0].split("_", 1)[1]].append(f)
    train = [f for m, fs in by.items() if m not in TEST_MOSHAF | VAL_MOSHAF for f in fs]
    held = lambda ids: [f for m in ids for f in (sorted(by[m])[0], sorted(by[m])[-1]) if m in by]
    return sorted(train), held(VAL_MOSHAF), held(TEST_MOSHAF)


def fetch(name, tmp, repo=REPO):
    return hf_hub_download(repo, name, repo_type="dataset", local_dir=tmp)


class PhoneClips(torch.utils.data.Dataset):
    """Selected clips of one phone shard. Labels are the expected tokens of the ayah."""

    def __init__(self, path, wanted, tokens_by_ayah, vocab, max_sec=20.0):
        self.path, self.items = path, []
        t = pq.ParquetFile(path).read(columns=["audio.path", "duration"]).to_pylist()
        for i, r in enumerate(t):
            key = wanted.get(r["audio"]["path"])
            if key is None or r["duration"] > max_sec:
                continue
            toks = tokens_by_ayah[key]
            if len(toks) > 25 * r["duration"] - 2:
                continue
            self.items.append((i, [vocab[x] for x in toks]))
        self._audio = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        if self._audio is None:
            self._audio = pq.ParquetFile(self.path).read(columns=["audio"]).column("audio")
        i, ids = self.items[k]
        wav, sr = sf.read(io.BytesIO(self._audio[i].as_py()["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, sr
        return torch.from_numpy(wav), torch.tensor(ids)


def stream(shards, make_ds, tmp, batch, seed, repo=REPO, keep=()):
    """Yields training batches forever; the next shard downloads while the current one is used."""
    rng = random.Random(seed)
    q = queue.Queue(maxsize=1)

    def producer():
        while True:
            order = shards[:]
            rng.shuffle(order)
            for name in order:
                try:
                    q.put((name, fetch(name, tmp, repo)))
                except Exception as e:
                    print("skip shard", name, type(e).__name__, flush=True)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        name, path = q.get()
        ds = make_ds(name, path)
        ks = list(range(len(ds)))
        if hasattr(ds, "items") and ds.items and len(ds.items[0]) == 5:
            # expert set: row groups in random order, rows shuffled inside each group
            groups = collections.defaultdict(list)
            for k, it in enumerate(ds.items):
                groups[it[1]].append(k)
            gids = list(groups); rng.shuffle(gids)
            ks = []
            for g in gids:
                rng.shuffle(groups[g]); ks += groups[g]
        else:
            rng.shuffle(ks)
        buf = []
        for k in ks:
            buf.append(ds[k])
            if len(buf) == batch:
                yield collate(buf); buf = []
        del ds
        if name not in keep:
            try:
                os.remove(path)
            except OSError:
                pass


def narrowband(wav, p):
    """Some clips go through 8 kHz and back, like a headset mic. Timing is unchanged."""
    if p <= 0:
        return wav
    for b in range(wav.size(0)):
        if random.random() < p:
            low = torchaudio.functional.resample(wav[b], 16000, 8000)
            wav[b] = torchaudio.functional.resample(low, 8000, 16000)[: wav.size(1)]
    return wav


def average(paths):
    """Mean of the weights in several checkpoints."""
    avg = None
    for p in paths:
        sd = torch.load(p, map_location="cpu")["model"]
        if avg is None:
            avg = {k: v.clone().double() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    avg[k] += v.double()
    return {k: (v / len(paths)).float() if v.is_floating_point() else v for k, v in avg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="/kaggle/temp/data")
    ap.add_argument("--base", default=None)
    ap.add_argument("--init", default=None, help="start from this checkpoint instead of a new head")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--lr-floor", type=float, default=0.1, help="lr decays linearly to this share of the start")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--avg-last", type=int, default=0, help="average this many checkpoints from the end")
    ap.add_argument("--phone", default=None, help="json from data_tools/select_phone.py")
    ap.add_argument("--phone-tokens", default=None, help="expected tokens per ayah, keyed sura:aya")
    ap.add_argument("--phone-frac", type=float, default=0.5)
    ap.add_argument("--narrowband", type=float, default=0.15, help="share of phone clips passed through 8 kHz")
    ap.add_argument("--hours", type=float, default=10.0, help="stop cleanly before the session limit")
    ap.add_argument("--dry", action="store_true", help="only iterate a few batches, no model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.tmp, exist_ok=True)
    torch.manual_seed(0); random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = json.load(open(os.path.join(ROOT, "qafft", "vocab.json")))["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}
    blank = len(tokens)

    train, val_f, test_f = shard_lists()
    print(f"shards: train {len(train)} val {len(val_f)} test {len(test_f)}", flush=True)
    expert = stream(train, lambda n, p: Segments([p], vocab, first_sura=1), a.tmp, a.batch, seed=0)

    phone = phone_val = None
    if a.phone:
        sel = json.load(open(a.phone))
        by_ayah = json.load(open(a.phone_tokens))
        wanted = lambda n: {c[0]: c[1] for c in sel["train"][n]}
        phone = stream(sorted(sel["train"]), lambda n, p: PhoneClips(p, wanted(n), by_ayah, vocab),
                       os.path.join(a.tmp, "phone"), a.batch, seed=1, repo=PHONE_REPO)
        pv = []
        for n, clips in sorted(sel["val"].items()):
            pv.append(PhoneClips(fetch(n, os.path.join(a.tmp, "phone_val"), PHONE_REPO),
                                 {c[0]: c[1] for c in clips}, by_ayah, vocab))
        phone_val = torch.utils.data.ConcatDataset(pv)
        print(f"phone: {sum(len(v) for v in sel['train'].values())} train clips in {len(sel['train'])} shards, "
              f"{len(phone_val)} val clips", flush=True)

    if a.dry:
        t0 = time.time()
        for n in range(6):
            src = phone if phone is not None and n % 2 else expert
            wav, wl, ids, il = next(src)
            print(n, "phone" if src is phone else "expert", tuple(wav.shape), il.tolist(), round(time.time() - t0, 1), flush=True)
        return
    hold = os.path.join(a.tmp, "held")
    va = Segments([fetch(f, hold) for f in val_f], vocab, first_sura=1)
    te = Segments([fetch(f, hold) for f in test_f], vocab, first_sura=1)
    print(f"segments: val {len(va)} test {len(te)}", flush=True)
    kw = dict(collate_fn=collate, num_workers=2, batch_size=a.batch)
    vl = torch.utils.data.DataLoader(va, **kw)
    pvl = torch.utils.data.DataLoader(phone_val, **kw) if phone_val is not None else None

    model = QafModel(len(tokens), a.base).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        assert ck["tokens"] == tokens, "token list differs from the checkpoint"
        model.load_state_dict(ck["model"])
        print("init from", a.init, "step", ck.get("step"), flush=True)
    head = list(model.up.parameters()) + list(model.out.parameters())
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": head, "lr": a.head_lr}], weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warm) * max(a.lr_floor, 1 - s / max(1, a.steps)))
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    def check(tag):
        per = evaluate(model, vl, blank, dev, max_batches=80)
        msg = f"{tag} val token error rate {per:.4f}"
        pper = None
        if pvl is not None:
            pper = evaluate(model, pvl, blank, dev)
            msg += f"  phone val {pper:.4f}"
        print(msg, flush=True)
        return per, pper

    if a.init:
        check("start")
    step, best, bad, run, t0 = 0, 1e9, 0, 0.0, time.time()
    snaps = []
    model.train()
    while step < a.steps and time.time() - t0 < a.hours * 3600:
        for _ in range(a.accum):
            use_phone = phone is not None and random.random() < a.phone_frac
            wav, wl, ids, il = next(phone if use_phone else expert)
            wav = augment(wav, wl)
            if use_phone:
                wav = narrowband(wav, a.narrowband)
            lp, hl = model(wav.to(dev), wl.to(dev), train=True)
            loss = ctc(lp.float().transpose(0, 1), ids.to(dev), hl, il.to(dev)) / a.accum
            if not torch.isfinite(loss):
                bad += 1
                if bad > 50:
                    raise SystemExit("loss is not finite for 50 batches; stopping")
                continue
            loss.backward(); run += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
        if step % 100 == 0:
            print(f"step {step} loss {run / 100:.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)
            run = 0.0
        if step % a.eval_every == 0:
            per, pper = check(f"step {step}")
            if per < best:
                best = per
                torch.save(dict(model=model.state_dict(), tokens=tokens, step=step, per=per),
                           os.path.join(a.out, "best.pt"))
            if a.avg_last:
                p = os.path.join(a.out, f"snap_{step}.pt")
                torch.save(dict(model=model.state_dict(), tokens=tokens, step=step), p)
                snaps.append(p)
                while len(snaps) > a.avg_last:
                    os.remove(snaps.pop(0))
    torch.save(dict(model=model.state_dict(), tokens=tokens, step=step), os.path.join(a.out, "last.pt"))

    result = dict(best_val=best, step=step)
    if snaps:
        model.load_state_dict(average(snaps))
        per, pper = check(f"average of {len(snaps)}")
        torch.save(dict(model=model.state_dict(), tokens=tokens, step=step, per=per,
                        averaged=[os.path.basename(p) for p in snaps]), os.path.join(a.out, "avg.pt"))
        result.update(avg_val=per, avg_phone_val=pper)
        for p in snaps:
            os.remove(p)
    else:
        ck = torch.load(os.path.join(a.out, "best.pt"), map_location=dev) if os.path.exists(os.path.join(a.out, "best.pt")) else None
        if ck:
            model.load_state_dict(ck["model"])
    per = evaluate(model, torch.utils.data.DataLoader(te, **kw), blank, dev)
    print(f"held-out reciters token error rate {per:.4f} (best val {best:.4f}, steps {step})", flush=True)
    result["test_token_error_rate"] = per
    json.dump(result, open(os.path.join(a.out, "result.json"), "w"))


if __name__ == "__main__":
    main()
