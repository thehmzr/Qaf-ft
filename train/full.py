"""Full-Quran training run: streams shards of the expert set one at a time.

  python train/full.py --out DIR [--steps N]
  python train/full.py --out DIR --init best.pt --phone phone_train.json --phone-tokens phone_tokens.json \
      --steps 12000 --lr 3e-5 --head-lr 1e-4 --sched cosine --lr-floor 0 --ema 0.999 --degrade 0.3

Each shard is downloaded, used once per pass and deleted, so disk use stays small.
Whole reciters are held out for validation and test (same split as the pilot); reciter 25.0
is left out. With --phone, batches of selected phone clips are mixed in (--phone-frac of them).
A run resumes from --out/last.pt if it exists.
"""
import argparse, collections, copy, io, json, math, os, random, sys, threading, queue, time

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
EXCLUDE_MOSHAF = {"25.0"}      # badly cut, left out on its authors' advice


def shard_lists():
    files = [s["rfilename"] for s in requests.get(f"https://huggingface.co/api/datasets/{REPO}").json()["siblings"]]
    by = collections.defaultdict(list)
    for f in files:
        if f.startswith("moshaf_") and f.endswith(".parquet") and "metadata" not in f:
            by[f.split("/")[0].split("_", 1)[1]].append(f)
    train = [f for m, fs in by.items() if m not in TEST_MOSHAF | VAL_MOSHAF | EXCLUDE_MOSHAF for f in fs]
    held = lambda ids: [f for m in ids for f in (sorted(by[m])[0], sorted(by[m])[-1]) if m in by]
    return sorted(train), held(VAL_MOSHAF), held(TEST_MOSHAF)


def fetch(name, tmp, repo=REPO):
    return hf_hub_download(repo, name, repo_type="dataset", local_dir=tmp)


class PhoneClips(torch.utils.data.Dataset):
    """Selected clips of one phone shard, kept in memory. Labels are the expected tokens of the ayah."""

    def __init__(self, path, wanted, tokens_by_ayah, vocab, max_sec=20.0):
        pf = pq.ParquetFile(path)
        rows, self.ids = [], []
        for i, r in enumerate(pf.read(columns=["audio.path", "duration"]).to_pylist()):
            key = wanted.get(r["audio"]["path"])
            if key is None or r["duration"] > max_sec:
                continue
            toks = tokens_by_ayah[key]
            if len(toks) > 25 * r["duration"] - 2:
                continue
            rows.append(i); self.ids.append([vocab[x] for x in toks])
        # only the selected clips' bytes stay in memory
        self.audio = pf.read(columns=["audio"]).column("audio").take(rows) if rows else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        wav, sr = sf.read(io.BytesIO(self.audio[k].as_py()["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, sr
        return torch.from_numpy(wav), torch.tensor(self.ids[k])


class ShardFiles:
    """Downloads shards in a background thread, in shuffled passes, one ahead of use.
    A file is deleted when nothing queued still needs it."""

    def __init__(self, names, tmp, seed, repo=REPO, keep=()):
        self.q, self.lock, self.users = queue.Queue(maxsize=1), threading.Lock(), collections.Counter()
        self.keep, rng = set(keep), random.Random(seed)

        def producer():
            while True:
                order = names[:]
                rng.shuffle(order)
                for name in order:
                    try:
                        with self.lock:
                            path = fetch(name, tmp, repo)
                            self.users[path] += 1
                    except Exception as e:
                        print("skip shard", name, type(e).__name__, flush=True)
                        continue
                    self.q.put((name, path))

        threading.Thread(target=producer, daemon=True).start()

    def get(self):
        return self.q.get()

    def done(self, name, path):
        with self.lock:
            self.users[path] -= 1
            if self.users[path] <= 0 and name not in self.keep:
                try:
                    os.remove(path)
                except OSError:
                    pass


def expert_batches(files, vocab, batch, seed):
    """Expert batches forever: row groups in random order, rows shuffled inside each group."""
    rng = random.Random(seed)
    while True:
        name, path = files.get()
        ds = Segments([path], vocab, first_sura=1)
        groups = collections.defaultdict(list)
        for k, it in enumerate(ds.items):
            groups[it[1]].append(k)
        gids = list(groups); rng.shuffle(gids)
        buf = []
        for g in gids:
            rng.shuffle(groups[g])
            for k in groups[g]:
                buf.append(ds[k])
                if len(buf) == batch:
                    yield collate(buf); buf = []
        del ds
        files.done(name, path)


def phone_batches(files, make_ds, n_shards, batch, seed, pool=4):
    """Phone batches forever, drawn at random across a pool of open shards."""
    rng = random.Random(seed)
    open_ds, empty = [], 0

    def refill():
        nonlocal empty
        while len(open_ds) < pool:
            name, path = files.get()
            ds = make_ds(name, path)
            files.done(name, path)          # clips are in memory now
            if len(ds) == 0:
                empty += 1
                if empty > 2 * n_shards:
                    raise SystemExit("no usable phone clips in any shard")
                continue
            empty = 0
            left = list(range(len(ds))); rng.shuffle(left)
            open_ds.append([ds, left])

    while True:
        refill()
        buf = []
        while len(buf) < batch:
            j = rng.choices(range(len(open_ds)), weights=[len(x[1]) for x in open_ds])[0]
            ds, left = open_ds[j]
            buf.append(ds[left.pop()])
            if not left:
                open_ds.pop(j); refill()
        yield collate(buf)


def degrade(wav, p):
    """For expert clips: on a share p of the batch, make the audio sound like a phone.
    Each picked clip gets one or more of: a lower sample rate and back (8, 11.025 or 12 kHz,
    so nothing above 4 to 6 kHz is left), mild clipping, room echo.
    None of these changes timing, so the labels stay right."""
    if p <= 0:
        return wav
    n = wav.size(1)
    for b in range(wav.size(0)):
        if random.random() >= p:
            continue
        x = wav[b]
        if random.random() < 0.5:
            sr = random.choice((8000, 11025, 12000))
            x = torchaudio.functional.resample(torchaudio.functional.resample(x, 16000, sr), sr, 16000)[:n]
            if x.numel() < n:
                x = torch.nn.functional.pad(x, (0, n - x.numel()))
        if random.random() < 0.3:
            peak = x.abs().max().clamp_min(1e-4)
            g = random.uniform(2, 5) / peak
            x = (x * g).clamp(-1, 1) / g
        if random.random() < 0.5:
            rt60 = random.uniform(0.2, 0.6)
            t = torch.arange(int(0.3 * 16000)) / 16000
            rir = torch.randn_like(t) * torch.exp(-6.9 * t / rt60)
            rir[0] = 1.0
            rir = rir / rir.norm()
            wet = torchaudio.functional.fftconvolve(x, rir)[:n]
            mix = random.uniform(0.2, 0.5)
            x = (1 - mix) * x + mix * wet * (x.norm() / wet.norm().clamp_min(1e-6))
        wav[b] = x
    return wav


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
        if e.dtype.is_floating_point:
            e.mul_(decay).add_(m.detach(), alpha=1 - decay)
        else:
            e.copy_(m)


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="/kaggle/temp/data")
    ap.add_argument("--base", default=None)
    ap.add_argument("--init", default=None, help="start from this checkpoint instead of a new head")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="encoder")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--sched", choices=["linear", "cosine"], default="linear")
    ap.add_argument("--lr-floor", type=float, default=0.1, help="lr decays to this share of the start")
    ap.add_argument("--ema", type=float, default=0.0, help="decay of a running weight average; 0 = off")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--phone", default=None, help="json from data_tools/select_phone.py")
    ap.add_argument("--phone-tokens", default=None, help="expected tokens per ayah, keyed sura:aya")
    ap.add_argument("--phone-frac", type=float, default=0.5)
    ap.add_argument("--degrade", type=float, default=0.0, help="share of expert clips made to sound like a phone")
    ap.add_argument("--studio-margin", type=float, default=0.003,
                    help="best.pt must keep studio validation error within this of the start")
    ap.add_argument("--hours", type=float, default=10.0, help="training time, counted from the start of the script")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="only iterate a few batches, no model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.tmp, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = json.load(open(os.path.join(ROOT, "qafft", "vocab.json")))["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}
    blank = len(tokens)

    train, val_f, test_f = shard_lists()
    print(f"shards: train {len(train)} val {len(val_f)} test {len(test_f)}", flush=True)
    expert = expert_batches(ShardFiles(train, a.tmp, seed=a.seed), vocab, a.batch, seed=a.seed)

    phone = phone_val = None
    if a.phone:
        sel = json.load(open(a.phone))
        by_ayah = json.load(open(a.phone_tokens))
        assert sel["train"], "the phone selection has no training clips"
        wanted = lambda n: {c[0]: c[1] for c in sel["train"][n]}
        names = sorted(sel["train"])
        files = ShardFiles(names, os.path.join(a.tmp, "phone"), seed=a.seed + 1, repo=PHONE_REPO)
        phone = phone_batches(files, lambda n, p: PhoneClips(p, wanted(n), by_ayah, vocab),
                              len(names), a.batch, seed=a.seed + 1)
        pv = []
        for n, clips in sorted(sel.get("val", {}).items()):
            if clips:
                pv.append(PhoneClips(fetch(n, os.path.join(a.tmp, "phone_val"), PHONE_REPO),
                                     {c[0]: c[1] for c in clips}, by_ayah, vocab))
        pv = [d for d in pv if len(d)]
        phone_val = torch.utils.data.ConcatDataset(pv) if pv else None
        print(f"phone: {sum(len(v) for v in sel['train'].values())} train clips in {len(names)} shards, "
              f"{len(phone_val) if phone_val else 0} val clips", flush=True)

    if a.dry:
        t0 = time.time()
        for n in range(6):
            src = phone if phone is not None and n % 2 else expert
            wav, wl, ids, il = next(src)
            if src is expert:
                wav = degrade(wav, a.degrade)
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

    def shape(s):
        warm = min(1.0, (s + 1) / a.warm)
        x = min(1.0, s / max(1, a.steps))
        decay = 0.5 * (1 + math.cos(math.pi * x)) if a.sched == "cosine" else 1 - x
        return warm * (a.lr_floor + (1 - a.lr_floor) * decay)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, shape)
    ema = None
    if a.ema:
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    def check(tag, m):
        per = evaluate(m, vl, blank, dev, max_batches=80)
        pper = evaluate(m, pvl, blank, dev) if pvl is not None else None
        print(f"{tag} val token error rate {per:.4f}" + (f"  phone val {pper:.4f}" if pper is not None else ""), flush=True)
        return per, pper

    step, run, bad = 0, 0.0, 0
    best = None                         # (score, studio) of the best saved model
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        if "opt" in ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            if ema is not None and ck.get("ema"):
                ema.load_state_dict(ck["ema"])
            step, best, start = ck["step"], ck.get("best"), ck.get("start")
            print("resumed at step", step, flush=True)
    if step == 0:
        start = check("start", model) if a.init else (None, None)
        if a.init:
            best = (start[1] if start[1] is not None else start[0], start[0])

    def score(per, pper):
        return pper if pper is not None else per

    def save_last():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        ema=ema.state_dict() if ema is not None else None, step=step, best=best, start=start,
                        tokens=tokens), last + ".tmp")
        os.replace(last + ".tmp", last)

    t0 = time.time()
    model.train()
    while step < a.steps and time.time() - t_start < a.hours * 3600:
        for _ in range(a.accum):
            use_phone = phone is not None and random.random() < a.phone_frac
            wav, wl, ids, il = next(phone if use_phone else expert)
            if not use_phone:
                wav = degrade(augment(wav, wl), a.degrade)
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
        if ema is not None:
            ema_update(ema, model, min(a.ema, (1 + step) / (10 + step)))
        if step % 100 == 0:
            print(f"step {step} loss {run / 100:.3f} {(time.time() - t0) / step:.2f}s/step", flush=True)
            run = 0.0
        if step % a.eval_every == 0:
            for tag, m in (("model", model), ("ema", ema)):
                if m is None:
                    continue
                per, pper = check(f"step {step} {tag}", m)
                s = score(per, pper)
                studio_ok = start[0] is None or per <= start[0] + a.studio_margin
                if studio_ok and (best is None or s < best[0]):
                    best = (s, per)
                    torch.save(dict(model=m.state_dict(), tokens=tokens, step=step, per=per, phone_per=pper, kind=tag),
                               os.path.join(a.out, "best.pt"))
                    print(f"  saved best.pt ({tag})", flush=True)
            model.train()
        if step % a.save_every == 0:
            save_last()
    save_last()
    if ema is not None:
        torch.save(dict(model=ema.state_dict(), tokens=tokens, step=step), os.path.join(a.out, "ema.pt"))

    tl = torch.utils.data.DataLoader(te, **kw)
    result = dict(step=step, start_val=start[0], start_phone_val=start[1], best=best)
    for tag, path in (("best", "best.pt"), ("ema", "ema.pt")):
        p = os.path.join(a.out, path)
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=dev)["model"])
            per = evaluate(model, tl, blank, dev)
            pper = evaluate(model, pvl, blank, dev) if pvl is not None else None
            print(f"{tag}: held-out reciters token error rate {per:.4f}"
                  + (f", phone val {pper:.4f}" if pper is not None else ""), flush=True)
            result[f"test_{tag}"] = per; result[f"phone_val_{tag}"] = pper
    if not os.path.exists(os.path.join(a.out, "best.pt")):
        print("no checkpoint beat the start on validation", flush=True)
    json.dump(result, open(os.path.join(a.out, "result.json"), "w"))


if __name__ == "__main__":
    main()
