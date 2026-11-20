"""Full-Quran training run: streams shards of the expert set one at a time.

  python train/full.py --out DIR [--steps N]

Each shard is downloaded, used once per pass and deleted, so disk use stays small.
Whole reciters are held out for validation and test (same split as the pilot).
"""
import argparse, collections, json, os, random, sys, threading, queue, time

import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
from pilot import Segments, collate, augment, QafModel, evaluate, TEST_MOSHAF, VAL_MOSHAF

REPO = "obadx/muaalem-annotated-v3"


def shard_lists():
    files = [s["rfilename"] for s in requests.get(f"https://huggingface.co/api/datasets/{REPO}").json()["siblings"]]
    by = collections.defaultdict(list)
    for f in files:
        if f.startswith("moshaf_") and f.endswith(".parquet") and "metadata" not in f:
            by[f.split("/")[0].split("_", 1)[1]].append(f)
    train = [f for m, fs in by.items() if m not in TEST_MOSHAF | VAL_MOSHAF for f in fs]
    held = lambda ids: [f for m in ids for f in (sorted(by[m])[0], sorted(by[m])[-1]) if m in by]
    return sorted(train), held(VAL_MOSHAF), held(TEST_MOSHAF)


def fetch(name, tmp):
    return hf_hub_download(REPO, name, repo_type="dataset", local_dir=tmp)


def batches(shards, vocab, tmp, batch, seed):
    """Yields training batches forever; the next shard downloads while the current one is used."""
    rng = random.Random(seed)
    q = queue.Queue(maxsize=1)

    def producer():
        while True:
            order = shards[:]
            rng.shuffle(order)
            for name in order:
                try:
                    q.put(fetch(name, tmp))
                except Exception as e:
                    print("skip shard", name, type(e).__name__, flush=True)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        path = q.get()
        ds = Segments([path], vocab, first_sura=1)
        # row groups in random order, rows shuffled within a window of groups
        groups = collections.defaultdict(list)
        for k, it in enumerate(ds.items):
            groups[it[1]].append(k)
        gids = list(groups); rng.shuffle(gids)
        buf = []
        for g in gids:
            ks = groups[g]; rng.shuffle(ks)
            for k in ks:
                buf.append(ds[k])
                if len(buf) == batch:
                    yield collate(buf); buf = []
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="/kaggle/temp/data")
    ap.add_argument("--base", default=None)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--hours", type=float, default=10.0, help="stop cleanly before the session limit")
    ap.add_argument("--dry", action="store_true", help="only iterate a few batches, no model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.tmp, exist_ok=True)
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = json.load(open(os.path.join(ROOT, "qafft", "vocab.json")))["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}
    blank = len(tokens)

    train, val_f, test_f = shard_lists()
    print(f"shards: train {len(train)} val {len(val_f)} test {len(test_f)}", flush=True)
    stream = batches(train, vocab, a.tmp, a.batch, seed=0)
    if a.dry:
        t0 = time.time()
        for n in range(6):
            wav, wl, ids, il = next(stream)
            print(n, tuple(wav.shape), il.tolist(), round(time.time() - t0, 1), flush=True)
        return
    hold = os.path.join(a.tmp, "held")
    va = Segments([fetch(f, hold) for f in val_f], vocab, first_sura=1)
    te = Segments([fetch(f, hold) for f in test_f], vocab, first_sura=1)
    print(f"segments: val {len(va)} test {len(te)}", flush=True)
    kw = dict(collate_fn=collate, num_workers=2, batch_size=a.batch)
    vl = torch.utils.data.DataLoader(va, **kw)

    model = QafModel(len(tokens), a.base).to(dev)
    head = list(model.up.parameters()) + list(model.out.parameters())
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": head, "lr": a.head_lr}], weight_decay=1e-3)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.1, 1 - s / max(1, a.steps)))
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    step, best, bad, run, t0 = 0, 1e9, 0, 0.0, time.time()
    model.train()
    while step < a.steps and time.time() - t0 < a.hours * 3600:
        for _ in range(a.accum):
            wav, wl, ids, il = next(stream)
            lp, hl = model(augment(wav, wl).to(dev), wl.to(dev), train=True)
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
            per = evaluate(model, vl, blank, dev, max_batches=80)
            print(f"step {step} val token error rate {per:.4f}", flush=True)
            if per < best:
                best = per
                torch.save(dict(model=model.state_dict(), tokens=tokens, step=step, per=per),
                           os.path.join(a.out, "best.pt"))
    torch.save(dict(model=model.state_dict(), tokens=tokens, step=step), os.path.join(a.out, "last.pt"))
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=dev) if os.path.exists(os.path.join(a.out, "best.pt")) else None
    if ck:
        model.load_state_dict(ck["model"])
    per = evaluate(model, torch.utils.data.DataLoader(te, **kw), blank, dev)
    print(f"held-out reciters token error rate {per:.4f} (best val {best:.4f}, steps {step})", flush=True)
    json.dump(dict(test_token_error_rate=per, best_val=best, step=step), open(os.path.join(a.out, "result.json"), "w"))


if __name__ == "__main__":
    main()
