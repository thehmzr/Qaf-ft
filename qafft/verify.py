"""Scores how well the audio fits each expected sound and each expected word."""
import torch
import torchaudio


def token_scores(logp, ids, blank):
    """logp: (T, V+1) log-probabilities. ids: expected token ids.
    Returns one score per expected token: the mean log-probability the model gives
    that token over the frames it is aligned to (0 = certain, very negative = unlikely)."""
    T = logp.shape[0]
    need = len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    if T < need:
        return None
    al, sc = torchaudio.functional.forced_align(logp.unsqueeze(0), torch.tensor([ids]), blank=blank)
    al, sc = al[0].tolist(), sc[0].tolist()
    out, pos, prev = [[] for _ in ids], -1, blank
    for t in range(T):
        a = al[t]
        if a != blank and (a != prev or (t and al[t - 1] == blank)):
            pos += 1
        if a != blank and 0 <= pos < len(ids):
            out[pos].append(sc[t])
        prev = a
    return [sum(v) / len(v) if v else -30.0 for v in out]


def group_scores(tok_scores, groups):
    """Weakest sound in each spoken group (word)."""
    out = {}
    for s, g in zip(tok_scores, groups):
        out[g] = min(out.get(g, 0.0), s)
    return out


def extra_sounds(logp, ids, groups, blank):
    """Sounds the model hears that the expected text does not contain.

    Alignment scoring only asks "is each expected sound there?". This asks the
    opposite: decode freely, line the result up with the expected tokens, and
    charge every unmatched heard sound to the word next to it. The charge is
    -log(1 - p), p being the model's confidence in the extra sound, so a faint
    blip costs little and a clear extra sound costs a lot.
    """
    import math
    from rapidfuzz.distance import Levenshtein
    best = logp.max(-1)
    heard, conf, prev = [], [], blank
    for t, (i, v) in enumerate(zip(best.indices.tolist(), best.values.tolist())):
        if i != blank and i != prev:
            heard.append(i); conf.append([v])
        elif i != blank:
            conf[-1].append(v)
        prev = i
    out = {}
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ids, heard):
        extra = range(j1, j2) if tag == "insert" else range(j1 + (i2 - i1), j2) if tag == "replace" else ()
        for j in extra:
            p = min(0.999999, math.exp(sum(conf[j]) / len(conf[j])))
            g = groups[max(0, min(len(groups) - 1, i1 - 1 if tag == "insert" else i2 - 1))]
            out[g] = max(out.get(g, 0.0), -math.log(1 - p))
    return out


def basic_classes(tokens):
    """Maps every token to a "words and vowels" class, ignoring tajweed detail.

    Kept: which letter, which short vowel, whether a long vowel is present, and doubling.
    Ignored: how many counts a madd or ghunnah is held, hidden versus plain noon and
    meem, and qalqalah. Returns (class index per token or None to drop, number of classes).
    """
    names, idx = {}, []
    for t in tokens:
        ch, n = t[0], (int(t[1:]) if len(t) > 1 else 1)
        if ch == "ڇ":
            idx.append(None); continue
        if ch in "اۥۦ":
            key = ch                      # any length of the same long vowel
        elif ch in "نں":
            key = "ن2" if (ch == "ن" and n >= 2) else "ن" if ch == "ن" else "ن2"
        elif ch in "م۾":
            key = "م2" if (ch == "م" and n >= 2) else "م" if ch == "م" else "م2"
        else:
            key = t
        idx.append(names.setdefault(key, len(names)))
    return idx, len(names)


def to_basic(logp, tokens, ids, groups):
    """Collapses frame probabilities and an expected sequence to the basic classes."""
    cls, k = basic_classes(tokens)
    p = logp.exp()
    out = torch.zeros(p.shape[0], k + 1)
    for i, c in enumerate(cls):
        out[:, k if c is None else c] += p[:, i]     # dropped tokens count as silence
    out[:, k] += p[:, len(tokens)]
    new_ids, new_groups = [], []
    for i, g in zip(ids, groups):
        c = cls[i]
        if c is None or (new_ids and new_ids[-1] == c):
            continue
        new_ids.append(c); new_groups.append(g)
    return out.clamp_min(1e-9).log(), new_ids, new_groups, k


def token_spans(logp, ids, blank):
    """Frames each expected token is aligned to (forced alignment). None if too short."""
    T = logp.shape[0]
    need = len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    if T < need or not ids:
        return None
    al, _ = torchaudio.functional.forced_align(logp.unsqueeze(0), torch.tensor([ids]), blank=blank)
    al = al[0].tolist()
    out, pos, prev = [[] for _ in ids], -1, blank
    for t in range(T):
        a = al[t]
        if a != blank and (a != prev or (t and al[t - 1] == blank)):
            pos += 1
        if a != blank and 0 <= pos < len(ids):
            out[pos].append(t)
        prev = a
    return out


def margin_scores(logp, ids, groups, blank, spans=None):
    """Competing-token margin per word.

    For each expected token: mean over its aligned frames of log p(expected) minus
    log p(best other non-blank token). A word gets the margin of its weakest token
    (negative = some other sound beats the expected one)."""
    spans = spans if spans is not None else token_spans(logp, ids, blank)
    if spans is None:
        return None
    lp = logp.clone()
    lp[:, blank] = float("-inf")
    out = {}
    for i, (e, fr) in enumerate(zip(ids, spans)):
        if not fr:
            m = -30.0
        else:
            x = lp[fr]
            own = x[:, e].clone()
            x[:, e] = float("-inf")
            m = float((own - x.max(-1).values).mean())
        g = groups[i]
        out[g] = min(out.get(g, float("inf")), m)
    return out


def _seq_loglik(logp, seqs, blank):
    """log P(seq | audio) for each sequence, by the CTC forward algorithm (batched)."""
    T = logp.shape[0]
    n = len(seqs)
    L = max(1, max(len(s) for s in seqs))
    tgt = torch.full((n, L), 0, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            tgt[i, : len(s)] = torch.tensor(s)
    lens = torch.tensor([len(s) for s in seqs])
    x = logp.unsqueeze(1).expand(T, n, logp.shape[1])
    nll = torch.nn.functional.ctc_loss(x, tgt, torch.full((n,), T, dtype=torch.long), lens,
                                       blank=blank, reduction="none", zero_infinity=False)
    return -nll


def gop_sf_scores(logp, ids, groups, blank, k=3, spans=None, pad=8):
    """Segmentation-free GOP per word for a CTC model.

    score = (log P(expected | audio) - max log P(competitor | audio)) / tokens in word,
    where competitors change one token of the word: substitute it with one of the top-k
    non-blank tokens by frame posterior over the word's aligned frames, delete it, or
    insert one of those tokens after it. Negative = some edit of the word explains the
    audio better than the expected text. For speed the forward pass runs on a window of
    the audio and sequence around the word (neighbouring words kept as context)."""
    spans = spans if spans is not None else token_spans(logp, ids, blank)
    if spans is None:
        return None
    T = logp.shape[0]
    order = []
    for i, g in enumerate(groups):
        if not order or order[-1][0] != g:
            order.append([g, i, i + 1])
        else:
            order[-1][2] = i + 1
    out = {}
    for w, (g, a, b) in enumerate(order):
        # context: one word either side
        ca = order[w - 1][1] if w > 0 else a
        cb = order[w + 1][2] if w + 1 < len(order) else b
        fr = [t for j in range(ca, cb) for t in spans[j]]
        wf = [t for j in range(a, b) for t in spans[j]]
        t0 = 0 if ca == 0 else max(0, min(fr) - pad) if fr else 0
        t1 = T if cb == len(ids) else min(T, max(fr) + pad + 1) if fr else T
        sub = logp[t0:t1]
        base = ids[ca:cb]
        if wf:
            post = logp[wf].exp().mean(0)
            post[blank] = 0
            cand = post.topk(min(k + 1, post.numel())).indices.tolist()
        else:
            cand = []
        seqs = [base]
        for j in range(a, b):
            r = j - ca
            seqs.append(base[:r] + base[r + 1:])                       # delete
            for c in cand[: k + 1]:
                if c != ids[j]:
                    seqs.append(base[:r] + [c] + base[r + 1:])         # substitute
                seqs.append(base[: r + 1] + [c] + base[r + 1:])        # insert after
        if a == ca:                                                     # insert at word start
            for c in cand[: k + 1]:
                seqs.append([c] + base)
        need = [len(s) + sum(1 for x, y in zip(s, s[1:]) if x == y) for s in seqs]
        keep = [s for s, nd in zip(seqs, need) if nd <= sub.shape[0]]
        if len(keep) < 2 or keep[0] is not base:
            out[g] = -30.0
            continue
        ll = _seq_loglik(sub, keep, blank)
        out[g] = float((ll[0] - ll[1:].max()) / (b - a))
    return out


def rank_badness(values, reference):
    """Maps badness values to the share of reference (correct-word) badness below them,
    so scores on different scales can be combined with max()."""
    import bisect
    ref = sorted(reference)
    n = max(1, len(ref))
    return [bisect.bisect_left(ref, v) / n for v in values]
