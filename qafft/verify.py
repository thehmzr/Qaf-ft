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
