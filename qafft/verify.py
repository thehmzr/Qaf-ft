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
