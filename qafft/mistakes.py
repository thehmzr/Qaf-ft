"""Mistake generator: turns a correct token sequence into a plausibly wrong one.

Each edit copies a mistake learners actually make: a letter swapped for a
neighbour in the mouth or throat, a wrong short vowel, a madd cut short or
stretched, a dropped doubling, ghunnah or qalqalah. Used to make unlimited
"expected vs said" pairs for training and for testing the checker.
"""
import random

LETTERS = {  # commonly confused pairs
    "ح": "هخ", "ه": "ح", "ع": "ء", "ء": "ع", "ق": "ك", "ك": "ق", "ص": "س", "س": "صث",
    "ض": "دظ", "د": "ض", "ط": "ت", "ت": "ط", "ذ": "زد", "ز": "ذظ", "ث": "س", "ظ": "زذ",
    "خ": "حغ", "غ": "خ",
}
VOWELS = "َُِ"
MADD = "اۥۦ"
KINDS = ("letter", "vowel", "madd", "doubling", "ghunnah", "qalqalah")


def _split(tok):
    return tok[0], (int(tok[1:]) if len(tok) > 1 else 1)


def _join(ch, n):
    return ch if n == 1 else f"{ch}{n}"


def options(tok):
    """All single-token edits for one token: list of (kind, new_token or None to delete)."""
    ch, n = _split(tok)
    out = []
    if ch in LETTERS and n <= 2:
        out += [("letter", _join(c, n)) for c in LETTERS[ch]]
    if ch in VOWELS:
        out += [("vowel", c) for c in VOWELS if c != ch]
    if ch in MADD and n >= 2:
        out += [("madd", _join(ch, m)) for m in (2, 4, 6) if m != n and abs(m - n) >= 2]
        if n == 2:
            out.append(("madd", None))                       # long vowel read short
    if ch not in MADD and ch not in VOWELS and n == 2 and ch not in "نم":
        out.append(("doubling", ch))                         # shaddah dropped
    if (ch in "نم" and n >= 2) or ch in "ں۾":
        out.append(("ghunnah", "ن" if ch in "نں" else "م"))  # nasal sound not held
    if ch == "ڇ":
        out.append(("qalqalah", None))
    return out


def perturb(tokens, groups, rng, vocab=None, kind=None):
    """Apply one edit. Returns (new_tokens, new_groups, info) or None if nothing applies.
    info = dict(kind, position, group, was, now)."""
    cand = []
    for i, t in enumerate(tokens):
        for k, new in options(t):
            if kind and k != kind:
                continue
            if new is not None and vocab is not None and new not in vocab:
                continue
            if new is not None and ((i and tokens[i - 1][0] == new[0]) or (i + 1 < len(tokens) and tokens[i + 1][0] == new[0])):
                continue                                     # would merge with a neighbour
            cand.append((i, k, new))
    if not cand:
        return None
    # pick the kind first so rare kinds are not drowned out by vowels
    k = rng.choice(sorted({c[1] for c in cand}))
    i, k, new = rng.choice([c for c in cand if c[1] == k])
    toks, grps = list(tokens), list(groups)
    info = dict(kind=k, position=i, group=groups[i], was=tokens[i], now=new)
    if new is None:
        if i and i + 1 < len(toks) and toks[i - 1][0] == toks[i + 1][0]:
            return None
        del toks[i]; del grps[i]
    else:
        toks[i] = new
    return toks, grps, info
