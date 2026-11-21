"""Legal reading choices within Hafs and best-of scoring across them.

The checker scores each word against one expected reading (default_moshaf). A reciter who
takes another legal choice (another free madd length, ghunnah of noon into lam/ra, seen or
saad in the few words that allow both) would be flagged although correct. Here a short list
of variant MoshafAttributes is phonetized (one attribute family at a time plus all short / all
long), identical token sequences are merged, and every printed word keeps its lowest badness
over the variants. Words are matched across variants through wordmap.map_tokens, so the
minimum is taken per printed word, never per token position.
"""
import difflib
from functools import lru_cache

from quran_transcript import quran_phonetizer

from qafft.phonetic import default_moshaf, fold_with_groups
from qafft.wordmap import map_tokens

# name -> MoshafAttributes fields changed from default_moshaf() (madd 4 everywhere)
VARIANTS = {
    "default": {},
    "monfasel_2": dict(madd_monfasel_len=2),
    "monfasel_5": dict(madd_monfasel_len=5),
    "mottasel_5": dict(madd_mottasel_len=5, madd_mottasel_waqf=5),
    "mottasel_6": dict(madd_mottasel_len=6, madd_mottasel_waqf=6),
    "aared_2": dict(madd_aared_len=2, madd_alleen_len=2),
    "aared_6": dict(madd_aared_len=6),
    "alleen_2": dict(madd_alleen_len=2),
    "ghonna_lam_raa": dict(ghonna_lam_and_raa="ghonna"),
    "saad": dict(yabsut="saad", bastah="saad", almusaytirun="saad", bimusaytir="saad"),
    "seen": dict(yabsut="seen", bastah="seen", almusaytirun="seen", bimusaytir="seen"),
    "all_short": dict(madd_monfasel_len=2, madd_mottasel_len=4, madd_mottasel_waqf=4,
                      madd_aared_len=2, madd_alleen_len=2),
    "all_long": dict(madd_monfasel_len=5, madd_mottasel_len=6, madd_mottasel_waqf=6,
                     madd_aared_len=6),
}


@lru_cache(maxsize=None)
def moshaf(name):
    return default_moshaf().model_copy(update=VARIANTS[name])


@lru_cache(maxsize=4096)
def span_variants(words, names=tuple(VARIANTS)):
    """words: tuple of printed words said on their own. Returns a list of distinct
    (name, tokens, word index of each token); the default reading comes first."""
    out, seen = [], set()
    for name in names:
        ph = quran_phonetizer(" ".join(words), moshaf(name), remove_spaces=False).phonemes
        toks, grps = fold_with_groups(ph)
        if tuple(toks) in seen:
            continue
        seen.add(tuple(toks))
        out.append((name, toks, map_tokens(list(words), toks, grps)))
    return out


def transfer(toks, grps, ref, alt):
    """Apply the differences between two readings of the same text (ref -> alt, token lists)
    to another token sequence `toks` (with groups) that follows ref closely, e.g. a clip or
    a planted mistake. Changed tokens take the group of the token they replace."""
    sm = difflib.SequenceMatcher(None, ref, alt, autojunk=False)
    edits = [(i1, i2, alt[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    if not edits:
        return list(toks), list(grps)
    pos = {}                       # ref index -> toks index, where they agree
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, toks, autojunk=False).get_opcodes():
        if op == "equal":
            pos.update({i1 + k: j1 + k for k in range(i2 - i1)})
    new_t, new_g = list(toks), list(grps)
    for i1, i2, rep in sorted(edits, reverse=True):
        span = [pos.get(i) for i in range(i1, i2)]
        if i1 == i2:               # insertion: needs the token before it in place
            if i1 - 1 not in pos:
                continue
            a = b = pos[i1 - 1] + 1
            g = new_g[a - 1]
        else:
            if None in span or span != list(range(span[0], span[0] + len(span))):
                continue           # region changed in toks (e.g. the planted mistake): keep it
            a, b = span[0], span[-1] + 1
            g = new_g[a]
        new_t[a:b] = rep
        new_g[a:b] = [g] * len(rep)
    return new_t, new_g


def best_of(scores):
    """scores: list of {word: badness} dicts (one per variant, None if it failed).
    Per word, the lowest badness over the variants that scored it."""
    out = {}
    for s in scores:
        if not s:
            continue
        for w, v in s.items():
            out[w] = min(out.get(w, v), v)
    return out
