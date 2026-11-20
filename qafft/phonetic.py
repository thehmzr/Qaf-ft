"""Quran text -> sound-level tokens.

The phonetic script writes a lengthened sound as a repeated symbol (a 4-count
madd is four alif symbols). Repeats are folded into one token "symbol+count",
so a token sequence never has two equal neighbours and stays short enough for
a 12.5 frames/second encoder.
"""
import json
from functools import lru_cache
from itertools import groupby


MAX_RUN = 8


def default_moshaf(madd=4):
    from quran_transcript import MoshafAttributes
    return MoshafAttributes(rewaya="hafs", madd_monfasel_len=min(madd, 5), madd_mottasel_len=max(madd, 4),
                            madd_mottasel_waqf=max(madd, 4), madd_aared_len=madd)


def fold(phonemes: str):
    """'ررَحِۦۦۦۦم' -> ['ر2', 'َ', 'ح', 'ِ', 'ۦ4', 'م']"""
    out = []
    for ch, grp in groupby(phonemes.replace(" ", "")):
        n = min(len(list(grp)), MAX_RUN)
        out.append(ch if n == 1 else f"{ch}{n}")
    return out


def unfold(tokens):
    return "".join(t[0] * (int(t[1:]) if len(t) > 1 else 1) for t in tokens)


@lru_cache(maxsize=None)
def ayah_uthmani(sura: int, aya: int) -> str:
    from quran_transcript import Aya
    return Aya(sura, aya).get().uthmani


def ayah_tokens(sura: int, aya: int, moshaf=None):
    from quran_transcript import quran_phonetizer
    ph = quran_phonetizer(ayah_uthmani(sura, aya), moshaf or default_moshaf(), remove_spaces=True).phonemes
    return fold(ph)


def all_ayahs():
    from quran_transcript import Aya
    for s in range(1, 115):
        for a in range(1, Aya(s, 1).get().num_ayat_in_sura + 1):
            yield s, a


def build_vocab(path):
    """Token list over the whole Quran for madd choices 2, 4 and 6."""
    seen = {}
    for madd in (2, 4, 6):
        m = default_moshaf(madd)
        for s, a in all_ayahs():
            for t in ayah_tokens(s, a, m):
                seen[t] = seen.get(t, 0) + 1
    vocab = sorted(seen, key=lambda t: (t[0], len(t), t))
    json.dump({"tokens": vocab, "counts": seen}, open(path, "w"), ensure_ascii=False, indent=0)
    return vocab
