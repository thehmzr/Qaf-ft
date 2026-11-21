import random

import pytest

from qafft import quran_text
from qafft.phonetic import all_ayahs, fold_with_groups
from qafft.wordmap import (ayah_phonemes, display_letters, empty_words, group_words, map_tokens,
                           token_groups, word_tokens)

needs_text = pytest.mark.skipif(not quran_text.available(), reason="Quran text not fetched")


def word_of(s, a):
    toks, tw, n = word_tokens(s, a)
    return [" ".join(t for t, w in zip(toks, tw) if w == i) for i in range(n)]


def test_display_letters():
    assert display_letters("\u0627\u0654\u064eنزَلَ") == ["ء", "ن", "ز", "ل"]   # seat + hamza mark
    assert display_letters("أَنزَلَ") == ["ء", "ن", "ز", "ل"]        # precomposed
    assert display_letters("ٱلرَّحۡمَٰنِ") == ["ٱ", "ل", "ر", "ح", "م", "ا", "ن"]   # dagger alif
    assert display_letters("بِهِۦ") == ["ب", "ه", "ي"]                 # small ya


def test_map_tokens_merged_group():
    # two display words spoken as one group: tanween joined to the next word
    tokens, groups = fold_with_groups("ثَمَنَںںںقَلِۦۦلَن")
    tw = map_tokens(["ثَمَنࣰا", "قَلِيلًا"], tokens, groups)
    assert [t for t, w in zip(tokens, tw) if w == 0] == ["ث", "َ", "م", "َ", "ن", "َ", "ں3"]
    assert tw[-1] == 1


@needs_text
def test_word_split_15_7():
    w = word_of(15, 7)
    assert len(w) == 7 and w[0] == "ل َ و م َ ا2"
    assert group_words(15, 7)[:2] == [[0], [0]]


@needs_text
def test_word_split_27_20():
    w = word_of(27, 20)
    assert len(w) == 11 and w[3] == "م َ ا2 ل ِ ي َ"
    assert group_words(27, 20)[3:5] == [[3], [3]]


@needs_text
def test_word_split_36_22():
    w = word_of(36, 22)
    assert len(w) == 7 and w[0] == "و َ م َ ا2 ل ِ ي َ"
    assert group_words(36, 22)[:2] == [[0], [0]]


@needs_text
def test_word_split_44_33():
    words = quran_text.words(44, 33)
    assert len(words) == 7 and words[5].endswith("\u0627\u0652")   # the loose alif is joined back
    w = word_of(44, 33)
    assert w[5] == "ب َ ل َ ا4 ء ُ" and w[6] == "م4 ُ ب ِ ۦ4 ن"
    assert group_words(44, 33) == [[0, 1], [2], [3], [4], [5, 6]]


@needs_text
def test_random_ayahs_invariants():
    ayahs = list(all_ayahs())
    assert len(ayahs) == 6236
    for s, a in random.Random(1).sample(ayahs, 150):
        toks, tw, n = word_tokens(s, a)
        assert toks == fold_with_groups(ayah_phonemes(s, a))[0]
        assert len(tw) == len(toks) == len(token_groups(s, a))
        assert tw == sorted(tw) and 0 <= tw[0] and tw[-1] < n
        assert empty_words(tw, n) == []
        gw = group_words(s, a)
        assert sorted({w for ws in gw for w in ws}) == list(range(n))
