import random

from qafft.phonetic import MAX_RUN, fold, fold_with_groups, unfold
from qafft.wordmap import ayah_phonemes


def test_fold_example():
    assert fold("ررَحِۦۦۦۦم") == ["ر2", "َ", "ح", "ِ", "ۦ4", "م"]


def test_fold_unfold_round_trip_random():
    rng = random.Random(0)
    for _ in range(300):
        chars = [rng.choice("abc")]
        for _ in range(rng.randint(0, 20)):
            chars.append(rng.choice([c for c in "abc" if c != chars[-1]]))
        s = "".join(c * rng.randint(1, MAX_RUN) for c in chars)
        toks = fold(s)
        assert unfold(toks) == s
        assert all(x[0] != y[0] for x, y in zip(toks, toks[1:]))


def test_fold_ignores_spaces_and_caps_runs():
    assert fold("ab ba") == ["a", "b2", "a"]
    assert fold("a" * (MAX_RUN + 3)) == [f"a{MAX_RUN}"]


def test_fold_with_groups_small():
    toks, groups = fold_with_groups("ab bc")
    assert toks == ["a", "b2", "c"]
    assert groups == [0, 0, 1]                     # a run belongs to the group it starts in


def test_fold_with_groups_matches_fold_on_real_ayahs():
    for s, a in [(1, 1), (2, 255), (15, 7), (44, 33), (112, 4)]:
        ph = ayah_phonemes(s, a)
        toks, groups = fold_with_groups(ph)
        assert toks == fold(ph)
        assert unfold(toks) == ph.replace(" ", "")
        assert groups == sorted(groups)
        assert set(groups) <= set(range(len(ph.split())))
