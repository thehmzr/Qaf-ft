import random

import pytest

from qafft.mistakes import KINDS, options, perturb
from qafft.wordmap import word_tokens


def test_options():
    assert ("letter", "ه") in options("ح")
    assert set(options("َ")) == {("vowel", "ُ"), ("vowel", "ِ")}
    assert ("madd", None) in options("ا2") and ("madd", "ا4") in options("ا2")
    assert options("ا4") == [("madd", "ا2"), ("madd", "ا6")]
    assert ("doubling", "ب") in options("ب2")
    assert options("ں3") == [("ghunnah", "ن")]
    assert options("ڇ") == [("qalqalah", None)]


def one_change(a, b):
    """True if b is a with exactly one token replaced or removed."""
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    return len(b) == len(a) - 1 and any(a[:i] + a[i + 1:] == b for i in range(len(a)))


@pytest.mark.parametrize("ayah", [(1, 1), (2, 255), (36, 22), (44, 33), (112, 4)])
def test_perturb_changes_one_place(ayah):
    toks, groups, _ = word_tokens(*ayah)
    rng = random.Random(0)
    for _ in range(200):
        r = perturb(toks, groups, rng)
        if r is None:
            continue
        new, ng, info = r
        assert one_change(toks, new)
        assert len(ng) == len(new)
        i = info["position"]
        assert info["was"] == toks[i] and info["group"] == groups[i]
        if info["now"] is None:
            assert new == toks[:i] + toks[i + 1:]
        else:
            assert new[i] == info["now"] != toks[i]
        assert all(x[0] != y[0] for x, y in zip(new, new[1:]))   # still a valid folded sequence


def test_perturb_kind_and_vocab():
    toks, groups, _ = word_tokens(2, 255)
    rng = random.Random(1)
    for k in KINDS:
        r = perturb(toks, groups, rng, kind=k)
        assert r is None or r[2]["kind"] == k
    assert perturb(["ب"], [0], rng) is None
    # an edit whose result is not in the vocabulary is never chosen
    r = perturb(["ح"], [0], rng, vocab={"خ"})
    assert r[0] == ["خ"]
