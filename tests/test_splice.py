import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_tools"))
from splice import crossfade_concat, cuts_from_frames, deletion, repetition, swap, true_tokens

SR = 1000          # 10 ms fade = 10 samples
WORDS = ["بِسمِ", "للَااهِ", "ررَحمَاانِ", "ررَحِۦۦم"]
CUTS = [0, 100, 250, 400, 600]


def clip(level=1.0):
    """each word a constant level so the pieces are easy to tell apart"""
    w = np.zeros(CUTS[-1], np.float32)
    for i in range(len(WORDS)):
        w[CUTS[i]: CUTS[i + 1]] = level * (i + 1)
    return w


def test_crossfade_length_and_ramp():
    a, b = np.ones(50, np.float32), np.zeros(40, np.float32)
    out, joins = crossfade_concat([a, b], SR, 10)
    assert len(out) == 50 + 40 - 10
    assert joins == [45]
    fade = out[40:50]
    assert np.all(np.diff(fade) < 0) and fade[0] < 1 and fade[-1] > 0
    assert np.all(out[:40] == 1) and np.all(out[50:] == 0)


def test_crossfade_keeps_constant_signal():
    out, _ = crossfade_concat([np.full(30, .5, np.float32)] * 3, SR, 10)
    assert len(out) == 70
    assert np.allclose(out, .5)


def test_deletion():
    audio, words, joins = deletion(clip(), WORDS, CUTS, 1, SR)
    assert len(audio) == 600 - 150 - 10
    assert words == ["بِسمِ", "ررَحمَاانِ", "ررَحِۦۦم"]
    assert not np.any(audio == 2)          # word 2 gone
    assert len(joins) == 1


def test_repetition():
    audio, words, joins = repetition(clip(), WORDS, CUTS, 2, SR)
    assert len(audio) == 600 + 150 - 20
    assert words == WORDS[:3] + [WORDS[2]] + WORDS[3:]
    assert np.sum(np.isclose(audio, 3)) == 2 * 150 - 10 - 10   # one overlap inside the copies, one mixed with the next word
    assert len(joins) == 2


def test_swap_and_true_tokens():
    donor = clip(10.0)
    audio, words, joins = swap(clip(), WORDS, CUTS, 1, donor, WORDS, CUTS, 2, SR)
    assert len(audio) == 600 - 150 + 150 - 20
    assert words == ["بِسمِ", "ررَحمَاانِ", "ررَحمَاانِ", "ررَحِۦۦم"]
    assert np.any(audio == 30) and not np.any(audio == 2)
    toks, grps = true_tokens(words)
    assert toks[:4] == ["ب", "ِ", "س", "م"]
    assert grps[-1] == 3 and sorted(set(grps)) == [0, 1, 2, 3]


def test_true_tokens_merge_across_words():
    # a run across a word boundary folds into one token owned by the first word
    toks, grps = true_tokens(["مِن", "ننَاس"])
    assert toks == ["م", "ِ", "ن3", "َ", "ا", "س"]
    assert grps == [0, 0, 0, 1, 1, 1]


def test_only_inner_words():
    import pytest
    for k in (0, 3):
        with pytest.raises(ValueError):
            deletion(clip(), WORDS, CUTS, k, SR)


def test_cuts_from_frames():
    # words at frames 0-4, 10-14, 15-19; 25 fps, 16 kHz
    cuts = cuts_from_frames([0, 10, 15], [4, 14, 19], 16000)
    assert cuts[0] == 0 and cuts[-1] == 16000
    assert cuts[1] == round((5 / 25 + 10 / 25) / 2 * 16000)   # middle of the gap
    assert cuts[2] == round(15 / 25 * 16000)                  # no gap: start of next word
