import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
from qafft.verify import gop_sf_scores, margin_scores, rank_badness, token_spans

V = 5          # tokens 0..4, blank = 5
BLANK = V


def frames(path, p=0.9):
    lp = torch.full((len(path), V + 1), (1 - p) / V)
    for t, k in enumerate(path):
        lp[t, k] = p
    return lp.log()


AUDIO = [0, 0, BLANK, 1, 1, BLANK, 2, 2, BLANK, 3, 3, BLANK]
GROUPS = [0, 0, 1, 1]


def test_token_spans():
    sp = token_spans(frames(AUDIO), [0, 1, 2, 3], BLANK)
    assert sp == [[0, 1], [3, 4], [6, 7], [9, 10]]
    assert token_spans(frames([0, 1]), [0, 1, 2], BLANK) is None


def test_margin_correct_vs_wrong():
    lp = frames(AUDIO)
    good = margin_scores(lp, [0, 1, 2, 3], GROUPS, BLANK)
    assert good[0] > 0 and good[1] > 0
    bad = margin_scores(lp, [0, 1, 4, 3], GROUPS, BLANK)    # word 1 expects 4, audio says 2
    assert bad[1] < 0 < bad[0]


def test_gop_flags_substitution_only_in_its_word():
    lp = frames(AUDIO)
    good = gop_sf_scores(lp, [0, 1, 2, 3], GROUPS, BLANK)
    assert good[0] > 0 and good[1] > 0
    bad = gop_sf_scores(lp, [0, 1, 4, 3], GROUPS, BLANK)
    assert bad[1] < 0 and bad[0] > bad[1]


def test_gop_flags_missing_and_extra_token():
    lp = frames(AUDIO)
    missing = gop_sf_scores(lp, [0, 1, 3], [0, 0, 1], BLANK)  # audio has 2 that text lacks
    assert missing[1] < 0
    extra = gop_sf_scores(lp, [0, 1, 2, 4, 3], [0, 0, 1, 1, 1], BLANK)
    assert extra[1] < 0


def test_rank_badness():
    assert rank_badness([0.5, 10, -1], [0, 1, 2, 3]) == [0.25, 1.0, 0.0]
