import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
from qafft.verify import basic_classes, extra_sounds, group_scores, to_basic, token_scores

V = 5          # tokens 0..4, blank = 5
BLANK = V


def frames(path, p=0.95):
    """Log-probabilities where frame t puts probability p on path[t]."""
    lp = torch.full((len(path), V + 1), (1 - p) / V)
    for t, k in enumerate(path):
        lp[t, k] = p
    return lp.log()


def test_token_scores_clean():
    ids = [0, 1, 1, 2]
    lp = frames([0, 0, BLANK, 1, 1, BLANK, 1, BLANK, 2, 2])
    ts = token_scores(lp, ids, BLANK)
    assert len(ts) == 4
    assert all(abs(s - math.log(0.95)) < 1e-4 for s in ts)


def test_token_scores_weak_token_and_too_short():
    ids = [0, 1, 2]
    lp = frames([0, 0, BLANK, 1, 1, BLANK, 2, 2])
    lp[3:5] = torch.tensor([0.05, 0.3, 0.05, 0.05, 0.5, 0.05]).log()   # token 1 unclear, 4 likelier
    ts = token_scores(lp, ids, BLANK)
    assert ts[1] < ts[0] - 1 and ts[1] < ts[2] - 1
    assert token_scores(frames([0]), [0, 0], BLANK) is None              # repeat needs a blank between


def test_group_scores():
    assert group_scores([-0.1, -2.0, -0.5, -0.2], [0, 0, 1, 1]) == {0: -2.0, 1: -0.5}


def test_extra_sounds():
    ids, groups = [0, 1, 2], [0, 0, 1]
    assert extra_sounds(frames([0, BLANK, 1, BLANK, 2]), ids, groups, BLANK) == {}
    out = extra_sounds(frames([0, BLANK, 1, BLANK, 3, BLANK, 2]), ids, groups, BLANK)
    assert set(out) == {0}                              # extra sound after token 1, charged to its group
    assert abs(out[0] + math.log(1 - 0.95)) < 1e-3


def test_basic_classes():
    toks = ["ن", "ن4", "ں3", "م", "م2", "۾2", "ا2", "ا4", "ڇ", "ب"]
    idx, k = basic_classes(toks)
    assert idx[1] == idx[2] and idx[0] != idx[1]       # held and hidden noon: one class
    assert idx[4] == idx[5] and idx[3] != idx[4]
    assert idx[6] == idx[7]                            # madd length ignored
    assert idx[8] is None
    assert k == len({i for i in idx if i is not None})


def test_to_basic():
    tokens = ["ب", "ا2", "ا4", "ڇ", "ت"]               # V = 5
    ids, groups = [0, 1, 3, 4], [0, 0, 1, 1]
    lp = frames([0, 1, 2, BLANK, 3, 4])
    out, new_ids, new_groups, k = to_basic(lp, tokens, ids, groups)
    assert k == 3 and out.shape == (6, 4)
    assert torch.allclose(out.exp().sum(-1), torch.ones(6), atol=1e-5)  # probability kept
    assert new_ids == [0, 1, 2] and new_groups == [0, 0, 1]             # qalqalah dropped
    assert out[2].argmax().item() == 1                                  # ا4 frame -> the alif class
    assert out[4].argmax().item() == k                                  # ڇ frame -> silence
