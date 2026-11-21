import json, os

import numpy as np
import torch

from qafft.phonetic import ayah_uthmani
from qafft.variants import best_of, span_variants, transfer
from qafft.verify import group_scores, token_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_variants_differ_only_at_madd():
    words = tuple(ayah_uthmani(2, 4).split())          # has madd monfasel and an aared stop
    v = {n: (t, w) for n, t, w in span_variants(words)}
    assert list(v)[0] == "default" and "monfasel_2" in v and "aared_6" in v
    d, short = v["default"][0], v["monfasel_2"][0]
    assert len(d) == len(short)
    diff = [(a, b) for a, b in zip(d, short) if a != b]
    assert diff and all(a[0] == b[0] for a, b in diff)   # same letter, only the length changes
    # every variant maps onto the same printed words
    assert all(sorted(set(w)) == list(range(len(words))) for _, w in v.values())


def test_seen_saad_only_where_allowed():
    names = [n for n, _, _ in span_variants(tuple(ayah_uthmani(2, 245).split()))]
    assert "saad" in names or "seen" in names
    names = [n for n, _, _ in span_variants(tuple(ayah_uthmani(2, 10).split()))]
    assert "saad" not in names and "seen" not in names and "ghonna_lam_raa" not in names


def test_transfer_applies_edits_and_keeps_groups():
    ref, alt = ["a", "b4", "c"], ["a", "b2", "c"]
    t, g = transfer(["x", "a", "b4", "c"], [0, 1, 1, 2], ref, alt)
    assert t == ["x", "a", "b2", "c"] and g == [0, 1, 1, 2]
    t, g = transfer(["a", "z", "c"], [0, 1, 2], ref, alt)    # region already changed: left alone
    assert t == ["a", "z", "c"]


def test_best_of_never_worse_than_default():
    tokens = json.load(open(os.path.join(ROOT, "qafft", "vocab.json")))["tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}; blank = len(tokens)
    words = tuple(ayah_uthmani(2, 4).split())
    vs = [(t, w) for _, t, w in span_variants(words) if all(x in vocab for x in t)]
    rng = np.random.default_rng(0)
    lp = torch.log_softmax(torch.from_numpy(rng.normal(size=(400, blank + 1))).float(), -1)
    scores = []
    for t, w in vs:
        ts = token_scores(lp, [vocab[x] for x in t], blank)
        scores.append({g: -v for g, v in group_scores(ts, w).items()})
    best = best_of(scores)
    assert set(best) == set(scores[0])
    assert all(best[k] <= scores[0][k] + 1e-9 for k in best)
