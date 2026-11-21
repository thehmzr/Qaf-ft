"""Cuts correct recitation at word boundaries to put a mistake in the audio.

A clip is described by its audio, its spoken words (phoneme strings, one per word,
as in the phonetic script split on spaces) and cut points: cuts[i] is the sample where
word i starts, cuts[-1] the end of the clip (len(cuts) = words + 1). Joins use a short
linear cross-fade.

  deletion(wav, words, cuts, k)               word k removed
  repetition(wav, words, cuts, k)             word k played twice
  swap(wav, words, cuts, k, d_wav, d_words, d_cuts, j)   word k replaced by donor word j

Each returns (audio, true words, join positions in samples). true_tokens(words) gives
the folded tokens and word index of the spliced audio.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qafft.phonetic import fold_with_groups

FADE_MS = 10


def crossfade_concat(pieces, sr=16000, fade_ms=FADE_MS):
    """Joins pieces with a linear cross-fade of fade_ms. Each join overlaps the end of
    one piece with the start of the next, so the result is fade samples shorter per join.
    Returns (audio, join positions = centre of each overlap)."""
    n = int(round(sr * fade_ms / 1000))
    pieces = [np.asarray(p, dtype=np.float32) for p in pieces if len(p)]
    if not pieces:
        return np.zeros(0, np.float32), []
    out, joins = pieces[0].copy(), []
    for p in pieces[1:]:
        f = min(n, len(out), len(p))
        if f == 0:
            joins.append(len(out)); out = np.concatenate([out, p]); continue
        ramp = (np.arange(f, dtype=np.float32) + 0.5) / f
        mix = out[len(out) - f:] * (1 - ramp) + p[:f] * ramp
        joins.append(len(out) - f // 2)
        out = np.concatenate([out[: len(out) - f], mix, p[f:]])
    return out, joins


def true_tokens(words):
    """Folded tokens of a list of word phoneme strings, and the word index of each token."""
    return fold_with_groups(" ".join(words))


def _check(words, cuts, k, inner=True):
    if len(cuts) != len(words) + 1:
        raise ValueError("need one cut per word plus the end")
    if inner and not 0 < k < len(words) - 1:
        raise ValueError("only inner words")


def deletion(wav, words, cuts, k, sr=16000, fade_ms=FADE_MS):
    _check(words, cuts, k)
    audio, joins = crossfade_concat([wav[: cuts[k]], wav[cuts[k + 1]:]], sr, fade_ms)
    return audio, words[:k] + words[k + 1:], joins


def repetition(wav, words, cuts, k, sr=16000, fade_ms=FADE_MS):
    _check(words, cuts, k)
    w = wav[cuts[k]: cuts[k + 1]]
    audio, joins = crossfade_concat([wav[: cuts[k + 1]], w, wav[cuts[k + 1]:]], sr, fade_ms)
    return audio, words[: k + 1] + [words[k]] + words[k + 1:], joins


def swap(wav, words, cuts, k, d_wav, d_words, d_cuts, j, sr=16000, fade_ms=FADE_MS):
    _check(words, cuts, k)
    _check(d_words, d_cuts, j)
    w = d_wav[d_cuts[j]: d_cuts[j + 1]]
    audio, joins = crossfade_concat([wav[: cuts[k]], w, wav[cuts[k + 1]:]], sr, fade_ms)
    return audio, words[:k] + [d_words[j]] + words[k + 1:], joins


def cuts_from_frames(first, last, n_samples, sr=16000, fps=25):
    """Cut points from each word's first and last aligned frame: each inner cut sits at
    the middle of the gap between one word's last frame and the next word's first frame
    (CTC ends come out early, so this widens the earlier word)."""
    cuts = [0]
    for i in range(1, len(first)):
        end_prev = (last[i - 1] + 1) / fps
        start = first[i] / fps
        t = (end_prev + start) / 2 if start >= end_prev else start
        cuts.append(int(round(t * sr)))
    cuts.append(n_samples)
    return cuts
