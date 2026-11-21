"""Display words <-> sound tokens.

Labels mark mistakes by display word (the printed Uthmani words of quran_text),
while scoring works on sound tokens and the phonetic script's spoken groups.
The two usually match one to one, but a group can join two words (idgham across
words) and a few printed words are spoken as two groups (e.g. "لَّوۡمَا").

The map lines up the bare letters of the display words with the letter tokens
of the sound script (edit distance, with letter variants the phonetic script
rewrites counted as equal), then gives every token the word of its aligned
letter. Vowel marks follow the token before them; unmatched tokens go to a
neighbour, preferring one in the same spoken group.
"""
from functools import lru_cache

# sound tokens that only mark the token before them
SOUND_MARKS = set("َُِڇ۪ۜ")

# display letters -> sound letters they may be written as
EQUAL = {
    "ء": "ءٲ", "ا": "اءٲ", "ٱ": "اء", "ى": "اۦي", "ي": "يۦ", "و": "وۥ",
    "ة": "ته", "ن": "نںم", "م": "م۾",
}
HAMZA_ON = "ٕٔ"          # hamza above / below, written after the seat letter
LONG = {"ٰ": "ا", "ۥ": "و", "ۦ": "ي", "ۧ": "ي", "ۨ": "ن"}


def display_letters(word):
    """Bare letters of a printed word, one string per letter."""
    out = []
    for ch in word:
        if ch in HAMZA_ON:
            if out and out[-1] in "اوىيـ":
                out[-1] = "ء"                         # seat + hamza mark = one hamza
            else:
                out.append("ء")
        elif ch in "أإؤئ":
            out.append("ء")                           # precomposed forms (other text sources)
        elif ch == "آ":
            out += ["ء", "ا"]
        elif ch in LONG:
            out.append(LONG[ch])
        elif "ء" <= ch <= "ي" or ch == "ٱ":
            out.append(ch)
        elif ch == "ـ":
            out.append("ـ")                           # tatweel, only kept to carry a hamza
    return [c for c in out if c != "ـ"]


SEP = "|"          # word / group boundary
# costs in tenths; an unmatched boundary costs less than an unmatched letter
GAP, SEP_GAP, SUB = 10, 6, 10


def _sub(x, y):
    if x == SEP or y == SEP:
        return 0 if x == y else 10 ** 6
    return 0 if y in EQUAL.get(x, x) else SUB


def _gap(x):
    return SEP_GAP if x == SEP else GAP


def _align(a, b):
    """Edit-distance alignment of lists a (display letters) and b (sound letters),
    both with SEP between words / groups so boundaries tend to line up.
    Returns, for each item of b, the index in a it is matched to, or None."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = d[i - 1][0] + _gap(a[i - 1])
    for j in range(1, m + 1):
        d[0][j] = d[0][j - 1] + _gap(b[j - 1])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + _sub(a[i - 1], b[j - 1]),
                          d[i - 1][j] + _gap(a[i - 1]), d[i][j - 1] + _gap(b[j - 1]))
    out, i, j = [None] * m, n, m
    while i and j:
        # on ties: exact match, then skipping a letter, then a substitution
        sub = _sub(a[i - 1], b[j - 1])
        if sub == 0 and d[i][j] == d[i - 1][j - 1]:
            out[j - 1] = i - 1; i -= 1; j -= 1
        elif d[i][j] == d[i - 1][j] + _gap(a[i - 1]):
            i -= 1
        elif d[i][j] == d[i][j - 1] + _gap(b[j - 1]):
            j -= 1
        else:
            out[j - 1] = i - 1; i -= 1; j -= 1
    return out


def map_tokens(words, tokens, groups):
    """For each token, the index of the display word it belongs to."""
    letters, owner = [], []
    for w, word in enumerate(words):
        if w:
            letters.append(SEP); owner.append(None)
        ls = display_letters(word)
        letters += ls; owner += [w] * len(ls)
    lt = [k for k, t in enumerate(tokens) if t[0] not in SOUND_MARKS]
    seq, pos = [], []                      # sound letters with SEP at group changes
    for k in lt:
        if seq and groups[k] != groups[pos[-1]]:
            seq.append(SEP); pos.append(None)
        seq.append(tokens[k][0]); pos.append(k)
    match = _align(letters, seq)
    tw = [None] * len(tokens)
    for k, mi in zip(pos, match):
        if k is not None and mi is not None:
            tw[k] = owner[mi]
    # unmatched letters: a matched neighbour in the same group, else the one before
    for k in lt:
        if tw[k] is not None:
            continue
        prev = next((j for j in range(k - 1, -1, -1) if tw[j] is not None), None)
        nxt = next((j for j in range(k + 1, len(tokens)) if tw[j] is not None and tokens[j][0] not in SOUND_MARKS), None)
        if prev is not None and (groups[prev] == groups[k] or nxt is None or groups[nxt] != groups[k]):
            tw[k] = tw[prev]
        elif nxt is not None:
            tw[k] = tw[nxt]
        else:
            tw[k] = 0
    # marks follow the token before them
    for k, t in enumerate(tokens):
        if tw[k] is None:
            tw[k] = tw[k - 1] if k else next((x for x in tw if x is not None), 0)
    return tw


def ayah_phonemes(sura, aya, moshaf=None):
    from quran_transcript import quran_phonetizer
    from qafft.phonetic import ayah_uthmani, default_moshaf
    return quran_phonetizer(ayah_uthmani(sura, aya), moshaf or default_moshaf(), remove_spaces=False).phonemes


def display_words(sura, aya):
    from qafft import quran_text
    from qafft.phonetic import ayah_uthmani
    return quran_text.words(sura, aya) or ayah_uthmani(sura, aya).split()


@lru_cache(maxsize=4096)
def _cached(sura, aya, words):
    from qafft.phonetic import fold_with_groups
    tokens, groups = fold_with_groups(ayah_phonemes(sura, aya))
    return tokens, groups, map_tokens(list(words), tokens, groups)


def word_tokens(sura, aya, words=None):
    """(tokens, token_word, n_words): the ayah's sound tokens (same as
    fold_with_groups gives) and the display word of each token."""
    words = tuple(words or display_words(sura, aya))
    tokens, _, tw = _cached(sura, aya, words)
    return list(tokens), list(tw), len(words)


def token_groups(sura, aya, words=None):
    """Spoken group of each token, matching word_tokens."""
    words = tuple(words or display_words(sura, aya))
    return list(_cached(sura, aya, words)[1])


def group_words(sura, aya, words=None):
    """For each spoken group, the display words its tokens belong to (sorted)."""
    words = tuple(words or display_words(sura, aya))
    _, groups, tw = _cached(sura, aya, words)
    out = [set() for _ in range(max(groups) + 1)] if groups else []
    for g, w in zip(groups, tw):
        out[g].add(w)
    return [sorted(s) for s in out]


def empty_words(tw, n_words):
    """Display words that got no token."""
    have = set(tw)
    return [w for w in range(n_words) if w not in have]


if __name__ == "__main__":
    # check every ayah: tokens in order, every word has a token
    import sys
    from qafft.phonetic import all_ayahs
    bad, n = [], 0
    for s, a in all_ayahs():
        toks, tw, nw = word_tokens(s, a)
        n += 1
        if any(x > y for x, y in zip(tw, tw[1:])) or empty_words(tw, nw):
            bad.append((s, a, empty_words(tw, nw)))
    print(n, "ayahs,", len(bad), "with a problem")
    for b in bad:
        print(*b)
    sys.exit(1 if bad else 0)
