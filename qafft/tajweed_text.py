"""Tajweed colouring of Uthmani text, letter by letter.

Reads the text itself and works out, from the letters and their signs, which rule
each letter calls for: qalqalah, ghunnah, a letter that is not said, the madd
lengths, and heavy letters.
"""
import re

ALLAH = re.compile("^(?:[\u0621-\u064a][\u064b-\u0652]?){0,2}[\u0671\u0627]?\u0653?\u0644\u0644\u0651\u064e\u0647[\u064e\u064f\u0650](?:\u0645\u0651\u064e)?$")
_HELPERS = re.compile("[\u06d6-\u06dc\u06de\u06e9\u06ed\u06e2\u06e5\u06e6 ]")


def bare(text):
    """the word without stop signs and small helper marks"""
    return _HELPERS.sub("", text)



FATHA, DAMMA, KASRA, SHADDA, SUKUN, MADD = "َ", "ُ", "ِ", "ّ", "ْ", "ٓ"
TANWEEN = set("ًٌٍ")
VOWELS = {FATHA, DAMMA, KASRA} | TANWEEN
DAGGER, SMALL_WAW, SMALL_YA = "ٰ", "ۥ", "ۦ"
ZERO, RECT_ZERO, LOW_MEEM, HIGH_MEEM = "۟", "۠", "ۭ", "ۢ"
MARKS = VOWELS | {SHADDA, SUKUN, MADD, DAGGER, SMALL_WAW, SMALL_YA, ZERO, RECT_ZERO, LOW_MEEM, HIGH_MEEM} | set(
    "ـۣ۪ٕۜۤۧۨ۫۬ٔۡ"
)
STOPS = set("\u06d6\u06d7\u06d8\u06da")  # signs where a reader may stop (not ۙ, which says do not, nor the three dots)
OTHER = set("\u06d9\u06db\u06de\u06e9\u200f ")
WASL = "ٱ"
QALQALAH = set("قطبجد")
IKHFA = set("تثجدذزسشصضطظفقك")
YANMU = set("ينمو")




class Letter:
    __slots__ = ("c", "m", "word", "first", "last", "stop", "end")

    def __init__(self, c, word):
        self.c, self.m, self.word = c, "", word
        self.first = self.last = self.stop = self.end = False

    def bare(self):
        """no sign that gives the letter a sound of its own"""
        return not (set(self.m) & (VOWELS | {SHADDA, SUKUN, MADD, DAGGER}))


def letters_of(words):
    """the ayah's letters in order, each with its signs"""
    out = []
    for w, text in enumerate(words):
        mine = []
        # two spellings made plain: آ is an alif with the madd sign; a hamza written over the line is a letter
        text = text.replace("\u0622", "\u0627\u0653")
        text = re.sub("\u0640([\u064b-\u0650]?)\u0654", "\u0621\\1", text)
        for ch in text:
            if ch in STOPS:
                if mine:
                    mine[-1].stop = True
            elif ch in OTHER:
                continue
            elif ch in MARKS:
                if mine:
                    mine[-1].m += ch
            else:
                mine.append(Letter(ch, w))
        if mine:
            mine[0].first, mine[-1].last = True, True
        out += mine
    if out:
        out[-1].end = True
    return out


def letter_rules(words):
    """the tajweed rules of every letter of the ayah: (letters, [set of rule names per letter])"""
    L = letters_of(words)
    tags = [set() for _ in L]
    n = len(L)

    def said_after(i):
        """the next letter that is pronounced when reading on"""
        j = i + 1
        while j < n:
            x = L[j]
            if x.c == WASL or ZERO in x.m or (x.c in "اى" and not x.m and x.last and set(L[j - 1].m) & TANWEEN):
                j += 1
                continue
            return j
        return None

    def vowel_before(i):
        """the short vowel on the letter before, if it is in the same word"""
        if i == 0 or L[i - 1].word != L[i].word:
            return None
        for v in (FATHA, DAMMA, KASRA):
            if v in L[i - 1].m:
                return v
        return None

    def long_or_soft(i):
        """letter i lengthens the vowel before it (a madd letter), or is a soft waw or ya after a fatha"""
        x, v = L[i], vowel_before(i)
        if DAGGER in x.m and x.c not in "ىي":
            return True  # the small alif, written on the letter it follows
        if x.c in "ىي" and DAGGER in x.m:
            return True
        if x.c == "ا" and not x.m:
            return v == FATHA
        if x.c == "و" and not (set(x.m) - {MADD}):
            return v == DAMMA
        if x.c in "يى" and not (set(x.m) - {MADD}):
            return v == KASRA
        if x.c in "ويى" and SUKUN in x.m:
            return v == FATHA
        return False

    whole = "".join(words)
    letters_only = not (set(whole) & (VOWELS | {SUKUN, SHADDA}))  # the opening letters of a surah (الٓمٓ)
    for i, x in enumerate(L):
        m = set(x.m)
        pause = x.last and (x.end or x.stop)
        j = said_after(i)
        nxt = L[j] if j is not None else None
        in_word_next = L[i + 1] if i + 1 < n and L[i + 1].word == x.word else None
        opening = letters_only or not (set(words[x.word]) & (VOWELS | {SUKUN, SHADDA}))
        # qalqalah: one of the five letters with a sukun, or the last letter where the reader stops
        if x.c in QALQALAH and (SUKUN in m or pause) and not opening:
            tags[i].add("qalqalah")
        # noon and meem with a shadda (not the first letter of an ayah: its shadda is only there for
        # reading on from the ayah before)
        if x.c in "نم" and SHADDA in m and i > 0:
            tags[i].add("ghunnah")
        # noon without a sign, and tanween written in its joined form: what follows decides
        hidden = None
        if x.c == "ن" and x.bare() and nxt is not None and not x.stop:
            hidden = "noon"
        elif m & TANWEEN and (LOW_MEEM in m or HIGH_MEEM in m) and nxt is not None and not x.end:
            hidden = "tanween"
        if hidden:
            if nxt.c == "ب":
                tags[i].add("ghunnah")
                if hidden == "noon":
                    tags[i].add("silent")  # the noon is not said; the small meem over it is
            elif nxt.c in YANMU and nxt.word != x.word:
                if not (hidden == "noon" and nxt.c == "ن"):
                    tags[i].add("silent")
                if nxt.c in "يو":
                    tags[j].add("ghunnah")
            elif nxt.c in "لر" and nxt.word != x.word:
                tags[i].add("silent")
            elif nxt.c in IKHFA:
                tags[i].add("ghunnah")
        # meem without a sign before ب
        if x.c == "م" and x.bare() and nxt is not None and nxt.c == "ب" and not x.stop:
            tags[i].add("ghunnah")
        # the madd sign: six counts in the opening letters and before a doubled or resting letter,
        # four or five before a hamza. Not at the end of an ayah, where the reader stops.
        if MADD in m:
            if opening or (in_word_next is not None and (SHADDA in in_word_next.m or SUKUN in in_word_next.m or in_word_next.bare() and in_word_next.c not in "اوىي")):
                tags[i].add("madd6")
            elif not (x.end or (in_word_next is not None and in_word_next.end and ZERO in in_word_next.m)):
                tags[i].add("madd45")
        # where the reader stops, a long or soft letter before the last letter may be held 2, 4 or 6 counts
        arid = False
        if in_word_next is not None and in_word_next.last and (in_word_next.end or in_word_next.stop) and MADD not in m:
            last = in_word_next
            if not (set(last.m) & {"\u064b"}) and last.c not in "اوىي" + "ء" * 0 and long_or_soft(i):
                arid = True
                tags[i].add("madd246")
        # the small alif, waw and ya, held two counts
        if MADD not in m and not arid:
            small = 0 if x.end else sum(1 for ch in x.m if ch in (SMALL_WAW, SMALL_YA, "\u06e7"))
            if DAGGER in m and not (x.c in "ىي" and x.last):
                small += 1
            (tags[i].add("madd2") if small else None)
        # letters written but not said
        after_stop = x.first and i > 0 and L[i - 1].stop
        if x.c == WASL and i > 0 and not after_stop:
            tags[i].add("silent")
        elif ZERO in m:  # (the upright zero of أَنَا۠ is said when stopping; the font leaves it black)
            tags[i].add("silent")
        elif x.c in "ىيو" and DAGGER in m and not (m & VOWELS) and SHADDA not in m and not (x.last and x.c != "و"):
            tags[i].add("silent")  # the ya or waw that only carries a small alif (هَدَىٰكُمْ, صَلَوٰتَكَ)
        elif "\u06dc" in m:
            tags[i].add("silent")
        elif x.c not in "نم" + WASL and not x.m and nxt is not None and SHADDA in nxt.m and j == i + 1 and x.c not in "اوى" and nxt.c != x.c:
            tags[i].add("silent")
        # heavy letters. A letter takes one colour: ق and ط with qalqalah show that instead.
        if x.c in "خصضغطقظ":
            if not (x.c in QALQALAH and (SUKUN in m or pause)):
                tags[i].add("heavy")
        elif x.c == "ر" and opening:
            tags[i].add("heavy")
        elif x.c == "ر":
            if pause or not (m & VOWELS):
                # resting, or stopped on: what comes before decides
                k = i - 1
                if k >= 0 and L[k].word == x.word and (SUKUN in L[k].m or not L[k].m) and L[k].c not in "يى" + WASL:
                    k -= 1  # a resting letter or a long alif or waw in between
                if k < 0 or L[k].word != x.word or L[k].c == WASL:
                    heavy = True  # ٱرْجِعُوا: the kasra one starts with is not part of the word
                elif L[k].c in "يى" and not (set(L[k].m) & {FATHA, DAMMA}):
                    heavy = False
                elif KASRA in L[k].m:
                    heavy = in_word_next is not None and in_word_next.c in "خصضغطقظ" and KASRA not in in_word_next.m
                else:
                    heavy = True
            else:
                heavy = bool(m & {FATHA, DAMMA, "\u064b", "\u064c"})
            if heavy:
                tags[i].add("heavy")
        elif x.c == "ل" and SHADDA in m and FATHA in m and in_word_next is not None and in_word_next.c == "ه" and i > 0 and L[i - 1].c == "ل" and L[i - 1].word == x.word and not L[i - 1].m and ALLAH.match(bare(words[x.word])):
            # the lam of the name of Allah: heavy after a fatha or damma, and when starting with it
            k = i - 2
            while k >= 0 and not (set(L[k].m) & (VOWELS | {SUKUN})):
                k -= 1
            if k < 0 or set(L[k].m) & {FATHA, DAMMA, "\u064b", "\u064c"}:
                tags[i].add("heavy")
    return L, tags


ORDER = ["madd6", "madd45", "madd246", "madd2", "ghunnah", "qalqalah", "silent", "heavy"]
COLOURS = {"heavy": "#3f48e6", "qalqalah": "#2fadff", "ghunnah": "#09b000", "silent": "#a3a5a5",
           "madd2": "#ce9e00", "madd246": "#ff7b00", "madd45": "#f40000", "madd6": "#b50000"}
NAMES = {"heavy": "Heavy letter", "qalqalah": "Qalqalah", "ghunnah": "Ghunnah", "silent": "Not pronounced",
         "madd2": "Madd, 2 counts", "madd246": "Madd, 2, 4 or 6 at a stop", "madd45": "Madd, 4 or 5", "madd6": "Madd, 6"}
_CLUSTER = re.compile("\u0640[\u064b-\u0650]?\u0654|.", re.S)


def coloured_words(words):
    """For display: each word as a list of [text, rule or None] pieces, text unchanged."""
    L, tags = letter_rules(words)
    out, k = [], 0
    for w, text in enumerate(words):
        pieces = []
        for m in _CLUSTER.finditer(text):
            ch = m.group(0)
            base = len(ch) > 1 or not (ch in MARKS or ch in STOPS or ch in OTHER)
            if base and k < len(L) and L[k].word == w:
                rule = next((r for r in ORDER if r in tags[k]), None)
                pieces.append([ch, rule]); k += 1
            elif pieces:
                pieces[-1][0] += ch
            else:
                pieces.append([ch, None])
        # merge neighbours with the same rule
        merged = []
        for t, r in pieces:
            if merged and merged[-1][1] == r:
                merged[-1][0] += t
            else:
                merged.append([t, r])
        out.append(merged)
        while k < len(L) and L[k].word == w:
            k += 1
    return out
