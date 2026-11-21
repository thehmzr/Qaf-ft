"""Mispronunciation detection metrics (the standard three-sequence scheme).

Per clip there are three token sequences:
  ref  : what should have been said
  said : what was actually said (human annotation)
  pred : what the model heard
Each reference position is scored:
  TA  correct sound, accepted            FR  correct sound, flagged (false flag)
  TR  wrong sound, flagged               FA  wrong sound, accepted (missed)
  CD / DE : among TR, the model named the actual sound correctly / wrongly
"""
from rapidfuzz.distance import Levenshtein


def _per_ref(ref, other):
    """For each ref position: the token aligned to it in `other`, or None if deleted.
    Insertions are attached to the preceding ref position as a mismatch."""
    out = list(ref)
    touched = [False] * len(ref)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, other):
        if tag == "replace":
            for k in range(i2 - i1):
                out[i1 + k] = other[j1 + k] if j1 + k < j2 else None
                touched[i1 + k] = True
        elif tag == "delete":
            for k in range(i1, i2):
                out[k], touched[k] = None, True
        elif tag == "insert" and ref:
            k = max(0, min(len(ref) - 1, i1 - 1))
            out[k], touched[k] = ("+", tuple(other[j1:j2])), True
    return out, touched


def count(ref, said, pred):
    s, s_err = _per_ref(ref, said)
    p, p_err = _per_ref(ref, pred)
    c = dict(TA=0, FR=0, TR=0, FA=0, CD=0, DE=0)
    for k in range(len(ref)):
        if not s_err[k]:
            c["FR" if p_err[k] else "TA"] += 1
        elif p_err[k]:
            c["TR"] += 1
            c["CD" if p[k] == s[k] else "DE"] += 1
        else:
            c["FA"] += 1
    return c


def summarize(counts):
    t = {k: sum(c[k] for c in counts) for k in ("TA", "FR", "TR", "FA", "CD", "DE")}
    div = lambda a, b: a / b if b else float("nan")
    p, r = div(t["TR"], t["TR"] + t["FR"]), div(t["TR"], t["TR"] + t["FA"])
    return dict(t, precision=p, recall=r, f1=div(2 * p * r, p + r),
                false_reject_rate=div(t["FR"], t["TA"] + t["FR"]),
                false_accept_rate=div(t["FA"], t["FA"] + t["TR"]),
                diagnosis_error_rate=div(t["DE"], t["CD"] + t["DE"]))


if __name__ == "__main__":
    ref = list("abcdef")
    assert count(ref, ref, ref) == dict(TA=6, FR=0, TR=0, FA=0, CD=0, DE=0)
    assert count(ref, list("abXdef"), list("abXdef"))["CD"] == 1
    assert count(ref, list("abXdef"), list("abYdef"))["DE"] == 1
    assert count(ref, list("abXdef"), ref)["FA"] == 1
    assert count(ref, ref, list("abXdef"))["FR"] == 1
    assert count(ref, list("abdef"), list("abdef"))["TR"] == 1
    print(summarize([count(ref, list("abXdef"), list("abXdeY"))]))
