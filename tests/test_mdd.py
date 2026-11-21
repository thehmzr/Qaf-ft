import math

from qafft.mdd import _per_ref, count, summarize

REF = list("abcdef")


def test_all_correct():
    assert count(REF, REF, REF) == dict(TA=6, FR=0, TR=0, FA=0, CD=0, DE=0)


def test_substitutions():
    assert count(REF, list("abXdef"), list("abXdef")) == dict(TA=5, FR=0, TR=1, FA=0, CD=1, DE=0)
    assert count(REF, list("abXdef"), list("abYdef"))["DE"] == 1
    assert count(REF, list("abXdef"), REF)["FA"] == 1
    assert count(REF, REF, list("abXdef"))["FR"] == 1


def test_deletion_and_insertion():
    c = count(REF, list("abdef"), list("abdef"))
    assert c["TR"] == 1 and c["CD"] == 1
    out, touched = _per_ref(REF, list("abcZdef"))
    assert touched == [False, False, True, False, False, False]    # insertion charged to the one before
    assert out[2] == ("+", ("Z",))
    assert count(REF, list("abcZdef"), REF)["FA"] == 1


def test_summarize():
    s = summarize([count(REF, list("abXdef"), list("abXdeY")), count(REF, REF, REF)])
    assert (s["TR"], s["FR"], s["FA"], s["TA"]) == (1, 1, 0, 10)
    assert s["precision"] == 0.5 and s["recall"] == 1.0
    assert abs(s["f1"] - 2 / 3) < 1e-9
    assert abs(s["false_reject_rate"] - 1 / 11) < 1e-9
    assert s["diagnosis_error_rate"] == 0.0


def test_summarize_no_mistakes_is_nan():
    s = summarize([count(REF, REF, REF)])
    assert math.isnan(s["precision"]) and math.isnan(s["recall"])
