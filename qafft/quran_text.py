"""The Uthmani text used for display and tajweed colouring.

Same file and same pinned version the mobile app downloads
(fawazahmed0/quran-api, edition ara-quranuthmanihaf, Unlicense), so the tool and the
app show identical text. Fetch it once with:  python -m qafft.quran_text
"""
import hashlib, json, os, re, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "quran-hafs-uthmani.json")
COMMIT = "93e6f64925ae76273520d636fcaaf061129f2316"
URLS = [f"https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@{COMMIT}/editions/ara-quranuthmanihaf.min.json",
        f"https://raw.githubusercontent.com/fawazahmed0/quran-api/{COMMIT}/editions/ara-quranuthmanihaf.min.json"]
SHA256 = "13a4668ca04e60898ee3922affa62a8a1d1a76d60f6e6f8431718eaa8a0e776d"
_cache = None
# this edition writes the alif of a tanween as a separate piece ("كِتَٰبٗ ا"); it belongs to the word before
_LOOSE = re.compile("^[\u0627\u0649][\u06d6-\u06ed]*$")


def fetch():
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    for url in URLS:
        try:
            raw = urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            continue
        if hashlib.sha256(raw).hexdigest() != SHA256:
            raise SystemExit("downloaded file does not match the expected fingerprint")
        open(PATH, "wb").write(raw)
        return PATH
    raise SystemExit("could not download the Quran text")


def available():
    return os.path.exists(PATH)


def words(sura: int, aya: int):
    """The ayah's words, or None if the file has not been fetched."""
    global _cache
    if _cache is None:
        if not available():
            return None
        _cache = {(v["chapter"], v["verse"]): v["text"] for v in json.load(open(PATH, encoding="utf-8"))["quran"]}
    text = _cache.get((sura, aya))
    if not text:
        return None
    out = []
    for w in text.split():
        if out and _LOOSE.match(w):
            out[-1] += w
        else:
            out.append(w)
    return out


if __name__ == "__main__":
    print(fetch())
