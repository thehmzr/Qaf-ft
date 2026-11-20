"""Local labelling tool.

  python labeler/server.py [--port 8765] [--data data/sessions]

Open http://localhost:8765 in a browser. Pick an ayah, record or load a
recitation, mark the words that were recited wrongly, and save. Each save writes
<data>/<id>.wav and <data>/<id>.json. Nothing leaves this machine.
"""
import argparse, json, os, random, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qafft.phonetic import ayah_sites, ayah_tokens, ayah_uthmani
from qafft.tajweed_text import COLOURS, NAMES, ORDER, coloured_words

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = None
CLIP_KINDS = ["usable", "incomplete", "different ayah", "several ayahs", "noise or unrelated"]
SURAS = ("Al-Fatihah Al-Baqarah Aal-Imran An-Nisa Al-Maidah Al-Anam Al-Araf Al-Anfal At-Tawbah Yunus Hud Yusuf "
         "Ar-Rad Ibrahim Al-Hijr An-Nahl Al-Isra Al-Kahf Maryam Ta-Ha Al-Anbiya Al-Hajj Al-Muminun An-Nur "
         "Al-Furqan Ash-Shuara An-Naml Al-Qasas Al-Ankabut Ar-Rum Luqman As-Sajdah Al-Ahzab Saba Fatir Ya-Sin "
         "As-Saffat Sad Az-Zumar Ghafir Fussilat Ash-Shura Az-Zukhruf Ad-Dukhan Al-Jathiyah Al-Ahqaf Muhammad "
         "Al-Fath Al-Hujurat Qaf Adh-Dhariyat At-Tur An-Najm Al-Qamar Ar-Rahman Al-Waqiah Al-Hadid Al-Mujadilah "
         "Al-Hashr Al-Mumtahanah As-Saff Al-Jumuah Al-Munafiqun At-Taghabun At-Talaq At-Tahrim Al-Mulk Al-Qalam "
         "Al-Haqqah Al-Maarij Nuh Al-Jinn Al-Muzzammil Al-Muddaththir Al-Qiyamah Al-Insan Al-Mursalat An-Naba "
         "An-Naziat Abasa At-Takwir Al-Infitar Al-Mutaffifin Al-Inshiqaq Al-Buruj At-Tariq Al-Ala Al-Ghashiyah "
         "Al-Fajr Al-Balad Ash-Shams Al-Layl Ad-Duha Ash-Sharh At-Tin Al-Alaq Al-Qadr Al-Bayyinah Az-Zalzalah "
         "Al-Adiyat Al-Qariah At-Takathur Al-Asr Al-Humazah Al-Fil Quraysh Al-Maun Al-Kawthar Al-Kafirun An-Nasr "
         "Al-Masad Al-Ikhlas Al-Falaq An-Nas").split()
MISTAKES = ["letter", "vowel mark", "madd length", "ghunnah", "qalqalah", "skipped", "added", "repeated", "other"]


def load_labels():
    out = []
    for f in sorted(os.listdir(DATA)):
        if f.endswith(".json"):
            out.append(json.load(open(os.path.join(DATA, f))))
    return out


def consistency(labels):
    """How often a second, blind labelling of the same clip agrees with the first."""
    by = {d["id"]: d for d in labels}
    pairs = [(by[d["repeat_of"]], d) for d in labels if d.get("repeat_of") in by]
    if not pairs:
        return dict(pairs=0)
    same = sum(a["verdict"] == b["verdict"] and a.get("clip_kind") == b.get("clip_kind") for a, b in pairs)
    overlap = []
    for a, b in pairs:
        x, y = {m["word"] for m in a["marks"]}, {m["word"] for m in b["marks"]}
        overlap.append(1.0 if not (x | y) else len(x & y) / len(x | y))
    return dict(pairs=len(pairs), same_verdict=same / len(pairs), word_overlap=sum(overlap) / len(overlap))


def sura_sizes():
    from quran_transcript import Aya
    return [Aya(s, 1).get().num_ayat_in_sura for s in range(1, 115)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html")
        if u.path == "/meta":
            return self.send(200, dict(sizes=sura_sizes(), mistakes=MISTAKES, clip_kinds=CLIP_KINDS, suras=SURAS,
                                       rules=[dict(id=r, name=NAMES[r], colour=COLOURS[r]) for r in ORDER]))
        if u.path == "/ayah":
            s, a = int(q["s"][0]), int(q["a"][0])
            text = ayah_uthmani(s, a)
            return self.send(200, dict(sura=s, aya=a, words=text.split(), tokens=ayah_tokens(s, a), sites=ayah_sites(s, a),
                                       coloured=coloured_words(text.split())))
        if u.path == "/list":
            items = []
            for f in sorted(os.listdir(DATA)):
                if f.endswith(".json"):
                    d = json.load(open(os.path.join(DATA, f)))
                    items.append(dict(id=d["id"], sura=d["sura"], aya=d["aya"], verdict=d["verdict"],
                                      marks=len(d["marks"]), saved=d["saved"], repeat=bool(d.get("repeat_of"))))
            return self.send(200, items)
        if u.path == "/audio":
            rid = os.path.basename(q["id"][0])
            path = os.path.join(DATA, rid + ".wav")
            if not os.path.exists(path):
                return self.send(404, dict(error="not found"))
            return self.send(200, open(path, "rb").read(), "audio/wav")
        if u.path == "/control":
            # an earlier clip to label again without seeing the first label
            labels = load_labels()
            done = {d.get("repeat_of") for d in labels}
            pool = [d for d in labels if not d.get("repeat_of") and d["id"] not in done]
            if len(pool) < 5:
                return self.send(200, dict(none=True, need=5 - len(pool)))
            d = random.choice(pool[:-3] or pool)
            return self.send(200, dict(id=d["id"], sura=d["sura"], aya=d["aya"]))
        if u.path == "/stats":
            return self.send(200, consistency(load_labels()))
        self.send(404, dict(error="not found"))

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/save":
            return self.send(404, dict(error="not found"))
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        # body = 4-byte big-endian JSON length, JSON, then WAV bytes
        jl = int.from_bytes(raw[:4], "big")
        label = json.loads(raw[4:4 + jl])
        wav = raw[4 + jl:]
        if wav[:4] != b"RIFF":
            return self.send(400, dict(error="audio must be WAV"))
        for m in label.get("marks", []):
            if m.get("type") not in MISTAKES:
                return self.send(400, dict(error="unknown mistake type"))
        if label.get("clip_kind", "usable") not in CLIP_KINDS:
            return self.send(400, dict(error="unknown clip kind"))
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        label.update(id=rid, saved=time.strftime("%Y-%m-%dT%H:%M:%S"),
                     expected_tokens=ayah_tokens(int(label["sura"]), int(label["aya"])))
        open(os.path.join(DATA, rid + ".wav"), "wb").write(wav)
        json.dump(label, open(os.path.join(DATA, rid + ".json"), "w"), ensure_ascii=False, indent=1)
        self.send(200, dict(id=rid))


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "sessions"))
    a = ap.parse_args()
    DATA = a.data
    os.makedirs(DATA, exist_ok=True)
    print(f"labelling tool on http://localhost:{a.port}  (saving to {DATA})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
