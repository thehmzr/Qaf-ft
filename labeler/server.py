"""Local labelling tool.

  python labeler/server.py [--port 8765] [--data data/sessions]

Open http://localhost:8765 in a browser. Pick an ayah, record or load a
recitation, mark the words that were recited wrongly, and save. Each save writes
<data>/<id>.wav and <data>/<id>.json. Nothing leaves this machine.
"""
import argparse, json, os, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qafft.phonetic import ayah_tokens, ayah_uthmani

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = None
MISTAKES = ["letter", "vowel mark", "madd length", "ghunnah", "qalqalah", "skipped", "added", "repeated", "other"]


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
            return self.send(200, dict(sizes=sura_sizes(), mistakes=MISTAKES))
        if u.path == "/ayah":
            s, a = int(q["s"][0]), int(q["a"][0])
            text = ayah_uthmani(s, a)
            return self.send(200, dict(sura=s, aya=a, words=text.split(), tokens=ayah_tokens(s, a)))
        if u.path == "/list":
            items = []
            for f in sorted(os.listdir(DATA)):
                if f.endswith(".json"):
                    d = json.load(open(os.path.join(DATA, f)))
                    items.append(dict(id=d["id"], sura=d["sura"], aya=d["aya"], verdict=d["verdict"],
                                      marks=len(d["marks"]), saved=d["saved"]))
            return self.send(200, items)
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
