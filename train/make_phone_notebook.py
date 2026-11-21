"""Builds the phone fine-tune notebook for a Kaggle GPU session.

  python train/make_phone_notebook.py [--steps 8000] [--out data/kaggle_phone.ipynb]

Needs data/phone_train.json and data/phone_tokens.json from data_tools/select_phone.py.
The clip list goes inside the notebook, so the output is written under data/ and kept out of git.
On Kaggle: Accelerator GPU, Internet on, and add the output of the qaf-full-run notebook as input.
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
read = lambda p: open(os.path.join(ROOT, p), encoding="utf-8").read()

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--avg-last", type=int, default=5)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--phone-frac", type=float, default=0.5)
ap.add_argument("--hours", type=float, default=11.0)
ap.add_argument("--select", default=os.path.join(ROOT, "data", "phone_train.json"))
ap.add_argument("--tokens", default=os.path.join(ROOT, "data", "phone_tokens.json"))
ap.add_argument("--out", default=os.path.join(ROOT, "data", "kaggle_phone.ipynb"))
a = ap.parse_args()


def code(src):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src)


def md(src):
    return dict(cell_type="markdown", metadata={}, source=src)


W = "/kaggle/working/qaf"
sel = json.load(open(a.select))
n_train = sum(map(len, sel["train"].values()))
args = (f"--out /kaggle/working/phone --init $INIT --phone {W}/phone_train.json --phone-tokens {W}/phone_tokens.json "
        f"--steps {a.steps} --lr {a.lr} --head-lr {a.lr} --warm 200 --lr-floor 0 --eval-every {a.eval_every} "
        f"--avg-last {a.avg_last} --phone-frac {a.phone_frac} --hours {a.hours}")
quiet = "grep -v -E 'NeMo [WI]|^\\s*$|Cuda graphs|Reason:|Fetching|Downloading'"

cells = [
    md(f"# Qaf-ft phone fine-tune\nSettings: Accelerator = GPU, Internet = On.\n"
       f"Input: output of the `qaf-full-run` notebook (for `best.pt`).\n\n"
       f"{n_train} phone clips in {len(sel['train'])} shards, mixed with expert shards. Then Run All."),
    code("!pip -q install 'nemo_toolkit[asr]' pyarrow soundfile 2>&1 | tail -2\n"
         "import torch; print(torch.__version__, torch.cuda.is_available())"),
    code(f"import os\nfor d in ('qafft', 'train'): os.makedirs('{W}/' + d, exist_ok=True)\n"
         f"open('{W}/qafft/__init__.py', 'w').close()"),
    code(f"%%writefile {W}/qafft/phonetic.py\n" + read("qafft/phonetic.py")),
    code(f"%%writefile {W}/qafft/vocab.json\n" + read("qafft/vocab.json")),
    code(f"%%writefile {W}/train/pilot.py\n" + read("train/pilot.py")),
    code(f"%%writefile {W}/train/full.py\n" + read("train/full.py")),
    code(f"%%writefile {W}/phone_train.json\n" + json.dumps(sel)),
    code(f"%%writefile {W}/phone_tokens.json\n" + open(a.tokens, encoding="utf-8").read()),
    code("import glob\nfound = sorted(glob.glob('/kaggle/input/**/best.pt', recursive=True))\n"
         "print(found)\nassert found, 'add the qaf-full-run output as input'\nos.environ['INIT'] = INIT = found[0]"),
    code(f"!cd {W} && python train/full.py {args} --dry 2>&1 | {quiet}"),
    code(f"!cd {W} && python train/full.py {args} 2>&1 | {quiet}"),
    code("!ls -la /kaggle/working/phone; cat /kaggle/working/phone/result.json"),
]
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
json.dump(nb, open(a.out, "w"), ensure_ascii=False, indent=1)
print("wrote", a.out, f"({os.path.getsize(a.out) // 1024} KB, {n_train} train clips)")
