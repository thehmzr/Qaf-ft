"""Builds data/kaggle_pilot.ipynb: a self-contained notebook for a Kaggle GPU session."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
read = lambda p: open(os.path.join(ROOT, p), encoding="utf-8").read()
sys.path.insert(0, ROOT)
from qafft.sources import hub_id
SOURCES = {"datasets": {k: hub_id(k) for k in ("expert", "base")}}
W = "/kaggle/working/qaf"


def code(src):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src)


def md(src):
    return dict(cell_type="markdown", metadata={}, source=src)


cells = [
    md("# Qaf-ft pilot (Juz 'Amma)\nSettings: Accelerator = GPU, Internet = On. Then Run All."),
    code("!pip -q install 'nemo_toolkit[asr]' pyarrow soundfile 2>&1 | tail -2\n"
         "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"),
    code("import os\nos.makedirs('/kaggle/working/qaf/qafft', exist_ok=True); os.makedirs('/kaggle/working/qaf/train', exist_ok=True)\n"
         "open('/kaggle/working/qaf/qafft/__init__.py','w').close()"),
    code("%%writefile /kaggle/working/qaf/qafft/phonetic.py\n" + read("qafft/phonetic.py")),
    code(f"%%writefile {W}/qafft/sources.py\n" + read("qafft/sources.py")),
    code(f"import os, json\nos.makedirs('{W}/data', exist_ok=True)\njson.dump(" + repr(SOURCES) + f", open('{W}/data/sources.json', 'w'))"),
    code("%%writefile /kaggle/working/qaf/qafft/vocab.json\n" + read("qafft/vocab.json")),
    code("%%writefile /kaggle/working/qaf/train/pilot.py\n" + read("train/pilot.py")),
    code("# Last shard of every moshaf holds suras 78-114.\n"
         "import collections, requests\nfrom huggingface_hub import hf_hub_download\n"
         "import sys; sys.path.insert(0, '/kaggle/working/qaf')\nfrom qafft.sources import hub_id\nfrom huggingface_hub import HfApi\n"
         "REPO = hub_id('expert')\n"
         "files = HfApi().list_repo_files(REPO, repo_type='dataset')\n"
         "by = collections.defaultdict(list)\n"
         "for f in files:\n    if f.startswith('moshaf_') and f.endswith('.parquet') and 'metadata' not in f: by[f.split('/')[0]].append(f)\n"
         "for m, fs in sorted(by.items()):\n"
         "    p = hf_hub_download(REPO, sorted(fs)[-1], repo_type='dataset', local_dir='/kaggle/temp/data')\n"
         "    print(m, os.path.getsize(p) // 2**20, 'MB')"),
    code("!cd /kaggle/working/qaf && python train/pilot.py --data /kaggle/temp/data --out /kaggle/working/pilot "
         "--steps 3000 --batch 8 --accum 2 --eval-every 250 2>&1 | grep -v -E 'NeMo [WI]|^\\s*$|Cuda graphs|Reason:'"),
    code("!ls -la /kaggle/working/pilot; cat /kaggle/working/pilot/result.json"),
]
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
out = os.path.join(ROOT, "data", "kaggle_pilot.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
