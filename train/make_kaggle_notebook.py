"""Builds train/kaggle_pilot.ipynb: a self-contained notebook for a Kaggle GPU session."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
read = lambda p: open(os.path.join(ROOT, p), encoding="utf-8").read()


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
    code("%%writefile /kaggle/working/qaf/qafft/vocab.json\n" + read("qafft/vocab.json")),
    code("%%writefile /kaggle/working/qaf/train/pilot.py\n" + read("train/pilot.py")),
    code("# Last shard of every moshaf holds suras 78-114.\n"
         "import collections, requests\nfrom huggingface_hub import hf_hub_download\n"
         "REPO = 'obadx/muaalem-annotated-v3'\n"
         "files = [s['rfilename'] for s in requests.get(f'https://huggingface.co/api/datasets/{REPO}').json()['siblings']]\n"
         "by = collections.defaultdict(list)\n"
         "for f in files:\n    if f.startswith('moshaf_') and f.endswith('.parquet'): by[f.split('/')[0]].append(f)\n"
         "for m, fs in sorted(by.items()):\n"
         "    p = hf_hub_download(REPO, sorted(fs)[-1], repo_type='dataset', local_dir='/kaggle/temp/data')\n"
         "    print(m, os.path.getsize(p) // 2**20, 'MB')"),
    code("!cd /kaggle/working/qaf && python train/pilot.py --data /kaggle/temp/data --out /kaggle/working/pilot "
         "--steps 3000 --batch 8 --accum 2 --eval-every 250 2>&1 | grep -v -E 'NeMo [WI]|^\\s*$|Cuda graphs|Reason:'"),
    code("!ls -la /kaggle/working/pilot; cat /kaggle/working/pilot/result.json"),
]
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
json.dump(nb, open(os.path.join(ROOT, "train", "kaggle_pilot.ipynb"), "w"), ensure_ascii=False, indent=1)
print("wrote train/kaggle_pilot.ipynb")
