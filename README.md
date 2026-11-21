# Qaf-ft

A speech model for Quran recitation: it writes down the sounds it hears and compares
them with the expected ayah to find recitation mistakes.

## Layout

| Folder | What it holds |
|---|---|
| `qafft/` | Text to sound tokens (`phonetic.py`), mistake-detection scoring (`mdd.py`) |
| `labeler/` | Local tool for recording and labelling recitations |
| `train/` | Training scripts and GPU notebooks |
| `bench/` | Scripts that measure existing models |
| `docs/` | Notes, assessment protocol, dataset reference |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install quran-transcript rapidfuzz
```

## Label recitations

```bash
python labeler/server.py
```

Open http://localhost:8765. Choose an ayah, record or load audio, click the words that
were recited wrongly, pick the kind of mistake, and save. Clips and labels are written to
`data/sessions/`, which is not tracked by git.

## Train

`train/kaggle_pilot.ipynb` (suras 78-114) and `train/kaggle_full.ipynb` (all suras) run on
a GPU notebook session with internet on.

Developer: Hmzr
