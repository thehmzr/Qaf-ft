# Handover

Everything a new working session needs to continue. Read `status.md` for the short version
and `research_summary.md` for numbers and sources.

## 1. What exists

| Piece | Where | State |
|---|---|---|
| Sound tokens (81) | `qafft/phonetic.py`, `qafft/vocab.json` | Done |
| Training scripts | `train/pilot.py`, `train/full.py`, `train/kaggle_*.ipynb` | Done, run on Kaggle |
| Trained model | Kaggle kernel `qaf-full-run`, output `full/best.pt` (418 MB) | 0.43% token error on held-out reciters |
| Checker | `qafft/verify.py` | Alignment score + extra-sound check; words-and-vowels mode added, not yet measured |
| Mistake generator | `qafft/mistakes.py` | Done |
| Metrics | `qafft/mdd.py`, `eval/evaluate.py` | Done |
| Labelling tool | `labeler/` | Batch queue, suggestion review, sound-level edits, tajweed colours |
| Phone-clip scoring | `data_tools/phone_filter.py`, `fetch_and_score.sh` | 2 of 79 files scored |
| Batch builder | `data_tools/make_queue.py`, `propose.py` | Done |
| Text and tajweed colours | `qafft/quran_text.py`, `qafft/tajweed_text.py` | Done |

Everything is on `main`. `models/full/best.pt` is the checkpoint from step 19,000 of the 30,000-step run, the one with the best validation score; the 0.43% test figure belongs to it.

## 2. Set up on a new machine

```bash
git clone https://github.com/thehmzr/Qaf-ft && cd Qaf-ft && git checkout labeler-tajweed
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes the model libraries (several GB). For the labelling tool alone,
`pip install quran-transcript rapidfuzz` is enough.

Files that are not in git and must be fetched again:

| File | How |
|---|---|
| Base checkpoint | `huggingface-cli download nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` or let the code download it |
| `models/full/best.pt` | Kaggle: kernel `qaf-full-run`, Output tab, or `kaggle kernels output thehmzr/qaf-full-run -p models/full` |
| Display text | first start of the labelling tool, or `python -m qafft.quran_text` |
| `data/ayah_tokens.json` | built by the snippet at the top of `data_tools/phone_filter.py`'s history; one loop over `ayah_tokens` |
| Held-out reciters' data | `obadx/muaalem-annotated-v3`, moshaf 5.0 and 13.0 (test), 19.0 (validation) |
| Phone recordings | `deepdml/tlog-clean-sf16k`, 79 files of 5,000 clips |

Accounts needed: a Kaggle API token (`~/.kaggle/kaggle.json`) to push and read runs, and a
Hugging Face token for gated sets.

## 3. Rules that hold

- Words and vowels are the target. Tajweed judging is a later, optional mode.
- Only permissively licensed sources go into the model (`data_licenses.md`). Excluded: sets under
  the Quran-Lab, QUA and non-commercial licences, and audio with no documented licence.
- Three reciters are never trained on: moshaf 5.0 and 13.0 (test), 19.0 (validation).
- Every result is split and resampled by reciter, and reported at fixed false-flag rates
  (`assessment_protocol.md`).
- Phone recordings and labels of private people stay out of git.
- Mixed precision is off in training: it gave invalid losses with this encoder.

## 4. What was learned

- Text-output models hide mistakes; sound tokens without a language model do not.
- The model is strong on studio reciters and weak on phones: on 25 hand-labelled correct phone
  clips the checker flagged 51% of words at the studio threshold.
- About half of phone clips in a "sounds differ" sample were cut off, another ayah or noise.
  When the model scores a phone clip as a perfect match, hand labels agreed 7 times out of 7.
- Ordinary reciters often do not apply tajweed lengths, so judging lengths produces false flags.
  Hence the words-and-vowels mode.
- Validation error swings between checkpoints (0.2% to 1.5%). Padding is ruled out: single and
  batched decoding give identical results. Likely the learning rate late in training.

## 5. Next steps, in order

1. Measure the words-and-vowels mode on the 35 usable hand-labelled clips
   (`data/user_labels` + batch 1 audio). Expect the 51% to drop; if it does not, the gap is acoustic.
2. Finish scoring phone clips (`fetch_and_score.sh`), then select training clips where
   `err_basic` is small (start at 0.05 or less).
3. Phone fine-tune: start from `full/best.pt`, mix expert shards with the selected phone clips,
   lower the learning rate (start 3e-5, decay to zero) and average the last checkpoints.
   `train/full.py` needs a second data source and a `--init` option for this.
4. Re-run step 1 and the planted-mistake evaluation on the new model.
5. Cut a long correct recitation into ayah clips: split at pauses, decode each piece, match to
   the sura's token stream with a moving window. Not built yet.
6. More hand labels, with suggestions showing, from suras outside 100-104.
7. Train on generated mistakes, then on labelled real ones.
8. Calibrated thresholds with a pass / unsure / flag decision.
9. Phone export (ONNX, int8) and speed on a device.
10. Tajweed mode.

## 6. Known loose ends

- Four ayahs split words differently between the display text and the sound script
  (15:7, 27:20, 36:22, 44:33); marks there can land one word off.
- `propose.py` sets its threshold on studio reciters, so it over-suggests on phone audio.
- The checker's intervals are meaningless with two test reciters; add more held-out voices.
- `eval/evaluate.py --sessions` maps marked words to sound groups by index and has not been
  run on real labels yet.
- Excluded on the authors' advice but still in the training stream: moshaf 25.0.

## 7. Map of the documents

| File | What it answers |
|---|---|
| `docs/status.md` | Where things stand, in one page |
| `docs/research_summary.md` | Every finding and measured number, with sources: existing models, methods, base architectures, datasets, each run and test |
| `docs/datasets.md` | Each dataset in detail: how it was built, format, quirks, exact licence clauses |
| `docs/data_licenses.md` | Which sources are used for training, for evaluation only, on hold, or excluded, and why |
| `docs/how_others_build.md` | How other teams collect, label and check recitation data, and what was adopted |
| `docs/assessment_protocol.md` | How results are judged: thresholds, false-flag arithmetic, resampling by reciter |
| `docs/notes/` | Chapter notes on the statistical learning textbook behind the protocol |
| `CREDITS.md` | Credits owed to each source |

## 8. Datasets at a glance

| Dataset | Role | Access |
|---|---|---|
| obadx/muaalem-annotated-v3 (848 h, expert reciters, sound labels) | Main training set | Open |
| deepdml/tlog-clean-sf16k (about 393,000 phone clips, unverified labels) | Phone training after filtering; source of labelling batches | Open |
| Surah Al-Ikhlas error set (1,506 clips, labelled by five experts) | Mistake examples | Open |
| RetaSy learner clips (911 judged correct or incorrect) | Evaluation only: no licence stated | Open |
| obadx/qdat_bench (159 learners, one ayah, tajweed-level labels) | Evaluation, for the later tajweed mode | Open |
| Luxbg/quran-muaalem-aligned, unseen-reciters split | Held-out reciters with word timings | Open |
| IqraEval sets, Tadabur | Not used: non-commercial | Access granted |
| Quran-Lab benchmarks, QUA aligned recitations | Not used: their licences would set the model's own licence | Access granted |
| Several Kaggle sets (tajweed by rule, children's mispronunciation, isolated letters, word-by-word Al-Fatihah) | Not yet examined beyond their descriptions | Kaggle account |

The user's own labels and any recitation recorded for the project are the only data with no
licence question, and the only source of word-located real mistakes.

## 9. Pages and tools outside the model code

| Thing | Where | What it is |
|---|---|---|
| Control room page | source `page/index.html`, published privately on claude.ai | Live list of background runs, the user's tasks, reports, results. Reads and writes a small shared store; button presses queue requests that a session applies when it next looks |
| System page | source `page/system.html`, published privately on claude.ai | Technical reference with diagrams: workflow, model, checker, stages, results |
| Background runner | `tools/jobs.py`, `tools/progress.py` | Queue, pause, resume, stop and restart long local runs, with progress and time estimates in `jobs/` |
| Benchmarks of existing models | `bench/` | Scripts and scoring used for the baseline comparisons |
| Kaggle runs | kernels `notebook9fe060249a` (pilot) and `qaf-full-run` | Notebooks are generated from `train/make_kaggle_notebook.py`; push and read with the Kaggle API |

The two published pages are tied to the account that created them. A new session can republish
from the source files, or update the existing ones if it has access to them.

## 10. Methods in one paragraph each

- **Listening.** Audio to 81 sound tokens with CTC; no text is given to the model and no language
  model is used, so a mistake is not corrected away.
- **Checking.** Align the expected tokens to the audio and take the weakest sound per word; decode
  freely and charge unmatched sounds to the neighbouring word; the larger of the two is the word's
  score. A threshold from known-correct recitation turns it into pass, unsure or flag.
- **Words-and-vowels mode.** Merge token classes that differ only in tajweed detail before scoring.
  Measured on 25 hand-labelled correct phone clips: 54% of words flagged with tajweed judged, 56%
  without. The phone gap is acoustic, not about lengths.
- **Filtering phone clips.** Score each clip against the ayah its file name claims; keep close
  matches for training, send partial matches to labelling, drop the rest.
- **Labelling.** Each clip stores expected and said token rows, a clip class, tajweed places, and
  the labeller's decision on each model suggestion. One clip in five is shown without suggestions.
- **Generated mistakes.** One realistic edit per sequence (letter, vowel, madd, doubling, ghunnah,
  qalqalah), used for planted-mistake evaluation and later for training.

## 11. Working conventions

- Commits use the author already set in git config. No co-author or attribution lines in commits
  or pull requests.
- Never push to `main`. Work on a branch; the user reviews and merges every pull request.
- Commit and pull request text: one short subject line and at most a few short bullets.
- No other apps named, and no mention of recordings or test audio, in commits or pull requests.
- Research notes and numbers go in `docs/`, never in commit or pull request text.
- People are named by role only: the user, the reciter, the qari, the listener.
- The user's own recording sessions stay local.
