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

Branch `labeler-tajweed` holds all of it. Nothing is merged into `main` yet.

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
