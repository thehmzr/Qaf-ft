# Qaf-ft work plan

For the session that continues this project on the user's own computer. Read
`handover.md` first for what exists; this file says what to do, in what order, and how to
know each piece is finished.

Structure: stage, then steps, then sub-steps, each with the process to follow and a check
that says it is done. Times are estimates, not measurements, unless stated.

--------------------------------------------------------------------------------

## A. How to work

### A1. The goal, in one sentence
A model that hears Quran recitation by ordinary people on phones and flags wrong, missing
or added words, wrong letters and wrong vowels, with almost no false flags. Tajweed judging
is a separate mode, built last.

### A2. The two numbers that decide everything
1. False-flag rate: share of correctly recited words that get flagged. Target 0.1% or
   lower per word on phone audio. Today: about 54% on hard phone clips, 0–4% on easy ones,
   1% on studio reciters (by construction).
2. Catch rate at that false-flag rate: share of real mistakes flagged. Target 50% or
   higher at 0.1%, 80% or higher at 1%. Today: unmeasured on real mistakes (3 clips).

Every stage below either moves one of these or makes measuring them trustworthy.

### A3. Working rules
- One branch per step, one pull request per branch, the user merges. Never push to main.
- Follow section 11 of `handover.md` for commit and naming conventions.
- Recordings and labels of private people never enter git. Code and notes do.
- Before any GPU run: dry-run the same code on the laptop for a few steps.
- After any run: write the result into `docs/results.md` (see A4) the same day.
- If a result looks too good, look for leakage before believing it.
- Ask the user before: spending GPU quota, deleting data, changing what counts as a
  mistake, or adding a data source.

### A4. Experiment log (create in stage 1)
`docs/results.md`, one row per experiment, newest first:

| Date | What changed | Data | Test set | False flags | Catch at 1% | Catch at 0.1% | Notes |

Rules: one change per experiment where possible; always name the checkpoint and the test
set version; never overwrite a row, add a new one.

### A5. Test sets and their names
Fix these names and never train on them:
- `studio-heldout`: reciters 5.0 and 13.0 of the expert set (exists).
- `studio-val`: reciter 19.0 (exists).
- `phone-labelled-v1`: the user's 67 labels on batch 1 (exists; small).
- `phone-heldout`: to build in stage 4, phone clips set aside before any phone training.
- `friend-correct`: the long correct recitation, once cut (stage 3).
- `mistakes-real`: labelled clips with word-located mistakes (grows from stage 6).

--------------------------------------------------------------------------------

## Stage 1. Foundation on the local machine  (about half a day)

Purpose: prove the local setup reproduces the known numbers, so later differences mean
something.

### Step 1.1 Pin the environment
- 1.1.1 Record the working versions (Python, torch, NeMo, torchaudio, pyarrow, soundfile,
  rapidfuzz, quran-transcript, kaggle, six) into `requirements.txt` with exact pins.
- 1.1.2 Add a `Makefile` or `scripts/setup.sh` that creates the environment from scratch.
- 1.1.3 Check: a fresh clone plus the script reaches "model loads" with no manual fix.

### Step 1.2 Reproduce the reference numbers
- 1.2.1 Download the last data file of reciters 5.0 and 13.0 from the expert set.
- 1.2.2 Run `eval/evaluate.py --shards … --first-sura 78 --limit 400`.
  Expected: AUC about 0.999; caught at 1% about 98%, at 0.1% about 92%.
- 1.2.3 If the numbers differ by more than a point, stop and find out why (library
  versions, audio decoding, device). Do not continue on a setup that disagrees.
- 1.2.4 Check: numbers match and are written as the first rows of `docs/results.md`.

### Step 1.3 Find out what the laptop can do
- 1.3.1 Time the model on 60 seconds of audio on CPU. Record the speed.
- 1.3.2 Try the GPU backend of the machine if it has one. Verify outputs match CPU to
  within rounding before trusting it. If it misbehaves, use CPU; correctness first.
- 1.3.3 Check: a line in `docs/results.md` with seconds of audio per second of compute.

### Step 1.4 Tidy the notes
- 1.4.1 Merge the open handover pull request or fold its content in.
- 1.4.2 Create `docs/results.md` with the rows already known (see `research_summary.md`).
- 1.4.3 Check: `handover.md`, `status.md` and `results.md` agree with each other.

--------------------------------------------------------------------------------

## Stage 2. Make measurement trustworthy  (about one day)

Purpose: every later decision depends on these scripts. Fix their known weaknesses first.

### Step 2.1 Word alignment between the two text editions
Problem: the display text and the sound script split words differently in 4 ayahs, and
marks are stored by display-word index.
- 2.1.1 Write `qafft/wordmap.py`: for an ayah, map each display word to the sound-token
  positions it covers. Method: align the letters of the display words to the letters of
  the sound-script groups (edit-distance alignment on bare letters), then carry token
  positions across.
- 2.1.2 Test on all 6,236 ayahs: every token belongs to exactly one word; every word has
  at least one token; the 4 known ayahs come out right (check them by eye).
- 2.1.3 Use it in the labelling server, in `propose.py` and in `evaluate.py`.
- 2.1.4 Check: a unit test file that runs in seconds and passes.

### Step 2.2 Evaluate on real labels
- 2.2.1 Make `evaluate.py --sessions` read the audio from either `data/sessions` or the
  batch folder (labels from a batch carry `source`).
- 2.2.2 Three label uses: clip judged correct (all words are negatives); clip with marks
  (marked words positive, others negative); unusable clips (excluded, counted).
- 2.2.3 Add both scoring modes (tajweed judged; words and vowels only) as an option.
- 2.2.4 Report per mode: false-flag rate at the studio threshold, and the full curve.
- 2.2.5 Check: reproduces 54% and 56% on the 25 hard correct clips of batch 1.

### Step 2.3 A phone threshold, not a studio one
- 2.3.1 Today's threshold comes from studio reciters, which over-flags phones.
- 2.3.2 Add an option to set the threshold from a named set of known-correct words
  (later: `friend-correct` and correct phone clips), never from the test clips themselves.
- 2.3.3 Check: thresholds are stored in a small JSON next to the checkpoint, with the set
  they came from.

### Step 2.4 Per-ayah and per-session reporting
- 2.4.1 Add to the report: share of correct ayahs with at least one flag.
- 2.4.2 This is the number a user feels. Report it beside the per-word rate every time.

### Step 2.5 Automatic tests
- 2.5.1 `tests/` with fast checks: token folding and unfolding, word map, metrics
  (`mdd.py`), mistake generator (output differs from input in exactly one place),
  checker on a synthetic probability matrix.
- 2.5.2 Check: `pytest` runs in under a minute with no model needed.

--------------------------------------------------------------------------------

## Stage 3. Cut a long recitation into ayahs  (one to two days)

Purpose: turn the friend's full, correct recitation into thousands of known-correct words
in a new voice. This is the best material for setting thresholds.

### Step 3.1 Prepare the audio
- 3.1.1 Convert to 16 kHz mono WAV. Keep the original untouched.
- 3.1.2 Listen to the first and last minute: note the opening (isti'adha, basmala) and
  anything that is not recitation.

### Step 3.2 Split at pauses
- 3.2.1 Compute short-time energy; mark silence where energy stays below a threshold for
  at least 0.3 seconds.
- 3.2.2 Cut at the middle of each silence. Merge pieces shorter than 1 second into a
  neighbour. Cap pieces at 25 seconds by cutting at the longest internal pause.
- 3.2.3 Check: plot or print the piece lengths; most should be 3–15 seconds.

### Step 3.3 Decode each piece
- 3.3.1 Run the model on every piece; keep the token sequence and the frame probabilities.

### Step 3.4 Match pieces to the text
- 3.4.1 Build the sura's full token stream with word boundaries (from `wordmap.py`).
- 3.4.2 Keep a pointer into the stream. For each piece, search a window ahead of the
  pointer (say 150 tokens) for the span that best matches the piece's tokens, using the
  words-and-vowels classes and edit distance.
- 3.4.3 Accept if the mismatch is below 15%; move the pointer to the end of the span.
- 3.4.4 If no span matches: the piece is a repeat, a correction, or noise. Search a
  window behind the pointer too. Mark it and do not move the pointer.
- 3.4.5 After a run of failures, re-anchor by searching the whole remaining sura.
- 3.4.6 Check: coverage report: share of the sura's words covered exactly once, list of
  gaps, list of repeats.

### Step 3.5 Review and export
- 3.5.1 Load the unmatched and low-confidence pieces into the labelling tool as a batch.
- 3.5.2 Export accepted pieces in the labelling format with verdict "correct", a reciter
  code, and their word spans.
- 3.5.3 Check: at least 90% of the sura's words covered; the user spot-checks 20 pieces.

### Step 3.6 Use it
- 3.6.1 Split the reciter's material: first half for thresholds, second half as test.
  Never both.
- 3.6.2 Record false flags on the test half with the current model. This is the first
  honest "new voice, correct recitation" number.

--------------------------------------------------------------------------------

## Stage 4. Choose the phone training data  (one to two days, mostly waiting)

### Step 4.1 Set the phone test set aside first
- 4.1.1 Pick 4 of the 79 phone files at random (fixed seed). Never score-filter them into
  training. They are `phone-heldout`.
- 4.1.2 From them, later, come labelling batches for testing.

### Step 4.2 Finish scoring
- 4.2.1 Run `data_tools/fetch_and_score.sh` over the other 75 files. It downloads one
  file, scores it, deletes it. Scores so far are in `data/phone_scores/`.
- 4.2.2 Each row has the full mismatch and the words-and-vowels mismatch (`err_basic`).
  Re-score file 0, which predates `err_basic`.
- 4.2.3 Estimated cost: about 10 minutes per file on a laptop CPU if the machine is
  otherwise idle; measure the first file and plan from that.

### Step 4.3 Decide what "trustworthy" means
- 4.3.1 Candidates: `err_basic` at most 0.02, 0.05, 0.10.
- 4.3.2 For each, sample 50 clips and have the user listen: share that are truly the
  claimed ayah, complete, and correct in words and vowels.
- 4.3.3 Choose the loosest setting that is right at least 95% of the time.
- 4.3.4 Danger to understand: this keeps clips the model already hears well, so it teaches
  less than it seems. That is acceptable for a first round; stage 6 adds harder clips with
  human labels.

### Step 4.4 Build the training list
- 4.4.1 Labels: the claimed ayah's sound tokens. For clips kept on words-and-vowels match
  only, train the tajweed-detail targets too, but record the clip as "lengths unverified".
- 4.4.2 Balance: cap clips per ayah (for example 30) so short suras do not dominate.
- 4.4.3 Write a manifest: file, clip name, sura, ayah, mismatch scores.
- 4.4.4 Check: counts by sura, by duration, total hours, written to `docs/results.md`.

--------------------------------------------------------------------------------

## Stage 5. Phone fine-tune  (one day of work, 9–10 GPU-hours)

### Step 5.1 Code changes to `train/full.py`
- 5.1.1 `--init PATH`: start from an existing checkpoint instead of the base model.
- 5.1.2 A second data stream for phone clips, read from the manifest of stage 4.
- 5.1.3 Mixing: each batch draws from expert and phone data at a set ratio. Start 50/50.
  Reason: training on phone clips alone would forget the expert reciters.
- 5.1.4 Learning rate: start 3e-5 for the encoder, 1e-4 for the head, cosine decay to zero.
  Reason: the model is already trained; the validation swings pointed at a rate that stayed
  too high.
- 5.1.5 Keep a running average of the weights (exponential moving average, decay 0.999)
  and evaluate the averaged copy. Save both.
- 5.1.6 Augmentation for expert clips only: band-limit, mild clipping, room echo, added
  noise. Phone clips are already real.
- 5.1.7 Exclude reciter 25.0 (flagged by its own authors as badly cut).
- 5.1.8 Validation on three sets every 1,000 steps: `studio-val`, a slice of correct phone
  clips, and `friend-correct` threshold half.
- 5.1.9 Save a checkpoint to the output folder every 2,000 steps so a stopped run is not lost.

### Step 5.2 Dry run
- 5.2.1 Five steps on the laptop: loss is a normal number, both streams deliver batches,
  a checkpoint is written and can be loaded again.

### Step 5.3 Run
- 5.3.1 Regenerate the notebook, push to Kaggle, GPU on, internet on.
- 5.3.2 Watch the first 15 minutes of the log: loss falling, no skipped-batch warnings.
- 5.3.3 12,000–15,000 steps is the first guess; stop early if all three validation sets
  have stopped improving.

### Step 5.4 Judge it
- 5.4.1 Run the full evaluation on every test set of A5 that exists.
- 5.4.2 Pass if: studio numbers are no worse than before by more than a small margin;
  false flags on `phone-labelled-v1` correct clips fall sharply; `friend-correct` test
  half improves.
- 5.4.3 If studio got worse: raise the expert share and repeat.
- 5.4.4 If phone did not improve: the selected clips were too easy. Go to stage 6 and
  come back with human-labelled hard clips.
- 5.4.5 Write all of it to `docs/results.md`.

--------------------------------------------------------------------------------

## Stage 6. Labelling at scale  (ongoing; the user's time)

Purpose: real mistakes located at the word are the scarcest thing in the project.

### Step 6.1 Improve what goes into a batch
- 6.1.1 Half of batch 1 was unusable. Pre-filter: drop clips much shorter or longer than
  the ayah, and clips whose best-matching ayah is a different one.
- 6.1.2 Sampling plan per batch of 100: 50 where the model doubts one or two words,
  20 it scores as perfect, 20 at random, 10 repeats of earlier clips.
- 6.1.3 Spread over many suras and ayah lengths.

### Step 6.2 Run the model locally inside the tool
- 6.2.1 Add an endpoint to the labelling server that scores a clip with the model, so
  suggestions appear without preparing batches elsewhere.
- 6.2.2 Keep one clip in five without suggestions.

### Step 6.3 Label quality
- 6.3.1 After every 100 clips show the agreement on repeats.
- 6.3.2 If agreement on verdict drops below 90%, stop and write down the rule that was
  unclear (for example: is a slightly open vowel a mistake?). Add it to a short
  `docs/labelling_guide.md`.
- 6.3.3 If a second listener is available, have them label 50 of the same clips.

### Step 6.4 Targets
- 6.4.1 First milestone: 300 usable clips, at least 60 with a word-located mistake.
- 6.4.2 Second: 1,000 usable, 200 with mistakes, at least 30 different reciter voices.
- 6.4.3 Split by voice where known, otherwise by source file: 60% train, 20% threshold
  setting, 20% test. Fix the split when a clip is labelled, not later.

### Step 6.5 Other sources of mistakes
- 6.5.1 The open Surah Al-Ikhlas error set (1,506 clips labelled by five experts): map
  its labels to the project's format.
- 6.5.2 Volunteers reciting chosen ayahs with planned mistakes. Useful, but mark them as
  acted; acted mistakes are easier than natural ones.

--------------------------------------------------------------------------------

## Stage 7. Teach the model what mistakes sound like  (two to four days, 5–10 GPU-hours)

### Step 7.1 Text-side examples (no audio change)
- 7.1.1 Already used for evaluation. For training the checker (stage 8), not the model.

### Step 7.2 Spliced audio
- 7.2.1 From word timings of the same reciter, replace one word with a different word of
  similar length, or remove a word, or repeat one.
- 7.2.2 Cross-fade 10 milliseconds at the joins. Label the said-token row accordingly.
- 7.2.3 Listen to 30 before using any: discard the method if the joins are obvious.

### Step 7.3 Train in order
- 7.3.1 First generated material, at a low rate, mixed with normal clips.
- 7.3.2 Then real labelled mistakes last, with the said-token row as the target.
- 7.3.3 Published evidence: this order beat mixing everything together.
- 7.3.4 Check on `mistakes-real` test split only.

--------------------------------------------------------------------------------

## Stage 8. The verifier: from scores to decisions  (two to three days)

### Step 8.1 Features per word
Weakest aligned sound; mean aligned score; extra-sound charge; margin to the best
competing sound; number of sounds; duration against the reciter's pace; scores of the
neighbouring words; signal level; position in the ayah.

### Step 8.2 Model
- 8.2.1 Start with regularised logistic regression. Standardise features inside each fold.
- 8.2.2 Try boosted trees only if they beat it clearly under cross-validation.
- 8.2.3 Cross-validation folds are whole reciters or whole source files.

### Step 8.3 Calibration and thresholds
- 8.3.1 Fit calibration on the threshold-setting split only.
- 8.3.2 Choose two cut-offs: below the first, pass; above the second, flag; between, say
  nothing. Set the second for 0.1% false flags.
- 8.3.3 Per ayah: with m words, require stronger evidence as m grows, or cap the flags
  shown to those passing a false-discovery limit (see `assessment_protocol.md`).

### Step 8.4 Check
- 8.4.1 Report catch rate at 1% and 0.1%, per-ayah false flags, and a calibration plot.
- 8.4.2 Report separately for men, women and children if the data allows.

--------------------------------------------------------------------------------

## Stage 9. Following a reciter live  (three to five days)

### Step 9.1 Chunked listening
- 9.1.1 Measure accuracy when the model sees audio in chunks (for example 1 second with
  some overlap) instead of whole clips.
- 9.1.2 If accuracy drops much, fine-tune with limited context, or use overlapping windows
  and keep only the middle of each.

### Step 9.2 Tracking position
- 9.2.1 Reuse the matcher of stage 3 as a live tracker: a pointer in the sura's token
  stream, moved as sounds arrive.
- 9.2.2 Handle: repeats, going back to correct, skipping ahead, stopping mid-ayah.
- 9.2.3 Decide when a flag is shown: only once the word is complete and the next one has
  begun, so a self-correction is not flagged.

### Step 9.3 Check
- 9.3.1 Replay labelled recordings through the live path; results must match the offline
  checker within a small margin.
- 9.3.2 Delay from end of word to decision: target under 1.5 seconds.

--------------------------------------------------------------------------------

## Stage 10. Onto the phone  (three to five days)

### Step 10.1 Export
- 10.1.1 Export encoder plus head to ONNX. Compare outputs with the original on 50 clips.
- 10.1.2 Quantise to 8-bit. Re-measure every test set; re-fit thresholds on the quantised
  model, since quantising shifts scores.

### Step 10.2 Measure on real devices
- 10.2.1 One recent phone and one older, low-memory phone.
- 10.2.2 Record: file size, memory, speed, battery over ten minutes.
- 10.2.3 Targets: under 150 MB, faster than real time with room to spare on the older phone.

### Step 10.3 If it is too heavy
- 10.3.1 Train a smaller student (30–60 million weights) on the big model's outputs over
  the same audio, then repeat stages 8 and 10.

### Step 10.4 The checker on the device
- 10.4.1 Port alignment scoring and the extra-sound check. Verify against the reference
  implementation on recorded probability matrices, to the last decimal that matters.

--------------------------------------------------------------------------------

## Stage 11. Into the app  (with the user)

- 11.1 Agree the interface: input is audio chunks plus the expected ayah range; output is
  word index, state (pass, unsure, flag), kind of mistake, and what was heard.
- 11.2 Feedback rules: show flags only; never show "unsure"; let the user dismiss a flag,
  and store dismissals (with consent) as future labels.
- 11.3 A setting for strictness that moves the threshold within tested limits.
- 11.4 Model files downloaded on first use, with a fingerprint check.

--------------------------------------------------------------------------------

## Stage 12. Tajweed mode  (last)

- 12.1 Only for reciters who ask for it.
- 12.2 Start with madd length relative to the reciter's own pace, ghunnah presence and
  qalqalah presence. Published results put the harder rules well below usable accuracy.
- 12.3 Needs its own labelled test set from trained reciters, marked at named places.
- 12.4 Advisory wording only until measured false flags are as low as the main mode.

--------------------------------------------------------------------------------

## Stage 13. Release

- 13.1 Model card: what it was trained on, every test set and number, known weaknesses,
  who it was not tested on.
- 13.2 Credits and licence file; confirm every source in the model allows the chosen licence.
- 13.3 A public benchmark report: method, data, numbers, and the comparison with existing
  open models on the same clips.
- 13.4 Consent record for every voice the project recorded or was given.

--------------------------------------------------------------------------------

## B. Order and dependencies

    1 Foundation
    └─ 2 Measurement ──────────────┐
       ├─ 3 Long recitation ───────┤
       └─ 4 Phone data ── 5 Phone fine-tune ── 7 Mistake training ── 8 Verifier
                    6 Labelling (runs alongside everything from stage 2 on)
                                         8 Verifier ── 9 Live ── 10 Phone ── 11 App
                                                                  12 Tajweed, 13 Release

Stages 3 and 4 can run in parallel. Stage 6 never stops.

## C. Budget (estimates)

| Item | GPU-hours | Notes |
|---|---|---|
| Phone fine-tune | 9–10 | Kaggle free quota is about 30 a week |
| Repeat with a different mix | 9–10 | only if 5.4.3 or 5.4.4 applies |
| Mistake training | 5–10 | |
| Student model, if needed | 20–40 | may need two weeks of quota or a rented GPU |
| Spent so far | about 11 | pilot attempts and the full run |

Money spent so far: none.

## D. Risks and what to do about them

| Risk | Sign | Response |
|---|---|---|
| Phone filter keeps only easy clips | Phone numbers do not move after stage 5 | Human-labelled hard clips (stage 6) before another run |
| Unverified phone labels teach wrong text | Studio numbers fall | Tighter filter; more expert share |
| Too few real mistakes | Wide intervals, unstable catch rate | Stage 6 targets; expert-labelled open set |
| Labels drift | Repeat agreement falls | Labelling guide; second listener |
| Thresholds do not transfer between voices | Good on one reciter, bad on another | Thresholds from many voices; per-session adaptation later |
| Model too slow on phones | Stage 10 measurements | Student model |
| Licence surprise | New source added in a hurry | Check `data_licenses.md` before any download |
| Test set leaks into training | Numbers jump | Fixed names (A5), splits decided at labelling time |
| Recordings of private people exposed | Files outside `data/` | Keep `data/` ignored; never attach clips to pull requests |

## E. What to ask the user, and when

| When | Question |
|---|---|
| Stage 3 | Which opening phrases were recited; is the recitation continuous or in sessions |
| Stage 4 | Listen to 3 × 50 sample clips to set the trust level |
| Before each GPU run | Go-ahead, with the expected hours |
| Stage 6 | Edge cases for the labelling guide |
| Stage 8 | How strict the app should be by default |
| Stage 11 | Target phones; how flags should look |
| Stage 13 | Which licence to release under |

## F. Glossary

- Sound token: one unit the model outputs: a letter, a short vowel, a long vowel with its
  length, a nasal sound, or a mark. There are 81.
- CTC: the training method; it needs audio and the token sequence, no timings.
- Checkpoint: a saved copy of the model's weights.
- False flag: a correct word marked as a mistake.
- Catch rate: share of real mistakes that get flagged.
- Threshold: the score above which a word is flagged.
- Held-out: data from voices the model never trained on.
- Planted mistake: a mistake introduced on purpose, in the text or the audio.
- Fine-tune: continue training an existing model on new data.
- Quantise: store weights in 8 bits to make the model smaller and faster.
