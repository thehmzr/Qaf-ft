# Qaf-ft research summary (condensed from three research threads, 2026-11-20)

Evidence note: numbers below were read by the research threads in model cards, leaderboard
files and arXiv papers. Some vendor pages were blocked, so another app's internals are
search-snippet or third-party only. Two threads (datasets, training methods) still pending.

## Key findings
1. Text-output models "auto-correct": on the bench_b board, of 100 planted mistakes only
   1.2-4.9 appear in the output of text models (Whisper, an Arabic LLM, FastConformer text fine-tunes)
   versus 19-27 for phoneme-CTC models (model_a 26.6, model_c V4 24.6).
   => output unit must be phonemes, no language model.
2. False flags are the unsolved problem. Best published phoneme system (the mdd_e task's 2026 winner)
   wrongly rejects 1.75% of correct phonemes => a false flag in about half of 40-phoneme
   ayahs. (Source to confirm: the figure is not in the mdd_e paper, whose own baselines
   reject 12.65-21% of correct phonemes; see notes/quran_recitation_sources.md.) Phoneme checkers flag 21-51 of 100 correct recitations on the bench_b board
   (model_a 50.5, model_c V4 21.0, model_d 40.3).
3. Real mistake data matters most: the mdd_e task's 2026 winner F1 0.46 (synthetic only) -> 0.68 after a
   second stage on ~1.5 h of real mispronounced speech -> 0.72 with ensemble.
4. Small CTC models trained on the right data mix beat huge general models on Quran audio:
   a small open Quran ASR model (33M): WER 0.56 studio / 2.13 phone; Whisper-large-v3 7.09 / 23.86.
5. Another app in production (third-party measurement, bench_a): 9.56 WER overall, 16.17 on real phone
   audio; word-level mistakes + tashkeel toggle; no tajweed; server-side only.
6. Tajweed today: qalqalah F1 0.95, mushaddad ghunnah 0.87, madd RMSE ~0.5-1 harakah,
   ikhfa F1 0.45 (learner_b benchmark, the expert set's paper). Ship as advisory.

## Base candidates
- base (the pretrained FastConformer Arabic model): 115M, diacritics, 390 h Quran (studio_b)
  in pretraining; proven on-device runtime path; int8 ~125 MB; fine-tune 740 h in ~3 h on an
  8 GB GPU (CTC head). Not streaming as shipped; cache-aware conversion documented.
- w2v-BERT 2.0 / model_a (v3_2): 606M, phonemes + 10 sifat heads, best
  offline teacher; its card and repository give conflicting terms; not streaming.
- A small Conformer-family encoder (model_b class) / small Conformer-CTC 16-65M from scratch: tops the checker board; 19-73 MB int8;
  needs ~1,500 h incl. phone audio; no permissive pretrained Arabic checkpoint.
- A 638M native streaming Arabic model; server-class.
- Not suitable: Whisper family (hallucination, 30 s windows), Citrinet (no Arabic checkpoint,
  no cache-aware streaming, superseded), several other streaming and multilingual models (no Arabic, or non-commercial).

## Method recommendations
- Quran phonetic script labels (the phonetizer package): durations as repeated symbols, sifat heads.
- Verification separate from recognition: CTC likelihood ratio (segmentation-free GOP) of the
  expected ayah vs alternatives; per-phoneme thresholds at a fixed false-reject budget; three
  states (correct / mistake / unsure); benign events (repeats, self-corrections, basmala).
- Never condition the recogniser on the reference text (recall collapses: F1 55 -> 40.5).
- Usable data: studio_b, phone (real phone recordings, gated, noisy
  labels), expert (848 h), studio_e, learner_a,
  learner_b, mdd_b real errors. studio_d and the mdd_e task's data are non-commercial.

## Compute estimates (for ~800 h)
- FastConformer 115M CTC-only: 5-10 small-GPU hours (free on the GPU notebook service) or <$5 rented.
- FastConformer hybrid + streaming conversion: 20-40 small-GPU hours or $6-25.
- Small Conformer-family model from scratch (1,500 h x 20 epochs): 80-150 small-GPU hours or $25-75.
- w2v-BERT 2.0 600M: 70-90 small-GPU hours per epoch or $25-50 per epoch on a large GPU.
- Platform jobs: small GPU $0.40/h, mid GPU $0.80/h, large GPU $2.50/h. GPU notebook service ~30 free GPU-h/week.

Targets to claim "best": phoneme false-reject <= 0.3% on correct recitation (published best
1.75%, source to confirm), recall >= 50% on real errors, WER <= 0.6 studio / <= 2 phone, int8 size <= 75 MB.

## Datasets (fifth thread, condensed)
Train: expert (848 h, 22 male experts, phoneme+sifat labels);
studio_b recitations de-duplicated (drop Warsh, teaching-style, translation folders);
phone clean split (393k real phone clips; open 16 kHz mirror 55 GB; labels inferred
from filename, must be filtered by ASR agreement); a commercial-use noise set.
Reciter-disjoint test: the aligned expert set's test_unseen split (2 reciters, 60 h, word timings).
Real-mistake data (all small): learner_a 1,166 labelled clips (396 correct / 476 mistake, clip-level,
inter-rater 0.63, ignores tajweed); learner_b 159 learners one verse
(tajweed-level); mdd_e 1,642 scripted MSA errors; mdd_d 46 Hafs
clips with word-level errors; mdd_c 1,506 clips (851 error).
studio_b's test split shares reciters with train: useless for generalisation claims.
Isolated letters: a set of 8,904 clips = 28 letters x 3 short vowels
(Indonesian students); a makharij drill set of 391 drills with sukoon/shaddah.
Usage terms of every source are kept with the source files; several (studio_d, the mdd_e task's
sets, studio_e, bench_a, bench_b) are non-commercial or forbid paid features, and model_a's
checkpoints carry extra conditions.
No open set has real, word-located mistakes across the full Quran: an own expert-labelled
test set is needed for a trustworthy false-flag number.

## Data repository check (direct access, 2026-11-20)
GPU notebook service (84 hits read, 12 descriptions read):
- A labelled Al-Fatiha set split word by word, ~175-220 different reciters per word
  (762 MB). Useful for word-level alignment tests and speaker variety.
- A tajweed set with clips by noon-sakinah rule (ikhfa/izhar/idgham/iqlab) per reciter, 1 GB.
- A TTS mispronunciation-detection set: 2,000 recordings, 16 speakers, expected vs
  realized phonemes with timings, 1,219 correct / 781 with errors, speaker-independent splits (not Quran).
- An Arabic speech mispronunciation set: 100 Egyptian children (2-8 y),
  top-100 words, expert segmental error labels (not Quran; child speech).
- Isolated letters: several small sets (1,439 clips of 28 letters; 1 s clips; a fathah-only set;
  a short-vowels set, referenced).
- A female recitation set (205 MB, terms unknown): rare female voices.
- An 87 GB copy of the expert set on the service (handy for training there).
- One "qaida" set is Urdu OCR images, not audio.
Other data repositories: nothing with mistake labels; one recitation-styles-and-reciters set (27 GB) is style classification.
mdd_c confirmed at source: 1,506 records, labelled by 5 Quran experts.
Still true: no dataset with word-located real mistakes across the full Quran.

## Pilot model as a checker (planted mistakes, one held-out reciter, Juz Amma)
Command: eval/evaluate.py on 400 clips of a reciter never trained on. Each clip scored against
its true text and against a text with one planted mistake (text-side, audio unchanged).
Separation (AUC) 0.942. Wrong words caught at 5% / 1% / 0.1% false flags: 87.5% / 86.3% / 79.0%.
By kind at 1% false flags: letter 1.00, vowel 1.00, doubling 1.00, ghunnah 0.97, madd 0.68,
qalqalah 0.00.
Limits: one reciter, so no interval; mistakes are planted in the text, which is easier than real
mistakes; expert studio audio. Qalqalah scores zero because the planted mistake removes an
expected sound, and alignment scoring cannot see a sound that is present in the audio but absent
from the expected text. Needs a second check against the free decode.

Update: with the extra-sound check added and both held-out reciters (800 clips): AUC 0.998;
caught at 5% / 1% / 0.1% false flags: 99.4% / 97.0% / 82.5%. By kind at 1%: letter 1.00,
vowel 1.00, doubling 1.00, ghunnah 0.99, madd 0.91, qalqalah 0.88. Intervals from two reciters
are not meaningful. Same limits as above: planted in the text, expert studio audio, Juz Amma.

## Full run (all suras, expert set, 30,000 steps, about 9 hours on one GPU of the GPU notebook service)
Token error on the two held-out reciters (first and last shard of each, 5,201 segments): 0.43%.
Best validation 0.18% (step 19,000); validation swung between 0.2% and 1.5% from check to check.
Checker on the same 800 held-out Juz Amma clips as the pilot (planted mistakes): AUC 0.999;
caught at 5% / 1% / 0.1% false flags: 99.9% / 98.3% / 92.1% (pilot: 99.4 / 97.0 / 82.5).
By kind at 1%: letter, vowel, doubling, ghunnah 1.00; madd 0.97; qalqalah 0.84.

## First hand-labelled phone clips (67 of batch 1, labelled without seeing model suggestions)
Batch 1 drew 85 clips the model scored as "right length, 10-50% of sounds differ" and 15 it scored perfect.
Of 60 labelled from the first group: 32 unusable (24 incomplete, 5 different ayah, 3 noise),
25 correct, 3 with a mistake. Of 7 labelled controls: 7 correct.
On the 25 clips judged correct, the checker (threshold set on studio reciters) flagged 48 of 95
words (51%), 18 strongly (19%). On the 7 controls it flagged 1 of 26 words, none strongly.
The 3 mistake clips: one vowel changed, one sound dropped, one with short ghunnah and madd; the
checker flagged the marked word in 2 of the 2 clips that had a word mark, among other words.
Reading: on phone audio the model's mismatch score does not separate incomplete clips, correct
recitation it mishears, and real mistakes (median mismatch 0.21 vs 0.22). The gap is phone audio,
not the checker logic. Small sample; one labeller; no second pass yet.

## Words-and-vowels mode on the hand-labelled phone clips
Same 35 usable clips (32 correct, 3 with a mistake). Each mode gets its own thresholds, set on the
two held-out test reciters (last shard of each, suras 78-114, 1,168 clips, 4,158 words): flag at
1% false flags, strong at 0.1%. Full mode 2.08 / 4.91, words and vowels 0.70 / 4.91.
Shares are of sound groups (words as the sound script splits them; 6 clips have fewer groups than
display words).

| Correct clips | Words | Full, flag | Full, strong | Words and vowels, flag | Words and vowels, strong |
|---|---|---|---|---|---|
| All 32 | 112 | 30% (22-40) | 17% (9-25) | 42% (32-52) | 18% (10-26) |
| 25 from the unsure group | 89 | 38% (29-49) | 21% (12-32) | 53% (43-63) | 22% (13-33) |
| 7 controls | 23 | 0% | 0% | 0% | 0% |

95% intervals from resampling clips. Controls were picked because the model already decoded them
almost perfectly, so their 0% says little. The earlier 51% (48 of 95) used another threshold set
and display words; rerun here, full mode gives 38% on the same 25 clips.

Mistake clips (F = flagged, S = strong):
- 103:3, vowel changed in word 6: flagged in both modes (2.99). Words 3, 5 and 7, judged correct,
  score higher and are strong in both.
- 104:4, sound dropped in word 1: missed in full mode (1.07), flagged in words-and-vowels mode.
  Word 3, judged correct, is strong in both.
- 104:3, short ghunnah (word 1) and short madd (word 2), no word mark: full mode flags word 2;
  words-and-vowels mode flags words 1 and 2 on other sounds. Words 0 and 3 are strong in both.

Reading: dropping tajweed detail does not lower the flags. At 1% it flags more, because the
threshold falls further on studio audio than the scores do on phone audio. Strong flags stay the
same (21% vs 22%): the words that fail clearly fail on letters and vowels, not on lengths. The gap
is acoustic, which points to the phone fine-tune. One labeller, small sample.

## Legal reading choices (best of several readings per word)

The checker scores each word against one reading (madd 4 everywhere). `qafft/variants.py`
phonetizes a span under a short list of other legal Hafs choices and keeps, per printed word
(matched through `wordmap.map_tokens`), the lowest badness.

- Choices that change the sound tokens: madd munfasil length (2, 4, 5), madd muttasil and its
  pause form (4, 5, 6), madd aarid (2, 4, 6) and madd leen at a pause (up to the aarid length),
  ghunnah of noon/tanween into lam and ra, seen or saad in the four words that allow both, and
  the fixed places of sakt or not (iwaja, marqadina, man raq, bal ran, maaliyah), saktat before
  hamz, and a few single words (ءآلذكرين, يلهث ذلك, اركب معنا, the meem of الم الله). Only the
  madd lengths, ghunnah of lam/ra and seen/saad are varied for now; the single-word cases are
  rare and the sakt places are outside the sets measured here.
- On the long correct recitation (sura 2) the lowest score came from the default reading for
  94% of words, from short munfasil for 4% and short aarid for 2%; other choices never won.
  At the default threshold false flags on the test half fell from 0.85% to 0.37%. With
  thresholds re-set on the first half, the 1% threshold drops from 9.16 to 5.82, so the checker
  can be stricter at the same false-flag rate.
- On 32 correct phone clips nothing changed (2 of 121 word scores lowered, no flag removed):
  their false flags are not about free madd lengths.
- Planted mistakes on one studio shard: catch at 1% and 0.1% unchanged (99.3%, 98.7%). A
  planted madd length that equals another legal choice would be accepted by design; the
  planted set did not show that cost.
- Recommendation: use best-of for the madd families (cheap: 2-8 readings per span). Thresholds
  must then be set with best-of scores too.

## Real recitation mistakes (three local sets, `eval/real_mistakes.py`)

First check on mistakes people actually made, not planted ones. Thresholds come from correct
words of the studio validation reciter (`studio-val`, 300 segments, 2,029 words), not from the
test sets. Labels per printed word: correct, word/vowel mistake, or tajweed-only mistake.

- `mdd_c` (sura 112, 397 clips with audio): each clip names the wrong words and an error type;
  "vowels" (and "vowels and tajweed") count as word/vowel, the rest (mostly qalqalah on the
  final dal, some articulation notes) as tajweed-only. AUC 0.77 tajweed judged, 0.79 words and
  vowels. At the 1% studio threshold: catch 44-45%, but 6-8% of correct words are flagged, so
  precision is about 50%. At 0.1%: catch 5%, precision under 40%.
- `mdd_d` (441 Hafs chunks, other readings dropped): word -> type dict; wording, letters and
  vowel marks count as word/vowel. AUC about 0.5 and 22% of correct words flagged even at the
  0.1% threshold: the checker does not work on these chunks as they stand (phone audio, chunk
  text may not match the audio, very few labels).
- `learner_b` (158 clips of the end of 5:109): word/vowel mistakes found by comparing the said
  phonetic transcript with the canonical one in the words-and-vowels classes (ghunnah and
  hidden-noon differences counted as tajweed). 41 clips fully correct, 26 word/vowel mistakes,
  297 tajweed-only. False flags stay near the studio rate (0.4-0.5%), catch is low (8-15% at 1%).

What this says: studio thresholds do not carry over to phone audio (false flags 6-23% instead
of 1%), and on real mistakes no threshold reaches 90% precision on any set, even when chosen on
the test set. The 0.1% threshold rests on two words of 2,029, so it is rough. Most useful next
steps: thresholds (or a calibration) from correct phone recordings, training on phone-quality
audio, and more labelled real word/vowel mistakes.
## Studio ayah clips from held-out voices (whole sura recordings cut by ayah timings)
Commands: eval/studio_clips.py heldout / score / report; clips cut by data_tools/studio_clips.py.
Voices r06, r10 (never trained on) and r08 (the base model may have heard it): suras 2 (ayahs
1-100), 18, 36, 55, 67, 78-114; about 920-960 ayah clips and 6,500-6,700 printed words each. Each
ayah is scored as a whole ayah (it ends at a pause, so the ayah form is what is said), every
printed word in both modes. Thresholds from `studio-val` (moshaf 19.0, 300 clips, 2,068 sound
groups): tajweed 0.77 (1%) / 9.15 (0.1%), words and vowels 0.64 / 9.02.

| Voice | Mode | Words flagged (1% thr) | Ayahs with a flag | Words flagged (0.1% thr) | Ayahs | Own-half split: words / ayahs (1%) |
|---|---|---|---|---|---|---|
| r06 | tajweed | 8.5% | 36.8% | 0.98% | 6.5% | 0.67% / 6.3% |
| r06 | words and vowels | 3.8% | 20.8% | 0.49% | 3.1% | 0.11% / 0.8% |
| r10 | tajweed | 5.4% | 30.6% | 0.26% | 1.7% | 0.52% / 5.0% |
| r10 | words and vowels | 2.8% | 15.9% | 0.20% | 1.3% | 0.73% / 7.2% |
| r08 | tajweed | 5.9% | 33.0% | 0.17% | 1.2% | 1.64% / 15.9% |
| r08 | words and vowels | 2.4% | 13.5% | 0.15% | 1.1% | 2.96% / 23.3% |

Word badness median is about 0.1 in every voice; the 99th percentile is 3.5-8.9 against 0.77 on
`studio-val`, so the `studio-val` 1% threshold sits far too low for long ayah clips in new voices:
the tail, not the bulk, moves. A planted mistake per clip is caught 99% of the time at that flag
threshold (tajweed) and 79% in words-and-vowels mode, 86% / 73% at the strong threshold.
The own-half thresholds are much higher (3-10) and are noisy at 0.1% (a few thousand words).

Word edges from CTC forced alignment against the per-word timing tables (r10: 5,052 words; r08:
4,339): start error median 0.07-0.08 s, 90th percentile 0.19 s; end error median 0.13-0.14 s,
90th percentile 0.41-0.44 s. Both edges come early (median -0.05 s start, -0.12 s end): CTC
spikes sit at the start of a sound and trailing lengthening is cut short, so durations from the
alignment alone will run short; end edges need a correction or a duration model of their own.

Limits: studio audio; `studio-val` thresholds are on sound groups of segments up to 20 s, here
printed words of whole ayahs up to 45 s (17 longer clips skipped); the timing tables are taken
as truth and have their own error; planted mistakes are text-side.
## Other word scores on planted mistakes (segment-free GOP, competing-token margin)
Same protocol as eval/evaluate.py --shards on the two held-out reciters (one shard each, suras
78-114, 200 clips each: 1,343 correct words, 400 planted mistakes). Scripts: eval/wordscores.py,
eval/wordscores_report.py; raw scores in data/eval/wordscores/. Thresholds come from the same
correct words (in-sample), so catch rates are optimistic for every score alike.

| Score | AUC | Catch at 1% | Catch at 0.1% |
|---|---|---|---|
| current (weakest sound, extra sound) | 1.000 | 99.5% | 95.3% |
| segment-free GOP | 0.964 | 81.0% | 51.0% |
| competing-token margin | 0.955 | 91.0% | 90.5% |
| max of rank-normalised current, GOP, margin | 1.000 | 99.5% | 96.0% |

By kind at 1%, GOP: vowel 1.00, ghunnah 0.93, letter 0.89, madd 0.65, doubling 0.56, qalqalah 0.00.
Margin behaves like the weakest-sound score (madd 0.70, qalqalah 0.00). Words-and-vowels mode
(to_basic): current 100% / 99.4%, GOP 82.9% / 51.1%, margin 91.9% / 91.3%.
Reading: GOP and margin only ask whether the expected sounds are the best explanation; the
current score's extra-sound check is what catches madd and qalqalah, and on its own extra sound
catches only 10%. GOP's one-token edits miss doubling and length changes, which a CTC model
can explain without a better single edit. Combining adds 3 words of 400 at 0.1%, within noise.
Keep the current score in the verifier.
On the long correct recitation in a new voice (sura 2, thresholds from the first half, test on
the second), false flags at the 1% / 0.1% thresholds: GOP 0.75% / 0.31% tajweed judged and
0.82% / 0.10% words and vowels; current 0.85% / 0.03% and 1.02% / 0.20%. No clear gain.
