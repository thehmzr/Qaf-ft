# Qaf-ft research summary (condensed from three research threads, 2026-11-20)

Evidence note: numbers below were read by the research threads in model cards, leaderboard
files and arXiv papers. Tarteel/NVIDIA pages were blocked, so Tarteel internals are
search-snippet or third-party only. Two threads (datasets, training methods) still pending.

## Key findings
1. Text-output models "auto-correct": on the Quran-Lab board, of 100 planted mistakes only
   1.2-4.9 appear in the output of text models (Whisper, Jais, FastConformer text fine-tunes)
   versus 19-27 for phoneme-CTC models (muaalem 26.6, TajweedFormer V4 24.6).
   => output unit must be phonemes, no language model.
2. False flags are the unsolved problem. Best published phoneme system (IQRA 2026 winner)
   wrongly rejects 1.75% of correct phonemes => a false flag in about half of 40-phoneme
   ayahs. Phoneme checkers flag 21-51 of 100 correct recitations on the Quran-Lab board
   (muaalem 50.5, TajweedFormer V4 21.0, Tarteel whisper-base 40.3).
3. Real mistake data matters most: IQRA 2026 winner F1 0.46 (synthetic only) -> 0.68 after a
   second stage on ~1.5 h of real mispronounced speech -> 0.72 with ensemble.
4. Small CTC models trained on the right data mix beat huge general models on Quran audio:
   Abdou/quran-asr-small 33M MIT: WER 0.56 studio / 2.13 phone; Whisper-large-v3 7.09 / 23.86.
5. Tarteel production (third-party measurement, QL-600): 9.56 WER overall, 16.17 on real phone
   audio; word-level mistakes + tashkeel toggle; no tajweed; server-side only.
6. Tajweed today: qalqalah F1 0.95, mushaddad ghunnah 0.87, madd RMSE ~0.5-1 harakah,
   ikhfa F1 0.45 (one-verse learner benchmark, muaalem paper). Ship as advisory.

## Base candidates
- nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0: 115M, CC-BY-4.0, diacritics, 390 h Quran
  in pretraining; proven sherpa-onnx/Flutter path; int8 ~125 MB; fine-tune 740 h in ~3 h on an
  8 GB GPU (CTC head). Not streaming as shipped; cache-aware conversion documented.
- facebook/w2v-bert-2.0 / obadx/muaalem-model-v3_2: 606M, phonemes + 10 sifat heads, best
  offline teacher; licence conflict (HF card MIT vs repo QML); not streaming.
- Zipformer2 / small Conformer-CTC 16-65M from scratch: tops the checker board; 19-73 MB int8;
  needs ~1,500 h incl. phone audio; no permissive pretrained Arabic checkpoint.
- nvidia/nemotron-3.5-asr-streaming-0.6b: 638M native streaming, Arabic, OpenMDW licence;
  server-class.
- Not suitable: Whisper family (hallucination, 30 s windows), Citrinet (no Arabic checkpoint,
  no cache-aware streaming, superseded), Parakeet/Canary/Kyutai (no Arabic), MMS/ArTST (NC).

## Method recommendations
- Quran phonetic script labels (obadx quran-transcript): durations as repeated symbols, sifat heads.
- Verification separate from recognition: CTC likelihood ratio (segmentation-free GOP) of the
  expected ayah vs alternatives; per-phoneme thresholds at a fixed false-reject budget; three
  states (correct / mistake / unsure); benign events (repeats, self-corrections, basmala).
- Never condition the recogniser on the reference text (recall collapses: F1 55 -> 40.5).
- Usable data: EveryAyah (CC-BY), tarteel-ai/tlog real phone recordings (CC-BY, gated, noisy
  labels), obadx muaalem-annotated-v3 (848 h), QUD universal ayahs, RetaSy learners,
  qdat_bench, Iqra_Extra real errors. Tadabur and IqraEval data are non-commercial.

## Compute estimates (for ~800 h)
- FastConformer 115M CTC-only: 5-10 T4-hours (free on Kaggle) or <$5 rented.
- FastConformer hybrid + streaming conversion: 20-40 T4-h or $6-25.
- Small Zipformer/Conformer from scratch (1,500 h x 20 epochs): 80-150 T4-h or $25-75.
- w2v-BERT 2.0 600M: 70-90 T4-h per epoch or $25-50 per epoch on A100.
- HF Jobs: T4 $0.40/h, L4 $0.80/h, A100 $2.50/h. Kaggle ~30 free GPU-h/week.

Targets to claim "best": phoneme false-reject <= 0.3% on correct recitation (published best
1.75%), recall >= 50% on real errors, WER <= 0.6 studio / <= 2 phone, int8 size <= 75 MB.

## Datasets (fifth thread, condensed)
Train: obadx/muaalem-annotated-v3 (848 h, 22 male experts, phoneme+sifat labels, MIT tag);
everyayah.com recitations de-duplicated (drop Warsh, Muallim, translation folders);
tlog clean (393k real phone clips; open mirror deepdml/tlog-clean-sf16k 55 GB; labels inferred
from filename, must be filtered by ASR agreement); freesound commercial noise set.
Reciter-disjoint test: Luxbg/quran-muaalem-aligned test_unseen (2 reciters, 60 h, word timings, MIT).
Real-mistake data (all small): RetaSy 1,166 labelled clips (396 correct / 476 mistake, clip-level,
inter-rater 0.63, ignores tajweed, no licence stated); obadx/qdat_bench 159 learners one verse
(tajweed-level, Apache-2.0); QuranMB.v2 1,642 scripted MSA errors (NC); sobolev210 46 Hafs
clips with word-level errors (MIT); Surah Ikhlas labelled set 1,506 clips (851 error).
EveryAyah test split shares reciters with train: useless for generalisation claims.
Isolated letters: Reinjin/Pelafalan_Huruf_Hijaiyah 8,904 clips = 28 letters x 3 short vowels
(Indonesian students, AFL-3.0); mark-muhammad/makharij-huruf 391 drills with sukoon/shaddah (MIT).
Licensing risks: everyayah.com audio has no licence (Tarteel card: "use at your own risk");
IqraEval, Tadabur, QUA, Quran-Lab data are non-commercial or forbid paid features; muaalem
checkpoints carry extra conditions (no ads, free-access path); Tanzil text needs attribution.
No open set has real, word-located mistakes across the full Quran: an own expert-labelled
test set is needed for a trustworthy false-flag number.

## Kaggle / Zenodo / Mendeley check (direct access, 2026-11-20)
Kaggle (84 hits read, 12 descriptions read):
- ahedya/labeled-quran-dataset: Al-Fatiha split word by word, ~175-220 different reciters per word
  (762 MB, custom licence). Useful for word-level alignment tests and speaker variety.
- alawdisoft/tajweed-dataset: clips by noon-sakinah rule (ikhfa/izhar/idgham/iqlab) per sheikh, CC0, 1 GB.
- abdelmonemhatem/tts-mispronunciation-detection: 2,000 recordings, 16 speakers, expected vs
  realized phonemes with timings, 1,219 correct / 781 with errors, speaker-independent splits (not Quran).
- abdelrhmansalah22/arabic-speech-mispronunciation-detection-dataset: 100 Egyptian children (2-8 y),
  top-100 words, expert segmental error labels, CC0 (not Quran; child speech).
- Isolated letters: alharthalhajhussein (1,439 clips, 28 letters), mazinali1990 (1 s clips),
  jabirmuktabir hijaiyah-fathah (Apache-2.0), amnaasif/arabic-short-vowels-audio-dataset (referenced).
- abdo3id/female-quran-recitation (205 MB, licence unknown): rare female voices.
- abdullahkhansherwani/muaalem-stage: 87 GB Kaggle copy of the muaalem data (handy for Kaggle training).
- atique/qaida-dataset is Urdu OCR images, not audio.
Zenodo: nothing with mistake labels; 5879098 "Qur'anic Recitation Styles and Reciters" (27 GB, CC-BY) is style classification.
Mendeley: Surah Al-Ikhlas Error Detection Dataset confirmed at source: 1,506 records, labelled by 5 Quran experts, CC BY 4.0.
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
