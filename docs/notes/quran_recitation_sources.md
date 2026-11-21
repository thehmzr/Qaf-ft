# Study notes: Quran recitation sources (for Qaf-ft)

Eleven sources on Quran recitation data, models, checking, cutting and tracking, read at the
source. They add to `docs/datasets.md`, `docs/how_others_build.md` and
`docs/research_summary.md`. Facts already recorded there are not repeated; points that
correct or extend them are marked **Flag**.

Conventions

- Datasets, models and their papers are named by the project's labels (see `docs/datasets.md`).
  People are named by role only.
- Anything that goes beyond what a source states is marked **(inference)**.
- "Ordinary reciter" means a non-expert reciting on a phone, the project's target user.
- PER = phoneme (sound) error rate. FR = false rejection (a correct sound flagged).

How each source was read

| # | Source | How read |
|---|---|---|
| 1 | The expert set's paper (v2) | Full HTML, all sections and appendices (Tables 1-17, Algorithm 1) |
| 2 | The mdd_e paper (v2) | Full HTML. v2 is the latest version; no separate "v2 paper" exists. The 2026 version of the mdd_e test set is described only on the dataset cards |
| 3 | A learner_b method paper | Full PDF text (16 pages) |
| 4 | The phone set's paper (2021) | **Full text not accessible**: the PDF is behind a browser check that was not bypassed. Read: abstract, metadata, both reviews and the decision through the review site's search API |
| 5 | The learner_a card | Card README in full |
| 6 | The studio_d paper (v1) | Full HTML |
| 7 | The bench_a card | Public rendered card in full (gated files not opened) |
| 8 | The model_b card | Public rendered card in full (gated files not opened) |
| 9 | An open live recitation tracker | README, sequence-detection doc, all backend Python files |
| 10 | A word-timing helper tool | English half of the README in full, licence, requirements; the 447 KB scripts were not read line by line |
| 11 | A vendor case study on real-time Arabic speech recognition | Full page, plus the vendor developer blog post it links to (now reachable; earlier threads found it blocked) |

---

## 1. The expert set's paper

**What it is.** The paper behind the expert set, learner_b's benchmark subset, the phonetizer
package and model_a. Version read: v2, 27 Aug 2026. The authors call the Quran Phonetic
Script (QPS) their main contribution.

**Quran Phonetic Script.**
- Two output levels: a phoneme level (43 symbols, Table 10) and a sifat level of 10
  attributes (Table 11). Vocabulary per level: phonemes 43; tafkheem/tarqeeq 3;
  shidda/rakhawa 3; the other eight attributes 2 each.
- Designed from classical tajweed scholarship, not IPA. Covers all Hafs tajweed rules except
  ishmam (and pausing with rawm or ishmam, per A.6). Three error classes are targeted:
  articulation (makhraj), attribute (sifat) and rule errors.
- Length is written by repetition: a 4-count natural madd is four alif symbols; madd lazim
  six; noon mushaddadah is three noons; ikhfa is three hidden-noon or hidden-meem symbols;
  idgham with ghunnah into yaa/waw is shown by doubling; sukoon is the absence of a vowel;
  imala uses two dedicated symbols; rawm uses a "dama mokhtalasa" marker.
- Built in two steps: Imlaei to Uthmani (an annotation tool fixed mismatched words), then 26
  ordered regex operations from Uthmani to QPS (listed in A.2: hamzat wasl, iltiqa
  al-sakinayn, ghunnah, tasheel, imala, madd, qalqalah, and so on). Sifat are then derived;
  inhiraf and leen are left out on purpose.
- Per-mushaf "Moshaf attributes" (A.4) fix every free choice: madd munfasil 2/3/4/5,
  muttasil, aared, leen, ghunnah of lam and raa, several sakt sites, takbeer, and others.
- Their argument for multi-level output: one flat vocabulary for all combinations would need
  99,072 tokens; 11 parallel levels need far fewer.

**Data pipeline (848 h).** Covered in `datasets.md` and `how_others_build.md`. Additions:
- Text source: a widely used open Quran text, chosen over another standard printed text for
  stability.
- 893 h collected before filtering; 22 reciters (15 from Egypt).
- Off-the-shelf voice-activity detectors worked poorly on recitation, so they trained a
  w2v-BERT frame classifier on pause labels derived from studio_b (augmented with background
  noise on 40% of clips, time stretch 0.8-1.5x, 1 s sliding windows). One epoch, lr 5e-5,
  batch 50, max 20 s. Test on an unseen mushaf: accuracy 0.9935, F1 0.9948.
- Segmentation was checked by hand on 50-75 random clips per mushaf plus all clips under
  3 s and over 35 s; a mushaf under 99% was to be excluded (25.0 measured 90%).
- Transcripts came from model_d (WER 5.75%), using 10 s windows for clips over 30 s.
- Tasmeea verification (Algorithm 1): for each segment, try start positions and window
  sizes in a 30-word window (6 words overlap) of the current surah, score
  1 - edit_distance / length, accept at > 0.85 and advance the pointer by the match; on
  rejection output nothing, widen the next search and advance one ayah. Rejects go to a
  person (5,400 of 286,537).

**Model.** w2v-BERT 2.0 (600M) encoder with 11 linear CTC heads ("multi-level CTC"), no
language model "because we aim to capture what the user actually said". Loss is a weighted
sum: phonemes 0.4; shidda/rakhawa and tafkheem/tarqeeq 0.0605 each; the other eight
0.069875 each. One epoch, adapter layer, AdamW, constant lr 5e-5, batch 64, bf16, clips
up to 30 s (3k of 250k dropped), background noise plus TimeStretch and GainTransition. 7 h
on one large data-centre GPU. Pilot runs on 4 mushafs found constant lr best and
**SpecAugment "significantly degraded performance"**.

**Results.**
- Expert test (mushafs 19.0, 29.0, 30.0; male experts): phoneme PER 0.543%; sifat PER
  0.08-0.17%; "average PER" 0.21% is the mean of all 11 levels, so 10 easy binary levels pull
  it down.
- learner_b benchmark (159 learners, one ayah, 120 women): phoneme PER 5.78% (bootstrap
  6.03 ± 0.53); sifat PER 0.93-3.10%; average 1.94%. Tajweed: noon mushaddadah F1 0.869
  (partial class recall 0.66, precision 1.00); qalqalah macro-F1 0.953; ikhfa average F1
  0.453, with the "partial ikhfa" class never predicted (recall 0) and plain noon recalled
  only 47%. Madd RMSE in counts: natural madd 0.40-0.55, munfasil 0.69, aared 1.03.
- Error analysis (Fig. 10): boundaries near perfect; a missed fatha; madd runs come out
  unbalanced, with the last repeated symbol stretched.

**Flag.** The abstract and conclusion say "84.7% accuracy on female reciters". Table 9 shows
0.8470 is `avg_tajweed_acc` over all 159 clips (120 women, 39 men). No women-only number is
given. Treat the claim as unsupported.

**Released items.** Code, models and data are released (paper; dataset links in Table 2).
The model_a card on the platform is an empty template; usage terms are kept with the source
files. The code repository's inference call takes the phonetized reference next to the audio;
it is used to explain differences and to line up sifat, not to decode **(inference from the
README code)**.

**Stated limitations.** Letter-specific attributes cannot be learned from correct speech:
the model will not catch ض without istitala or ر without tikrar; rare rules (imala, rawm,
tasheel) are thin; only male experts; Hafs only; the learner_b benchmark has one annotator
and one ayah.

**Relevance.** The closest system to Qaf-ft: same label script, same training set, CTC
with no language model. Its learner/expert gap (5.8% vs 0.54% PER, about 10x) is the same
shape as our phone gap. See Application (a).

---

## 2. The mdd_e paper

**What it is.** The organisers' paper for the mdd_e task: a phoneme set, training data, the
first public test set (mdd_e v1) and SSL baselines. Version v2, 12 Jun 2025. Scope is Modern
Standard Arabic reading of Quran text, **without tajweed**.

**Task.** Mispronunciation detection and diagnosis: recognise the phonemes actually said and
compare with the target sequence.

**Data.**
- Training set: 82.37 h of voices_a Arabic (Table 1: 79 h train, 3.33 h dev) re-vowelized
  with an in-house vowelizer and assumed correct, plus Quran recitation samples.
- TTS: seven single-speaker voices (5 male, 2 female); 26 h of correct speech plus 26 h with
  errors made by changing four random characters or diacritics per sentence from a
  confusion matrix (examples ت/ط, س/ص, غ/خ), built from phoneme-similarity data.
- mdd_e v1 test: 98 verses read by 18 native speakers (**14 women, 4 men**), about 2.2 h,
  in MSA at normal pace, ignoring tajweed, deliberately producing the error the recording
  tool highlighted; speakers read each sentence silently first.
- Phoneme set: an MSA TTS phonetizer reduced to 68 phonemes (emphatic-context vowel
  variants merged; geminates are doubled symbols).

**Method.** SUPERB setup: frozen SSL encoder (wav2vec2, HuBERT, WavLM, all 94M English-only;
mHuBERT-147 multilingual), weighted layer sum into a 2-layer 1024-unit BiLSTM with CTC.
Adam lr 1e-4, batch 16, 12.5k updates, greedy decoding.

**Metrics.** Hierarchical MDD counts: true accept, false reject, false accept, correct and
error diagnosis; precision = TR/(TR+FR), recall = TR/(TR+FA), F1.

**Results (Table 2).** Best: mHuBERT on voices_a+TTS, F1 29.88%, precision 18.70%, recall
74.29%, **false rejection of correct phonemes 12.65%**, PER 16.42%. TTS-only mHuBERT: F1 29.00.
All models: precision 13-19%, FR 12.6-21.1%.

**Usage terms.** Usage terms are kept with the source files.

**Limitations.** Acted errors; MSA, not recitation; mostly women speakers; frozen encoders;
no speaker or confidence analysis.

**Flag.** `research_summary.md` quotes 1.75% phoneme false rejection for the "2026 mdd_e task
winner". That figure is not in this paper; it comes from the later challenge. The paper's
own baselines reject 12.65% or more of correct phonemes.

**Relevance.** A metric template (FR on correct sounds reported next to recall) and an
outside reference for letter and vowel substitutions. Weak fit for tajweed or for our
alphabet.

---

## 3. A learner_b method paper

**What it is.** A method paper that uses the learner_b data (it is not the learner_b release
itself). Three rules: madd munfasil ("separate stretching", 4-5 counts), noon mushaddadah
("tight noon", 2-count ghunnah) and ikhfa ("hide").

**Data.** learner_b's parent set: more than 1,500 clips of part of 5:109 by speakers of both
genders and different ages, **recorded through a messaging app, WAV, 11 kHz, mono, 16-bit**.
Published on the platform.

**Method.** Pre-emphasis 0.97, MFCC (32 ms tapered frames, 40 mel filters), three 256-unit
LSTM layers, three dense layers (16, 16, 8), dropout 0.2, one sigmoid output per rule,
RMSprop lr 0.0005. **Random 90/10 train/test split of clips.**

**Results.** Accuracy: munfasil 96%, ikhfa 95%, noon mushaddadah 96% (recall 95/95/98, F1
95/95/97). Baselines on the same split: random forest 91/83/91, KNN 91/81/94, SVM 81/71/85,
logistic regression 88/79/91.

**Usage terms.** Not stated in the paper.

**Limitations (inference).** Each speaker recorded the same verse about ten times
(`datasets.md`, learner_b), so a random clip split puts the same voices in train and test.
One verse, one judgement per rule per clip, no location.

**Flag.** `how_others_build.md` §4 says simple classifiers on learner_b "look far better than
they generalise". The likely cause is this speaker leakage **(inference)**. Compare the
speaker-disjoint learner_b benchmark result in source 1: ikhfa F1 0.45, not 95%.

**Relevance.** learner_b audio (both the parent set and the benchmark subset) is narrowband
messaging-app audio (11 kHz, so nothing above 5.5 kHz). That makes the learner_b benchmark a
phone-like test, not a studio one **(inference)**.

---

## 4. The phone set's paper (2021)

**What it is.** A submission to a 2021 machine-learning conference's datasets and benchmarks
track, **rejected**. Not on arXiv. Only the abstract, reviews and decision could be read.

**Data (abstract).** 25,000 clips, 67.39 h, "a wide variety of recitation styles,
proficiencies, and speeds", collected over six months from more than 1,200 individuals of
different ages, genders and ethnicities. It also proposes a standard schema for paired
Quran audio and text and gives baseline results.

**Collection and labelling (reviews).**
- Volunteers were unpaid; reviewers noted this makes cleaning harder.
- Quality control combined crowd binary checks (is the clip complete; does it match the
  verse) with automatic label generation. The paper itself describes "serious limitations"
  of both (its sections 5.1 and 5.2), and says a four-way label split was tried and had
  limits. Reviewers could not tell what was finally used.
- Demographic data is "hardly present" in the download. Text labels and phoneme labels are
  under-explained.
- The only experiment combined this data with studio_b.
- Decision: reject, for lack of analysis of crowd verification and automatic labelling.

**Key numbers.** 25k clips, 67.39 h, 1,200+ speakers. No baseline numbers could be read.
Source 1 lists it as "v1" of this data (25K utterances, 68 h).

**Usage terms.** Not readable. The forum points to a dataset address that now returns 404.

**Limitations.** Labels are the intended verse with weak checks, the same problem as the
phone set. The tracker README (source 9) claims the paper reports ">98% accuracy in
word-level alignment"; that could not be checked.

**Relevance.** Shows that this kind of app-collected data has been weakly labelled from the
start. It is a different and much smaller set than the phone set (about 393k clips); how
the two relate is not stated anywhere read.

---

## 5. The learner_a card

Mostly covered in `datasets.md` §13 and `how_others_build.md` §3. Additions from the card:
- A third quality number: **0.89 agreement between the algorithm's labels and expert
  judgements**, next to crowd accuracy 0.77 and inter-rater agreement 0.63.
- Each row's `annotation_metadata` holds every annotator's label and quality scores
  (control tasks solved, MCC, accuracy, F1). Annotator quality varies widely: the card's own
  example row has annotators with accuracy 0.92, 0.57 and 0.83.
- `reciter_gender`, `reciter_age` and `reciter_country` fields exist (often "Unknown"), so
  results could be split by gender for labelled rows.
- The annotation platform is named on the card; the collection app is a volunteer-based
  crowdsourcing API.
- Usage terms: none on the card (unchanged).

**Relevance.** The weighted vote (annotator accuracy on control tasks) is a usable recipe
for our own labelling. Data stays evaluation-only until terms are obtained.

---

## 6. The studio_d paper

**What it is.** The paper for studio_d. v1, 21 Apr 2026. Usage terms are kept with the
source files (`datasets.md` §3).

**Method.** Long recordings from public archives; a large language model classifies titles
and descriptions and pulls out surah and reciter; Whisper large-v3 with a word-timestamp
add-on gives word timestamps; an "Ayah Alignment Module" matches transcript segments to
canonical ayah text from a public Quran API by cosine similarity of text embeddings; a
recitation-stop segmenter (the expert set's paper's segmenter approach) fixes each clip end
after adding a 5 s buffer; near-duplicates per reciter and ayah (audio embeddings, cosine
> 0.9) are merged with union-find.

**Pipeline evaluation (Table 1, five professional reciters).** Ayah coverage: embedding
matching with their own fine-tuned ASR 96.63%; with model_d's larger sibling (a Quran-tuned
Whisper) 95.50%; with Whisper small 82.57%. Fuzzy text matching: 86.03%, 87.23% and 72.80%.
Coverage means "found and cut"; whether each cut is correct is not measured.

**ASR baselines on studio_d (Table 3, WER/CER after removing diacritics).** A Quran-tuned
Whisper (74M) 8.7/6.5; five general ASR models 11.2/8.1, 15.1/11.2, 24.3/14.0, 25.2/9.9;
Whisper small 29.2/16.1; two multilingual self-supervised wav2vec 2.0 models 51.1/16.6 and
57.4/21.9.

**Related-work survey.** Thin. It lists only three datasets: a 6,689-clip, 12-reciter
reciter-identification set on the platform; studio_f (226,129 ayah clips, 30 reciters, no
word timing); another multi-reciter set (187,080 samples, 30 reciters). It leaves out
studio_b, the phone set, the expert set, learner_b, learner_a and the mdd_e task.
**`datasets.md` is far more complete; nothing here needs adding to it** apart from the
187,080-sample multi-reciter set (professional reciters).

**Limitations (stated).** Some reciters miss ayahs (mainly ASR failures); word timestamps
"not always precise" because the aligner was not built for Quran audio.

**Flags (inference).**
- Section 3.3 says the pipeline uses Whisper large-v3, but section 4.3 says the adopted
  setup is the authors' own fine-tuned model. Which one built the released data is unclear.
- The labels come out of an ASR-and-matching pipeline, so ranking ASR models against them
  is partly circular; a model like the pipeline's ASR is favoured.
- Embedding similarity is built to ignore small differences. That is good for finding an
  ayah, and it is why it must never be used to judge a word.

**Relevance.** A recipe for cutting long recitations (find the ayah, then refine the end
with a pause detector). No use as a test set. Not used for training.

---

## 7. The bench_a card

Configs and quirks are in `datasets.md` §10. Additions:
- The card's warning: "**[the phone config] is the one that matters for apps; studio numbers
  flatter every model.**" The phone holdout clips are called "the hard, real-world case".
- Scoring: WER and CER after stripping marks, Quranic signs and tatweel and unifying hamza
  forms and alef maqsura; an extra "alef-insensitive" column strips alef to remove
  Uthmani-vs-Imlaei spelling differences.
- Run rules: Arabic language hint for every model; greedy or default decoding; **no
  language-model rescoring and no forced alignment; no model ever sees the reference.**
- A public leaderboard space accompanies the benchmark on the platform.
- v1.1 fixed ten references in the recitation-library holdout config where the reciter
  repeats the ayah (in one clip: 51:56, 51:57, 51:56). The wrong references came from
  upstream per-ayah files that were presented as human-reviewed. Lesson: per-ayah clips from
  any source can hold repeats.

**Flag.** The platform lists two small Quran ASR models as "trained or fine-tuned on" the
benchmark. On the model cards this comes from evaluation metadata (the smaller model's
training data lists only the phone set and the mdd_e task's training set, and it says no
benchmark recording was used). That card also reports that **one phone clip of the
benchmark (36:71) was the same recording as one in its training data, uploaded twice under
different ids.** So the phone set has duplicate recordings under different file names;
removing the 200 holdout ids by name is not enough.

**Relevance.** The idea of "correct recitation tested separately, every legal reading
accepted, phone audio reported separately" is already adopted. The data stays on hold
(its terms bind a model that is merely evaluated on it).

---

## 8. The model_b card

**What it is.** A 65.5M-parameter streaming Conformer-family encoder with a CTC head that outputs a
251-symbol phonetic alphabet built on the phonetizer package (Hafs, murattal). "No language
model, anywhere": no transducer, no rescoring, no text prior. Superseded by a newer version
of model_c (offered through an SDK); this card stays as a reference. v3.1 is a madd
fine-tune with the same interface.

**Architecture and streaming.** Cache-aware streaming CTC; trained only at chunk sizes 8,
16 and 24 frames; ONNX export for an on-device inference runtime (chunk 48 input frames,
0.48 s per call); input must be 80-dim log-mel filterbank features computed in the classic
speech-toolkit style with its default window: "a slaney-mel front end roughly doubles the
error rate". Packages in the phone's native model format (float8 63 MB, float16 124 MB) are
split into two functions so that both run on the phone neural accelerator; the previous
version took 6.8 ms per chunk on a recent high-end phone (70x real time). Numerical fixes for
fp16 hardware are listed (overflow guard, max-subtracted log-softmax, no fp32 epsilons).
`tokens.txt` is required because blank is id 250, not 0. A `decode_with_confidence.py`
script gives per-symbol confidence; the card advises using the margin at the emission peak,
since margins sampled elsewhere differ between runtimes.

**Training data.** About 5,400 effective hours per epoch for 10 epochs: complete mushafs by
36 studio_b reciters, 1,161 more reciters cut from public archives, real phone recordings,
concatenated multi-ayah windows, same-ayah repetition clips, and a broad-Arabic tier; noise,
reverb, wind, speed and tempo augmentation. Average of the last 3 epochs.

**Results (PER, benchmark v1.1).** Studio benchmark reciters 1.43% (v2: 5.19%); real phone
3.65% (7.92%); unseen professional reciter 9.10% (11.91%). An earlier 4.92% for the unseen
reciter did not reproduce on the shipped weights and was corrected. An audit found 24 stray
clips of the "absent" studio reciters in training. The emphatic-consonant gap (ص ض ط ظ vs
plain letters) is "statistically indistinguishable from zero".
On the learner_b benchmark, madd RMSE (counts) vs model_a: qalo alif 0.210 vs 0.449; qalo
waw 0.514 vs 0.456; allam alif 0.363; munfasil 0.887 vs 0.687; aared 1.458.

**Honest limitations (quoted points).**
1. Trained on correct recitation with canonical labels, it "tends to transcribe the rule
   rather than the deviation": on the learner_b benchmark it reported full ikhfa in **81 of
   94** clips where annotators heard a plain noon.
2. Free-choice madd (munfasil, aared) is unreliable in the token stream because labels fix
   them at 4 counts while reciters legitimately choose 2, 4 or 6. "**Duration grading should
   be done from forced-alignment timing, not from the emitted symbol run-length.**"
3. "**Under-12 reciters score 2 to 3 times worse than adults on [the learner_b
   benchmark].**"
4. Streaming chunks match or beat full context in domain; on out-of-domain audio full
   context is about 2 PER points better.

**Usage terms.** Gated; usage terms are kept with the source files. The gate also requires
not presenting the output as an authoritative religious ruling, so it stays on hold.

**Flag (inference).** Phone PER (3.65%) is lower than unseen-professional PER (9.10%). The
training data includes real phone recordings, very likely from the same app pool as the
phone holdout clips, which have no speaker ids and contain duplicates (source 7). The phone
number may be optimistic; the unseen-reciter number is the more honest generalisation figure.

**Relevance.** The best public evidence on streaming size, front-end and phone export for a
model of our size class, and the source of the duration-grading advice. See Application (d).

---

## 9. An open live recitation tracker

**What it is.** A small Python web-server demo with websockets that highlights words as they
are recited and warns on skipped ayahs or the wrong page. Open source. 151 stars. No
evaluation data.

**Method.**
- Browser records WebM chunks; the server converts them to 16 kHz WAV, keeps a **cumulative
  8 s sliding buffer** and re-transcribes the whole buffer on every chunk. ASR is a cloud
  Whisper API by default, or a local Conformer-CTC trained with the training toolkit on
  voices_a Arabic (not Quran). Chunks below an RMS energy threshold (0.01) are dropped as
  silence.
- Text is normalised (diacritics, tatweel and punctuation removed). Isti'adha and basmala are
  stripped first if the opening words match them at average similarity 0.7.
- Candidate segments of 5-25 Quran words (stride 3) are scored as
  0.7 x (1 - edit distance/max length) - 0.3 x (length difference/max length); then a global
  sequence alignment maps spoken words to Quran words; a word is "correct" if its
  similarity is at least the word threshold (0.45 in the app config; the engine default is 0.7).
- **Tracking mode** searches 60 words ahead and 15 behind the anchor. After 3 chunks with
  confidence below 0.4 it switches to **search mode** over the whole displayed page.
- The anchor **only moves forward** (`max(previous, furthest correct word)`).
- Sequence checks: a jump of at least 12 words covering at least 1 full ayah is a skipped
  ayah; repeated low confidence on the page is "page mismatch"; a large move back
  (tolerance 3 words) is a "backwards anomaly". Alerts need confidence of at least 0.5.
- The README's "fuzzy matching vs temporal alignment" section argues that text-based fuzzy
  matching is lighter, more robust to ASR errors and easier to debug than acoustic DTW.

**Numbers.** README table: skip detection "80-90%", false positives "5-10%", latency under
1 s. No test set or script backs these.

**Limitations (from the code, inference).** Re-transcribing an overlapping buffer re-emits the
same words each chunk; the forward-only anchor cannot follow a learner who goes back to
repeat; text-level matching after removing diacritics cannot see vowel mistakes (its own doc
says it does not check harakat or tajweed); a general Arabic ASR will "correct" towards real
words.

**Relevance.** A clear, small reference design for the position state machine (tracking vs
search, skip rules). The scoring layer is not usable for checking. See Application (c).

---

## 10. A word-timing helper tool

**What it is.** A command-line tool that makes word-level timing files (verse, word
position, start and end in ms) for complete surah recordings by expert reciters, to drive
word highlighting. Developed on a desktop computer; 0 stars.

**Pipeline.**
1. Whisper large-v3 (or a Quran-tuned Whisper from the platform, the default)
   transcribes the whole surah with word timestamps, **prompted with up to 100 words of the
   expected Quran text** (plus isti'adha and basmala).
2. The transcript is aligned to the known text by dynamic programming (min score 0.55,
   lookahead 15 words); missing words are completed or interpolated.
3. Recovery passes repair Whisper's known failures: openings and huruf muqatta'at (re-
   transcribe a leading gap of 4 s or more without the prompt; snap the first word back to the
   last silence before it; a fixed bracket for muqatta'at); interior gaps (re-transcribe the
   gap with 600 ms padding); severe under-transcription (below 85% coverage, re-run without
   the prompt and keep whichever aligns more real words); madd word ends extended up to 1.8 s
   (3.5 s at verse end) and stopped at silences.
4. **wav2vec2 CTC forced alignment** (default: a public Arabic wav2vec2 XLSR-53 model,
   character level) places words that are still estimated, by aligning the known text to the
   gap between two real anchors. Wider gaps (over 15 s) are cut into 45 s chunks. A full-surah
   CTC fallback runs when half the words lack real times (clips up to 2 min). Separate passes
   handle identical adjacent ayat (94:5/94:6) and near-adjacent ones (109:3/109:5), which
   Whisper merges into one.
5. Quality gate: a surah is ok / needs_review / needs_fix; target match rate 0.995 (0.9955 in
   validation), red below 0.90; word length 20 ms-12 s. A "recitation anomaly" check reports
   (never fixes) repeated words or phrases: an extra token at least 0.80 similar to a
   neighbouring matched word.

**Numbers.** Thresholds only; no accuracy measurement is published.

**Usage terms.** Usage terms are kept with the source files. They reach anything built with
the tool or derived from it, including the timing files it generates, so it is not used.

**Limitations (inference).** Built for correct expert recitation: the text prompt and the
"force the known text in" fallbacks hide skips and mistakes, which is right for highlighting
and wrong for checking. CTC forced alignment places every word even when a word was not
said, so its scores need a floor (the tool defaults several floors to 0.0).

**Relevance.** A long list of tested failure cases for cutting long recitations, and evidence
that Whisper timestamps alone are not enough at openings, long madds and repeated ayat.

---

## 11. A vendor case study on real-time Arabic speech recognition

**What it is.** A short vendor page about another app for Quran recitation, plus a 2022
vendor developer blog post on speech applications.

**Architecture (as stated).** A Quranic Arabic ASR model fine-tuned with the vendor's
training toolkit, optimised and served with the vendor's inference stack on data-centre
GPUs. Server-side.

**Lessons (blog).** On-device speech frameworks for phones were tried first and rejected:
"designed more for commands and short sentences than precise recitations". A first custom
fine-tune lowered WER but missed accuracy and latency goals. Latency over **300 ms** "becomes
very annoying". The app marks the word being said in green and wrong or missed words in red,
counts mistakes and links to similar passages.

**Key number.** "4 percent WER", with no test set, audio type or date.

**Usage terms.** Not applicable.

**Limitations.** Marketing material: no architecture details, no evaluation method. A later
third-party measurement in `research_summary.md` (9.56 WER overall, 16.17 on phone audio) is
the better figure.

**Relevance.** A latency budget (300 ms) for live tracking, and a reminder that the leading
product is server-side, so an on-device checker is a real gap. Little else.

---

## Application to Qaf-ft

Priorities: **P1** = do before or during the phone fine-tune; **P2** = next stage (verifier,
cutting, tracking); **P3** = later (tajweed, export). Nearly everything here is
**(inference)** from the sources above; direct source facts are cited by number.

### (a) What the expert set's paper implies for the checker and verifier

1. **P1. Accept every legal reading before scoring a word.** Our 81 tokens carry madd length
   (ا2, ا4, ا5, ا6, ۥ4 and so on) taken from each training mushaf's settings. Ordinary
   reciters legitimately choose other lengths for munfasil (2/4/5), aared (2/4/6), leen and
   the sakt sites (source 1, A.4; source 8, limitation 2). In word/vowel mode, either build
   the expected sequence as the set of all legal variants and score the best one, or fold
   free-choice madd tokens into one "long" class before scoring. Some of the 51% of correct
   phone words flagged may be legal length choices; measure this on the labelled phone clips
   first.
2. **P1. Use their numbers as a yardstick, with care.** Their 0.21% "average PER" averages 11
   levels and is not comparable. The like-for-like figure is the phoneme level: 0.543% on
   three unseen expert mushafs vs our token error of 0.43% on two unseen reciters (different
   reciters and token sets, so only "same order"). The useful outside number is their
   **5.78% phoneme PER on the learner_b benchmark**. Expand our repeat tokens into their
   43-symbol script and report learner_b benchmark PER, so Qaf-ft has one directly comparable
   learner number.
3. **P2. Sifat heads: a cheap side experiment, not a redesign.** In the training data the
   sifat are fixed functions of the phonemes (derived from text, source 1), so they add no
   new label information. Possible gains are regularisation and a second, partly independent
   vote at test time. A word verifier could require the phoneme head and the matching sifat
   head (for example tafkheem for ص/س, itbaq, hams/jahr for ذ/ث) to agree before flagging a
   letter. Try it as 2-3 small extra CTC heads on the FastConformer encoder with low loss
   weights (their phoneme weight was 0.4 of the total), and keep it only if the false-flag
   rate at fixed recall drops on phone clips.
4. **P1. Expect the same blind spots.** A model trained on correct experts cannot learn to
   see ض without istitala, ر without tikrar, or a skipped ikhfa (source 1 §7; source 8: 81 of
   94 plain noons heard as ikhfa). Our extra-sound check against the free decode helps, but
   these errors need real mistake data. Keep them out of anything the app promises.
5. **P2. Training details worth testing on the phone fine-tune.** They found SpecAugment
   harmful and a constant lr best, on studio data only. Since our gap is the phone channel,
   test both choices again on phone clips rather than copying them. Their augmentations
   (background noise, time stretch, gain transitions) are a cheap baseline.
6. **P2. Verifier output.** Source 1 reports per-level error rates only, with no confidence
   or false-flag numbers, so it does not help set thresholds. Keep the fixed-false-flag
   thresholds and three-state output already planned.

### (b) Test sets for honest evaluation, and their limits

| Priority | Set | What it measures | Limit |
|---|---|---|---|
| P1 | Own labelled phone clips (in progress) | The target: false flags per correct word on phones; real word mistakes | Private voices: keep local |
| P1 | learner_b benchmark | Learners, 120 women, mixed ages, real mistakes in QPS; **narrowband 11 kHz messaging-app audio** (source 3), so a phone-like channel; 53 clips are fully correct, which allows a false-flag count | Evaluation only (speaker leakage with learner_b's parent set) |
| P1 | The aligned expert set, unseen-reciter test split | Studio generalisation to 2 unseen expert reciters | Usage terms kept with the source files |
| P1 | Own phone holdout from the phone set | Phone ASR accuracy, and checker false flags on clips that pass the agreement filter | **Remove duplicates by audio fingerprint, not file name** (source 7), and drop bench_a's 200 phone holdout clip ids anyway |
| P2 | mdd_c | Word-located real mistakes, 384 participants, 4 short ayahs | Usage terms kept with the source files |
| P2 | learner_a (labelled part) | Clip-level correct vs mistake for non-Arab learners; gender field allows a split | No stated terms: report only, ask the authors first |
| P3 | mdd_e (v2) | Letter and vowel substitution recall vs a public baseline (best F1 29.9%, FR 12.65%) | Not for training; needs a mapping to the 68-symbol MSA set; acted, no tajweed, mostly women |
| Hold | bench_a phone holdout clips and model_b comparisons | Phone ASR vs a public board | Its terms bind a model that is merely evaluated on it |

Rules that follow from the sources:
- Report studio and phone separately, always (source 7). Never let a studio number stand for
  the app.
- When comparing with source 1 or 8 on the learner_b benchmark, use the same metric (phoneme
  PER after expanding repeats; madd RMSE in counts) and say that it is one ayah with one
  annotator.
- Per-ayah clips can hold repeats even when "reviewed" (source 7 v1.1). Repeats must be a
  normal, not-flagged event in our scoring.

### (c) Cutting long recitations and live tracking

**Cutting a long recitation into ayahs (P2).** Combine three pieces:
1. **Pause candidates.** Off-the-shelf voice activity detection did poorly on recitation
   (source 1); a trained pause classifier reached F1 0.995. Start with energy/VAD pauses and
   our own CTC blank runs; train a pause head only if cut errors show up.
2. **Find where each piece is in the text with our own free decode**, matched by a moving
   pointer (source 1's Tasmeea: 30-word window, accept above 0.85, widen the window after a
   miss). Match on the sound tokens, not on diacritic-stripped text (sources 9, 10), so that
   vowel mistakes stay visible. Embedding similarity (source 6) is fine for "which ayah" but
   must never judge a word.
3. **Place boundaries by CTC forced alignment of the known text with our own model**, inside
   the window found in step 2. Forced alignment never drops a word (source 10), so it will
   also "place" a word that was skipped: gate every word on its alignment score and send
   low-scoring words to the checker as possibly missing, never fill them in.
Known hard cases to test, all listed by source 10: isti'adha and basmala before the first
ayah (try "with/without prefix" hypotheses and keep the better score); huruf muqatta'at; long
madd at ayah ends; identical ayat (94:5/6, 109:3/5, refrains); repeated ayahs or phrases.
**Never prompt or bias the recogniser with the expected text** (source 10 does, which is right
for highlighting and wrong for checking; agrees with `research_summary.md`).

**Live tracking (P2).** Use source 9's position logic, rebuilt on our sound-token stream:
- States: *tracking* (search about 60 words ahead and 15 behind), *lost* (after 3 low-score
  windows, search the page or surah), and a **repeat** state that allows moving backwards.
  Source 9's forward-only anchor would treat a learner going back to fix a word as an error.
- Skip rule as a starting point: at least one full ayah or 12 or more words jumped (source 9),
  tuned on labelled clips.
- Use a cache-aware streaming encoder instead of re-transcribing an 8 s buffer (source 9),
  which repeats work and re-emits words. Source 8 shows 0.48 s chunks work for a phoneme CTC
  model; full context was about 2 PER points better on out-of-domain audio, so give the final
  per-word verdict from a full-context pass at the end of each ayah.
- **Live display shows position only; mistake flags appear after the ayah ends** and after
  the verifier has run. That keeps live, low-context guesses away from the false-flag budget.
- Latency target: under 300 ms from speech to highlight (source 11).
- Source 9's claimed 80-90% skip detection and 5-10% false alerts have no test behind them;
  measure our own on labelled long recordings.

### (d) Duration grading for the later tajweed mode (P3)

1. **Grade length from forced-alignment timing, not from the count of repeated tokens**
   (source 8, explicit advice; source 1 Fig. 10 shows uneven madd runs). Align the expected
   tokens, measure each madd's duration, and divide by the reciter's own count length in the
   same ayah (estimated from 2-count natural madds or short vowels).
2. **Grade only clear errors.** Best published madd RMSE on the learner_b benchmark is
   0.40-0.55 counts for natural madd and 0.69-1.46 for munfasil and aared (sources 1, 8). That
   is close to one count, which is the difference between legal choices. Flag only natural
   madd at roughly 1 count or less, lazim clearly under 4, and munfasil/aared below 2; accept
   2/4/6 where the rule allows.
3. **Check consistency within one recitation** (the same madd type should keep the same
   length), which needs no absolute threshold.
4. **Leave out ikhfa in the first tajweed release.** Two independent models fail on it (F1
   0.45 with the "partial" class never found; 81 of 94 plain noons called ikhfa). Qalqalah
   (F1 0.95) and noon mushaddadah (F1 0.87, partial recall 0.66) are the best candidates, as
   advisory hints.
5. Keep Qaf-ft's three-state output (correct / mistake / unsure) for every tajweed site.

### (e) Children's and women's voices

What the sources show:
- The main training set is male experts only (source 1). Its learner benchmark is 75% women,
  but no per-gender result is given (the "84.7% on women" claim is a misreading, source 1 Flag).
- Under-12 reciters score 2-3x worse than adults on the learner_b benchmark (source 8), even
  for a model trained on thousands of hours including phone audio.
- mdd_e v1 has 14 women of 18 speakers (source 2); learner_a has a gender field (source 5);
  the learner_b benchmark has gender and age per clip (`datasets.md` §11).

Recommendations:
1. **P1.** Report every checker metric split by gender and by age band (under 12, 12-17,
   adult) on the learner_b benchmark and on our own phone clips. Track the false-flag rate per
   group against the 0.1% per-word target, not only the average.
2. **P1.** In the phone fine-tune, check how much of the phone set is women's and children's
   speech (sample and label a few hundred clips by ear), since no card says.
3. **P2.** Add pitch shift and vocal-tract-length perturbation to augmentation to cover higher
   voices; test the gain on the women/children slice only.
4. **P2.** For our own collection, aim for clips from women and children with consent, and
   hold them out by speaker for testing first.
5. Until the child false-flag rate is measured, do not claim the app works for children.

### Top priorities in one list

1. Accept all legal readings (free-choice madd and sakt) before scoring any word; measure how
   many current phone false flags this removes. (a1)
2. Build the phone test from the phone set with fingerprint deduplication, and add the
   learner_b benchmark as a phone-like, learner, mostly-women test with a false-flag count on
   its 53 correct clips. (b)
3. Report learner_b benchmark phoneme PER in the 43-symbol script, next to the expert set's
   paper's 5.78%, split by gender and age. (a2, e1)
4. Cut long recitations by free decode, moving-pointer match on sound tokens, and score-gated
   forced alignment; never prompt with the expected text. (c)
5. Live mode: position-only display with a repeat-tolerant state machine; per-word verdicts
   only after each ayah, from a full-context pass. (c)

---

## Corrections and additions to existing docs (for whoever updates them)

| Doc | Point | Source |
|---|---|---|
| research_summary.md | "2026 mdd_e task winner 1.75% false reject" is not in the mdd_e paper; that paper's baselines reject 12.65-21% of correct phonemes | 2 |
| research_summary.md | "Ship as advisory" for ikhfa is too generous; two models fail on it | 1, 8 |
| research_summary.md | The vendor pages are now reachable; they hold no architecture detail beyond the vendor's training toolkit and inference stack, server-side, 300 ms latency goal | 11 |
| how_others_build.md §4 | learner_b 95-99% results come from a random clip split with about ten takes per speaker (speaker leakage) | 3 |
| datasets.md §11 | The learner_b benchmark audio derives from 11 kHz messaging-app recordings, so it is narrowband | 3 |
| datasets.md §2 / §10 | The phone set contains the same recording under different ids; deduplicate by audio, not file name | 7 (via a model card) |
| datasets.md §13 | Add the 0.89 algorithm-vs-expert agreement; per-annotator scores are in `annotation_metadata` | 5 |
| datasets.md §3 | The studio_d paper is unclear which ASR built the released data | 6 |
| Expert set's paper citations | "84.7% accuracy on female reciters" is the all-clip tajweed accuracy, not a women-only number | 1 |
