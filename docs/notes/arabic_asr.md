# Arabic ASR study notes: benchmarks, base model, open corpora (for Qaf-ft)

Sources read in full for these notes (fetched 2026-11-21):

- The paper of an open Arabic ASR leaderboard, its code repository (README, `eval.py`,
  the Conformer model wrapper) and the live leaderboard table on the platform (`app.py`,
  last update listed 2026-10-19).
- The broadcast_b challenge paper, and the gated usage page of broadcast_b on the platform.
- The base model's paper, the model cards of the base and of its sibling MSA-only model,
  and the base authors' public data-processing configs for Arabic (the data-processing
  toolkit, `dataset_configs/arabic/`).
- The broadcast_a paper and the broadcast_a card.
- voices_b: its data repository page (description, file layout, usage terms) and its card
  on the platform. The voices_b paper itself (a 2022 speech conference, paywalled) was
  **not read**.
- voices_a Arabic: the official release statistics (v17.0 and v27.0 JSON), the v24.0
  Arabic datasheet, and the notice that releases moved to a new data-hosting service in
  October 2025. That service's dataset pages did not render and its account terms were
  **not read**.

Conventions

- Everything under "Application to Qaf-ft" is this note-taker's application. Points
  that go beyond what a source states are marked **(inference)**.
- Sources are cited by project label. People are named by role only.
- "WER" without qualification means words scored after removing punctuation and
  diacritics, as each source defines it. Numbers are copied from the sources' tables.

---

## 1. An open Arabic ASR leaderboard

### What it is

A continuing benchmark of open Arabic ASR models, evaluated zero-shot on test sets that
each contain several dialects. The stated aim is a common evaluation framework for
multi-dialect Arabic ASR, modelled on an English open ASR leaderboard. Hosted on the
platform; evaluation code in a public code repository.

### Method and data

- Only multi-dialect test sets are used. The paper argues it is "methodologically
  unsound" to compare dialects across different datasets, because recording conditions
  differ between datasets.
- Test sets in the paper: a Saudi multi-dialect test set (10.7 h, 10 dialects), voices_a
  18.0 test (12.6 h, 25 dialects), voices_b test-clean (10.5 h, 7 dialects), voices_b
  test-noisy (8.9 h, 14 dialects), broadcast_b test (9.6 h). broadcast_a was left out
  because of "downloading issues". A further multi-country dialect test set (7.7 h,
  8 dialects) was added in January 2025. The leaderboard's metrics tab gives voices_b
  noisy-test as 14.9 h while the paper and README give 8.9 h (unresolved).
- Models in the paper: Whisper small/medium/large/large-v2/large-v3/large-v3-turbo; a
  commercial Conformer-CTC-large Arabic model (about 120M parameters, trained on over
  3,600 h), greedy and with its released 4-gram LM; a large multilingual speech
  translation model (best of three language settings); a massively multilingual
  wav2vec 2.0 model; and four community fine-tunes (wav2vec 2.0 XLS-R, a second
  wav2vec 2.0, w2v-BERT 2.0, HuBERT-large).
- Normalisation (paper): remove punctuation and diacritics, normalise hamzas and maddas,
  convert Eastern Arabic numerals to Western. Latin characters are kept, so
  code-switching errors count.
- Normalisation (repository `eval.py`, which is what produces the numbers) does more
  than the paper says: deletes U+064B to U+0652 only (tanween, harakat, shadda, sukun);
  maps آ أ إ to ا, ؤ to و, ئ to ي; **deletes standalone ء**; maps پ to ب and ڤ to ف;
  attaches a free-standing و to the next word. It does **not** touch ة, ى, the
  superscript alif (U+0670), alif wasla (U+0671) or Quranic annotation signs.
- Ranking is by the unweighted mean WER over test sets.

### Key numbers

Paper, Table 1 (five test sets): the commercial Conformer-CTC + LM ranks first, average
WER 25.71%, CER 10.02%; greedy 27.46%; Whisper-large-v3 29.87%; the multilingual
translation model 32.55%; smallest fine-tunes 54 to 60%. The paper notes that rank
follows labelled-data size: 3,600 h for the Conformer, over 700 h of Arabic for Whisper,
under 100 h for the fine-tunes.

Robustness on the Saudi set (Table 2, WER clean / noisy / music): Conformer + LM 37.83 /
49.21 / 47.30; Whisper-large-v3 51.45 / 58.47 / 58.71. Every model degrades in noise and
music.

Dialects on the Saudi set (Table 3): every model is best on MSA. Conformer + LM: MSA
19.23, Egyptian 40.97, Hijazi 36.96, Khaliji 48.23, Najdi 36.34.

Speakers on the Saudi set (Table 4): elderly speakers generally have higher WER than
adults, and women "slightly higher" than men. Conformer + LM: adult 37.85, elderly
44.31, female 40.19, male 37.91. Whisper-large-v3: female 54.32, male 49.60. The
multilingual translation model is the exception (female 48.84, male 55.53).

Efficiency (Table 5, one data-centre GPU, batch 1, mean clip 6.25 s): Conformer-CTC
greedy RTF 0.047, 0.48 GB to load; Whisper-large-v3 RTF 0.43, 2.88 GB.

Live table (six test sets, 39 models): top average WER is 22.82 (an industry ASR model),
and the commercial Conformer + LM is now 32.91 with the multi-country set included. The
base and its MSA-only sibling are **not listed**. voices_a test WER for the best models
is now 5 to 9%, while the multi-country set stays at 45% or more for every model.

### Licence

Usage terms are kept with the source files. The test sets keep their own terms.

### Limitations (stated and observed)

- The paper acknowledges in-domain leakage: voices_a splits are reshuffled between
  releases, and some models were trained on voices_a. broadcast_b is over 70% MSA.
- The README asks submitters to avoid training on the same datasets as the tests; the
  base and its sibling trained on voices_b and voices_a, so they could not be scored
  fairly here **(inference)**.
- No read Classical Arabic or Quran test set; no phone-microphone test set; no child
  speakers.
- Diacritics and hamza are removed before scoring, so vowel errors are invisible.

### Relevance

Low for model choice, high for evaluation practice: it shows how much a normalisation
choice hides, and gives the only published age and gender breakdown for these models.

---

## 2. broadcast_b

### What it is

An Arabic multi-genre broadcast challenge of 2016: about 1,200 h of Arabic news-channel
TV (19 programme series, March 2005 to December 2015) with lightly supervised
transcripts, 110M words of the channel's web text for language modelling, and grapheme
and phoneme lexicons. Two tasks: transcription and word alignment.

### Method and data

- Programmes were chosen because they had manual transcripts on the broadcaster's
  website. These are not verbatim: rephrasing, removed repetitions, summaries in overlap.
  WER between the published text and a verbatim version is "about 5%" on dev.
- Genre: conversation 63%, interview 19%, report 18%. Over 70% of speech is MSA; the rest
  Egyptian, Gulf, Levantine and North African. Dubbed English and French speech is
  included and not marked.
- Light alignment: a grapheme LSTM ASR system with a trigram LM decoded every programme;
  its output was matched to the published text (local sequence alignment, exact and
  approximate matches). Segment "anchor rate" (matched words / transcript words): 48%
  exact, 15% approximate, 37% no match. Unmatched words got linearly interpolated times.
  Each segment carries average word duration (AWD), phoneme and word matching error rates
  (PMER, WMER) so users can filter. About 550,000 segments, 1,200 h; about 300 h of short
  highlight clips were removed.
- Dev and test: 10 h each, from the last month of 2015, verbatim transcribed, manually
  segmented (3 to 10 s), with overlap marked.
- Four WER scorings: original text (with any punctuation and diacritics); after removing
  punctuation and diacritics; with a Global Mapping File for numbers and words without a
  standard spelling (official); and after normalising alef, yaa and taa marbuta.

### Key numbers

Baseline (GMM then DNN, 250 h): 34% WER on non-overlap dev. Best submission (the
organisers' own team, three acoustic models combined, n-gram plus RNN LM, speed and
volume perturbation): 14.7% official WER excluding overlap, 17.3% including overlap.
Scoring with diacritics and punctuation (WER1) adds about 6 points to every system (best
team 21.1 vs 15.0). A later version of the test set (broadcast_a paper, Table 2): 111
speakers, 87 male (78%), 13 female (11%), 78% MSA.

### Licence

Usage terms are kept with the source files. Access is gated and approved after a request
form.

### Limitations

Broadcast MSA, mostly adult male speakers, studio and outside-broadcast audio, no
diacritics, non-verbatim training transcripts.

### Relevance

None as training data (usage terms and domain). Two lessons carry over: published text
is not what was said, and per-segment match scores let you filter weak labels; and WER
moves by several points depending on which characters are normalised.

---

## 3. The base model's paper

### What it is

The paper behind the project's base model. It releases two FastConformer
Transducer-CTC models: an MSA model (punctuation, no diacritics) and the first public
unified MSA plus Classical Arabic model (punctuation and diacritics). The second is the
base (the pretrained 115M FastConformer Arabic model that Qaf-ft fine-tunes). Recipes
were published in the base authors' data-processing toolkit.

### Training data

Table I (raw train / processed train hours): voices_b 646 + 855 h (clean + noisy) to
433 + 255 h; voices_a 17.0 90 to 66 h; voices_c 6 to 5 h; studio_b 829 to 392 h.
studio_b was used only for the CA model. The model card gives about 1,100 h total for
the base (voices_b 690, voices_a 65, voices_c 5, studio_b 390) and about 760 h for the
MSA-only model. Blind evaluation (never trained on): broadcast_b test and a media speech
test set.

Usage terms are kept with the source files. The paper and the studio_b card disagree on
the terms of the studio_b recordings.

### Text processing and tokenizer

- Alphabet: 28 letters plus six hamza forms (آ ئ ؤ ء إ أ), ة and ى: 36 characters. Six
  diacritic classes (fatha, kasra, damma, sukun, tanween, shadda) bring it to 44 symbols.
  Punctuation: ".", "؟", "،".
- The public studio_b config's allowed alphabet is exactly those 36 letters and U+064B
  to U+0652. Samples with any other symbol are dropped. So the base never emitted
  superscript alif, alif wasla, small high marks or any tajweed sign.
- Steps: drop out-of-alphabet samples; drop clips under 0.1 s or over 20 s; numerals,
  text normalisation, ligature decomposition, NFKC (also fixes diacritic order),
  remove rare punctuation and tatweel; then filter by word rate, character rate, and WER
  and CER from a pretrained model.
- Overlap removal: studio_b test texts also found in train (other reciters reading the
  same ayah) were removed, shrinking test from about 100 h to 23 h. Results are shown
  "before / after".
- Tokenizer: SentencePiece unigram, 1,024 tokens, built from training transcripts.

### Model and training

- FastConformer hybrid RNN-T plus CTC, about 115M parameters, 8x convolutional
  downsampling, non-streaming, 16 kHz mono. CTC loss weight 0.3.
- Dropout 0.1; SpecAugment with 4 frequency masks and 20 time masks.
- AdamW, cosine annealing, peak learning rate 5e-3, 2,000 warm-up steps, 200 epochs;
  the five best checkpoints were averaged.
- Initialisation (Table III, WER voices_b / voices_a / voices_c): from scratch 12.74 /
  17.53 / 12.38; English 12.23 / 10.67 / 8.37; Spanish 11.63 / 10.21 / 8.18;
  multilingual 11.66 / 10.53 / 8.12. The Spanish FastConformer was chosen "for faster
  convergence".
- Noisy-data mining (Table IV): a model trained on clean data transcribed voices_b noisy;
  clips with WER above 60 or CER above 30 were dropped, keeping 255 h. voices_b WER fell
  from 9.95 (all 395 h noisy) to 8.50 (255 h filtered).

### Results (Table V)

Three metrics: WER (no punctuation or diacritics), WER_PC (punctuation kept),
WER_PC,D (punctuation and diacritics kept).

| Model | voices_b | voices_a | voices_c | studio_b (before / after) | broadcast_b |
|---|---|---|---|---|---|
| WER, no PC, no diacritics model | 8.50 | 8.20 | 5.01 | 6.28 / 8.47 | 23.4 |
| WER, PC + diacritics model without studio_b | 8.59 | 8.66 | 5.07 | 6.14 / 7.75 | 23.97 |
| WER, CA model (= base) | 8.64 | 8.52 | 5.39 | 1.34 / 5.73 | 24.70 |
| WER_PC,D, model without studio_b | 16.51 | 25.41 | 13.09 | 91.60 / 90.49 | 23.97 |
| WER_PC,D, CA model (= base) | 16.67 | 25.60 | 12.94 | 1.55 / 6.65 | 25.58 |
| WER_PC,D, Whisper-large-v3 | 13.82 | 16.74 | 8.28 | 98.78 / 99.40 | 21.90 |

Findings stated by the paper:

- An MSA-only model already transcribes Quran words well (about 6% undiacritised WER on
  studio_b) but scores over 90% WER_PC,D, because it "tends to produce partially
  diacritized text". Adding studio_b drops this to 1.55 / 6.65.
- Including diacritics in training has "a minor impact" on word recognition.
- MSA diacritic WER is high because MSA diacritics "lack standardized rules" and the
  datasets are inconsistent; some voices_a test sentences are diacritised, some are not.
- On out-of-domain broadcast speech (broadcast_b) these models are worse than Whisper and
  the multilingual translation model (about 21 to 25% vs 18.5 to 21.5%).

Model card (greedy, no LM): base Transducer decoder voices_b 16.67, voices_a 25.60,
voices_c 12.94, studio_b 6.55; **CTC decoder** 17.56, 27.42, 14.21, **7.61**. The
MSA-only card: Transducer 11.46 / 10.20 / 8.18; CTC 12.11 / 11.38 / 9.23. The CTC head
is about one point worse.

Internal inconsistencies: the paper's conclusion quotes 11.37 / 9.76 / 7.73 for the MSA
model, which match no Table V row; the card's 6.55 for studio_b differs from the
paper's 6.65; the base card's metadata names the model "pc_v2.0".

### Released checkpoints

| Checkpoint | Output | Training data | File | Created |
|---|---|---|---|---|
| MSA-only sibling | MSA, punctuation, no diacritics | about 760 h, no Quran | 424,164,993 bytes checkpoint | 2024-11-01 |
| base | MSA + CA, punctuation and diacritics | about 1,100 h incl. studio_b 390 h | 459,233,280 bytes checkpoint | 2024-12-25 |

Both about 115M parameters, saved in the training toolkit's format. Usage terms are kept
with the source files. The card's restrictions: non-streaming, "noise-sensitive", "can
have poor performance in dialectal Arabic speech"; no bias mitigation measures taken.

### Limitations

- Quran data is professional studio recitation only (studio_b); clips over 20 s were
  dropped, so long ayahs are under-represented **(inference from the 20 s filter and the
  829 to 392 h reduction)**.
- No phone audio, no learner speech, no speaker breakdown by gender or age.
- The studio_b train/test split is by recording, not by reciter or text; the paper's
  overlap fix removes shared texts from test, but reciters remain shared.

### Relevance

Direct: it defines what the encoder has heard, which text symbols it knows, the
training recipe the fine-tune departs from, and the leakage risk for any studio_b-
derived test set.

---

## 4. Open corpora: voices_a Arabic, voices_b, broadcast_a

### 4a. voices_a Arabic

**What it is.** A general Arabic read-speech corpus: volunteers read prompted sentences
into a browser or phone; other volunteers vote on whether audio matches text.

**Size.** v17.0 (used by the base model): 134,892 clips, 155.81 h recorded, 90.27 h
validated, 1,561 speakers; official splits train 28,369, dev 10,470, test 10,480 clips.
v27.0 (2026-09-11): 136,832 clips, 158.27 h, 93.34 h validated, 1,679 speakers, average
clip 4.16 s. The splits use 62.5% of validated clips (v24 datasheet), so about 50,000
validated clips sit outside train/dev/test.

**Content and labels.** Text is the prompted sentence, not a transcript of what was
said. 59,439 validated sentences. The datasheet sample mixes plain MSA and fully
vowelled classical prose (for example a hadith chain of narrators). No Arabic
spontaneous-speech release exists (none in the v5.0 spontaneous-speech statistics).
The leaderboard paper describes the 18.0 test as covering 25 dialects (accents
are self-reported).

**Speakers.** Self-declared, per clip, v27: gender unknown 54.8%, male 28.1%, female
17.1%; age unknown 54.5%, twenties 29.4%, thirties 10.9%, teens 3.8%, forties 1.3%,
older under 0.2%. No band below "teens"; no children label.

**Licence.** Usage terms are kept with the source files. Since October 2025 releases are
only on a new data-hosting service behind an account and its terms of service, which
were not read.

**Limitations.** Read speech of short sentences, mostly MSA; many voices are recorded on
consumer devices (collection is browser and app based) but device metadata is not
provided **(inference: device type is not a field)**. Repeated sentences across speakers;
split reshuffles between versions cause leakage. Mixed diacritisation.

**Relevance.** The only openly usable source here with women, teens and ordinary
microphones in quantity. The base already trained on the v17 train split (66 h after
filtering).

### 4b. voices_b

**What it is.** About 1,000 h of 16 kHz speech crawled from over 700 online video
channels; multi-regional, multi-genre, multi-dialect. Released with a 3-gram LM, an
end-to-end baseline model and 12M unique words from social media. One 169.41 GB archive.

**Data and labels.** Training labels are the channels' Arabic subtitles (`.vtt`),
described as "raw training data". Dev and test (about 10 h) were manually aligned and
transcribed, with diacritics and punctuation removed and letters normalised (alef forms
to ا, ة to ه, per the base model's paper). Subsets: clean and noisy, each with train,
dev, test; the base authors counted 646 h clean and 855 h noisy raw training audio, more
than the headline 1,000 h. Metadata per video: category, channel, country, dialect,
gender; a channel quality file.

**Licence.** Usage terms are kept with the source files. Rights in the underlying videos
are not discussed.

**Limitations.** Subtitle labels are not verbatim (the base authors kept only 255 of
395 h noisy after WER filtering). Gender and dialect are video-level metadata. Content
includes religious speech (the card's example is a supplication); whether Quran
recitation channels are included is not stated **(inference: possible)**. Paper not
read, so per-subset hours, gender split and channel selection criteria are unknown here.

**Relevance.** A large, openly usable source of everyday Arabic voices and noisy
real-world audio. The base already saw 690 h of it.

### 4c. broadcast_a

**What it is.** 2,041 h of Arabic news-channel broadcast (2004 to 2015, 3,545 episodes,
19 programmes), lightly supervised like broadcast_b but with linguistically motivated
segmentation, punctuation, speaker names and gender, code-switching marks, and 130M words
of LM text. The broadcast_b dev/test sets are excluded so they remain valid tests.

**Key numbers.** 1.6M segments, mean 4 s (84% between 2 and 6 s), 14.3M words, 11,092
unique speakers. Gender is known for 2,000 speakers covering 82% of segments: 1.2M
segments male (69%), 99K female (6%). End-to-end Transformer ASR: test WER 14.7 trained
on broadcast_a vs 14.3 on broadcast_b; broadcast_a's model had about half its errors as
deletions, traced to segments longer than its training distribution (effective duration
7 s vs 9 s in test).

**Licence.** Usage terms are kept with the source files. Download is through a request
form whose terms were not read. The card has no data files.

**Limitations.** Broadcast MSA, heavily male, non-verbatim transcripts.

**Relevance.** Low. Its useful lesson is the duration mismatch result.

---

## Application to Qaf-ft

### (a) What the base-model paper implies for fine-tuning

1. **Only the encoder transfers.** The base predicts 1,024 SentencePiece pieces over
   36 letters and 8 diacritics; Qaf-ft predicts 81 sound tokens. The RNN-T decoder,
   joint and CTC head carry no usable output layer, which `train/full.py` already
   reflects by training a new head with its own learning rate **(inference)**.
2. **The encoder knows vowels from professional recitation, not from learners.** The
   paper shows MSA data teaches letters and studio_b teaches Quranic vowelling (WER_PC,D
   on studio_b 90.5 without it, 6.65 with it). So the encoder's vowel features were
   tuned on studio reciters and have never been tested on phone or learner audio
   **(inference)**. Do not freeze the upper encoder blocks wholesale; if any freezing is
   tried, freeze the subsampling and lowest blocks first and compare on phone audio
   **(inference: the paper reports no freezing experiments)**.
3. **Tajweed has no prior.** No madd, ghunna, qalqala or small Quranic marks were in the
   base alphabet; samples containing them were dropped. Any length or nasalisation
   distinction in the 81 tokens is learned from scratch in the fine-tune
   **(inference)**. This fits the priority order: words and vowels ride on the base;
   tajweed needs project data.
4. **Learning rate.** The base was trained from a Spanish initialisation at peak 5e-3
   with 2,000 warm-up steps. Fine-tuning rates two orders of magnitude lower (the
   current 3e-5 encoder, 1e-4 head) are consistent with keeping what it learned
   **(inference)**.
5. **Keep the base's augmentation in place.** It was trained with SpecAugment (4
   frequency, 20 time masks) and dropout 0.1. Turning SpecAugment off during fine-tuning
   changes the input distribution the encoder expects; keep it on, at similar or milder
   strength **(inference)**.
6. **Average checkpoints.** The base averaged its five best checkpoints. Averaging the
   last few fine-tune checkpoints (or keeping the EMA already in `train/full.py`) is a
   cheap gain worth confirming on held-out reciters **(inference)**.
7. **Segment length.** The base saw no clip over 20 s. Expert-set waqf segments longer
   than 20 s and long user recordings at inference are out of its range; cap training
   segments at about 20 s, or test accuracy by length bucket before relying on long
   clips (the broadcast_a deletion result shows what a duration mismatch does)
   **(inference)**.
8. **CTC head quality.** The base's own CTC decoder is about one WER point worse than its
   Transducer on every set (studio_b 7.61 vs 6.55). A CTC-only fine-tune with no LM
   starts from the weaker of the two decoders' training signals; expect to need the
   fine-tune data to close that gap **(inference)**.
9. **Leakage.**
   - The base trained on the studio_b train split (about 390 h, by recording, not by
     reciter), so it has heard every studio_b reciter and very likely every ayah text
     **(inference: the train split covers the whole Quran across reciters, and the paper
     found test texts duplicated in train)**. This confirms datasets.md: bench_a's
     studio_b held-out config and any studio_b-derived test is not held out for this
     project.
   - The project's speaker-disjoint test reciters are expert-set moshaf 5.0 and 13.0. The
     reciter of moshaf 13.0 has a folder on studio_b's website (same name as in the
     expert set's `moshaf_metadata`). Whether that reciter is inside the studio_b copy on
     the platform could not be confirmed: the dataset viewer statistics are partial (they
     list 15 reciters, up to the letter "h") and row lookups failed. **Check the
     `reciter` column of all three studio_b splits before calling 13.0 unseen.** No
     studio_b folder matched the name of moshaf 5.0's reciter (source: another
     recitation website); spelling variants not checked.
   - Other expert-set reciters are in studio_b: at least those of moshaf 0.x, 2.x, 29.0
     and 30.0 (partial viewer statistics) and 3.0 (last train row). Good results on them
     partly measure the base, not the fine-tune **(inference)**.
   - voices_b may contain recitation channels; not checked.
10. **Which checkpoint to start from.** The MSA-only model never saw Quran; the base did.
    Since Qaf-ft discards the output layer, starting from the MSA-only model would remove
    the studio_b leakage at the cost of the Quran-trained encoder. Worth one comparison
    run on the unseen-reciter and phone sets if leakage becomes a question for reporting
    **(inference)**.

### (b) Which corpora could be added, and how

None of the four corpora contains Quran recitation labels, so none can supply the 81
sound-token targets directly. Their use is to keep the encoder robust to ordinary voices
and devices. Ranked:

| Rank | Source | Notes | Proposed use |
|---|---|---|---|
| 1 | voices_a Arabic (v17 to v27) | Hosting-service account terms to read first | Auxiliary and rehearsal data for voice variety |
| 2 | voices_b clean + noisy | Usage terms kept with the source files | Rehearsal, noise robustness, noise-only source |
| 3 | voices_c Arabic (seen by base) | Usage terms kept with the source files | Small rehearsal set |
| - | broadcast_a | Request-form terms unread | Not worth it: broadcast, 6% female |
| - | broadcast_b | Research-only access | Not used |

How to use them, most promising first:

1. **Auxiliary character CTC head (multi-task).** Keep a second, small CTC head that
   predicts Arabic letters (and diacritics where present) on voices_a and voices_b
   batches, while the main head learns sound tokens on recitation batches. The encoder
   then keeps seeing women, teens and consumer microphones without forcing wrong sound
   labels onto MSA speech **(inference)**. Start with a low share (for example 10 to 20%
   of batches) and check it does not hurt the unseen-reciter score.
2. **Rehearsal against forgetting.** The base's robustness to ordinary voices came from
   voices_b and voices_a. Fine-tuning on male professional reciters only can erode it.
   Mixing a slice of the base's own training data through the auxiliary head is the
   standard counter **(inference)**. Measure with a voices_a/voices_b dev WER from the
   auxiliary head before and after.
3. **Noise and channel augmentation only.** Use voices_b noisy segments (music, rooms,
   street) as additive background for recitation clips, alongside the existing
   degradation (`--degrade`). This is still training use, under the source's usage terms
   **(inference)**.
4. **Voice augmentation for women and children.** No source here has children, and women
   are 17% of declared voices_a clips, 6% of broadcast_a, 11% of broadcast_b test, and
   absent from the expert set. Pitch and formant (vocal tract length) perturbation of
   recitation clips is the only way to cover children with what is available
   **(inference)**. Validate on the three women's recordings in learner_b and on the
   phone set where gender is known.
5. **Self-training on phone audio.** The base authors' voices_b mining (drop clips with
   WER over 60 or CER over 30 against the model's own output) is a template for the phone
   set: decode with the current model, keep clips whose output matches the intended ayah
   closely, train on those. Use a tighter threshold than theirs because Qaf-ft cares
   about vowels **(inference)**. Remove the 200 phone holdout clips first.

Usage terms for each source are kept with the source files. Open point: the paper and
the studio_b card disagree on the terms of the base's studio_b training data, so the
base inherits that question and it belongs in "Open questions" **(inference)**.

### (c) Evaluation lessons

1. **Normalisation decides the score.** The leaderboard removes harakat, shadda, sukun,
   tanween, hamza seats and standalone hamza before scoring, which is exactly the class
   of mistakes Qaf-ft must catch. Its WER is useful only as a secondary "right words"
   number, never as the main metric **(inference)**.
2. **Report a ladder of metrics, as both broadcast_b and the base model's paper do.** For
   Qaf-ft: (i) sound-token error rate on the 81 tokens (primary); (ii) letters plus
   vowels error rate after mapping tokens back to text (vowel mistakes); (iii)
   undiacritised skeleton WER (word mistakes); (iv) tajweed-only errors last. The gap
   between (ii) and (iii) shows vowel errors separately, as WER vs WER_PC,D did for the
   Whisper models (8 to 17% vs 98 to 99% on studio_b) **(inference)**.
3. **Write the normaliser down and test it.** The leaderboard paper and its code differ
   (the code also deletes ء and joins و). Its diacritic range misses U+0670 and Quranic
   signs, so Uthmani references would score those as errors. Qaf-ft's scorer should list
   every mapping and have unit tests on Uthmani and simple-script samples **(inference)**.
4. **Speaker-disjoint and text-aware splits.** studio_b shares texts and reciters across
   splits; voices_a reshuffles splits between versions. Hold out by reciter across
   all corpora (as datasets.md says) and report scores on ayahs the fine-tune never saw
   as well as on seen ayahs **(inference)**.
5. **Break results down by speaker group and condition.** The leaderboard found higher WER
   for women and older speakers and large drops in noise and music. Report Qaf-ft scores
   by gender (where known), by device (studio vs phone) and by noise, not just one
   average **(inference)**.
6. **Dialect matters to learners too.** Every model is far worse on Egyptian and Gulf
   speech than on MSA. Learners carry dialect habits into recitation (vowel colouring,
   letter substitutions), so a learner set with known regional background would show
   whether the model mistakes accent for error **(inference)**.
7. **Do not compare across test sets.** The leaderboard's rule applies directly: a better
   score on the phone set than on learner_b says nothing about phones vs studio unless the
   same speakers and texts appear in both **(inference)**.
8. **Check for in-domain contamination before quoting any number.** The leaderboard asks
   that models not train on the test datasets; the base trained on voices_b, voices_a and
   studio_b, so any of those used as a Qaf-ft test is contaminated by the base itself.

### Open items

- Enumerate reciters in the studio_b copy on the platform (all splits) and compare with
  every held-out reciter.
- Read the voices_a hosting-service terms and the broadcast_a request-form terms.
- Obtain the voices_b paper for per-subset hours, gender balance and channel selection.
