# Dataset reference

Working reference for every dataset the recitation model trains or evaluates on.

Labels used here and in the rest of `docs/`:

| label | what it is |
|---|---|
| expert | the main training set: 848 h of expert recitation with sound-level labels |
| aligned expert set | the same audio as expert, re-cut with word timings and a speaker-disjoint split |
| phone | crowd-sourced phone clips of single ayahs (about 393,000 clips), plus its 16 kHz mirror |
| phone holdout clips | the phone clips used by bench_a and kept out of training |
| studio_b | per-ayah studio recitations of several reciters |
| studio_c | full-surah studio recordings |
| studio_d | a newer large non-commercial Quran audio set |
| studio_e | a large ayah-level collection with word timings across several readings |
| studio_f | an ayah-level corpus of about 30 professional reciters |
| learner_a | crowdsourced learner clips judged correct or incorrect |
| learner_b | one-verse learner set with tajweed-rule labels (benchmark subset of 159 clips; parent of about 1,500 clips) |
| mdd_b | sound-level mispronunciation set with reference and mispronounced phonemes |
| mdd_c | one-sura word-level error set |
| mdd_e | the mispronunciation test set with reference versus annotation phonemes; "the mdd_e task" is the shared task around it |
| bench_a | the held-out ASR benchmark with studio and phone configs |
| bench_b | a held-out tajweed benchmark of correct recitations |
| tts_a | an ear-verified recitation corpus for TTS |
| base | the pretrained 115M FastConformer Arabic model that is fine-tuned |

## How this file was produced

Sources read on 2026-11-20: each dataset card on the platform (`README.md`), the repository file list from the platform's API, each terms file the card points to, and the papers and project pages the cards link to where the host was reachable. No audio was downloaded. Where a claim rests on something other than the card, the source is named.

Verification levels used below:

- **card**: stated in the dataset card or its YAML header.
- **file list**: read from the platform's API listing of the repository.
- **first rows**: read from the Parquet footer and the text columns of the first row group of one shard (audio bytes were never fetched). This confirms the schema and shows a few rows; it does not prove the whole dataset is uniform.
- **paper**: read in the linked paper.
- **not verified**: could not be checked at the source.
- **not stated**: the card is silent.

Hosts that were blocked from the working machine: a paper-sharing site (learner_b's original paper), the studio_f project page, and the platform's dataset viewer API. Nothing was done to route around them.

Usage terms for every source are kept with the source files, not in this file.

---

## 1. expert

**What it is.** About 848 hours (card table: 847.99 h, 286,537 segments) of Hafs recitation by expert reciters, cut at pauses (waqf) rather than at ayah ends, with a phonetic transcript and ten articulation attributes (sifat) per phoneme group. It is the training set behind model_a.

**How it was built (the expert set's paper; project README).**
1. Complete mushafs by expert reciters were collected from public recitation sites (the `moshaf_metadata` config records `publisher` and `sources` URLs per mushaf). Criteria in the paper: "perfect recitation, studio recording, complete Mushaf, high audio quality".
2. Audio was segmented at pauses with the authors' fine-tuned Wav2Vec2-BERT segmenter.
3. Each segment was transcribed with a public Whisper model fine-tuned on Quran recitation (the paper gives its WER as 5.75%).
4. The transcript was matched to the Quran text with the authors' fuzzy-matching algorithm; "Samples below 85% threshold are sent to human annotators". The paper reports 5,400 of 286,537 segments annotated by hand (a 98% automated pipeline).
5. Imlaei text was converted to Uthmani, and Uthmani to the Quran Phonetic Script (QPS) with the phonetizer package, using each mushaf's recorded reading choices (madd lengths, sakt, and so on).

So the phoneme and sifat labels are **derived from the text plus the mushaf's declared reading choices**, not measured from the audio. The paper's stated assumption is that the input is "100% correct expert recitations".

**Speakers.** 22 reciters, 27 mushafs (card). All male (paper: "the training dataset comprises only expert male reciters"). Country codes in `reciters_metadata` are mostly EG, also SA, SY, SO, KW, AE (first rows, all 22 rows read). Reciters are referred to in this project by moshaf number only.

**Format.**
- Configs: 27 audio configs named `moshaf_<id>` (ids 0.0, 0.1, 0.2, 0.3, 1.0, 2.0, 2.1, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0, 19.0, 22.0, 24.0, 25.0, 26.0, 26.1, 27.0, 28.0, 29.0, 30.0), plus `moshaf_metadata` (27 rows) and `reciters_metadata` (22 rows). Every config has one split, `train`.
- Audio: 16 kHz mono, embedded in Parquet (card). Container/codec inside the Parquet: not stated.
- Columns: `audio`, `segment_index` (string, e.g. `001.0001`), a raw ASR transcript column (list of strings), `moshaf_id`, `moshaf_name`, `reciter_id` (int32), `reciter_arabic_name`, `reciter_english_name`, `sura_or_aya_index`, `index_type`, `sample_rate` (int32), `duration_seconds` (float32), `timestamp_seconds` (list of float32: start and end inside the source track), `imlaey`, `uthmani`, `has_quran`, `has_istiaatha`, `has_bismillah`, `has_sadaka` (bools), `start_span` / `end_span` (struct of `sura_idx`, `aya_idx`, `imlaey`, `uthmani` word indices; null for non-Quran segments), `match_ratio` (float32), `phonemes` (string), `sifat` (list of structs).
- `sifat` struct fields: `phonemes`, `hams_or_jahr`, `shidda_or_rakhawa`, `tafkheem_or_taqeeq`, `itbaq`, `safeer`, `qalqla`, `tikraar`, `tafashie`, `istitala`, `ghonna`. Values are English identifiers such as `jahr`, `shadeed`, `moraqaq`, `not_moqalqal`.
- Label alphabet (QPS): Arabic-script symbols, 43 phoneme symbols (paper, Table 10). Madd length is written by **repeating** the vowel symbol; shadda by doubling the consonant; special symbols include `ۦ` (yaa madd), `ۥ` (waw madd), `ں` (hidden noon), `۾` (hidden meem), `ڇ` (qalqalah), `ۜ` (sakt). Words are separated by spaces.
- Example row (first rows, `moshaf_0.0`): `uthmani` = `بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ`, `phonemes` = `بِسمِ للَااهِ ررَحمَاانِ ررَحِۦۦم`, `match_ratio` = 1.0, `timestamp_seconds` = [3.63, 7.25].
- Size: per-config download sizes in the card range from about 1.95 GB (`moshaf_25.0`) to 5.8 GB (`moshaf_2.0`); the sum over the 27 audio configs is roughly 96 GB (computed from the card's `download_size` values).

**Loading only what is needed.** One directory per config: `moshaf_<id>/train-000NN-of-000MM.parquet` (5 to 12 shards each), `moshaf_metadata/train-00000-of-00001.parquet`, `reciters_metadata/train-00000-of-00001.parquet`.

```python
ds = load_dataset(EXPERT_REPO, name="moshaf_0.0", split="train")
meta = load_dataset(EXPERT_REPO, name="moshaf_metadata")["train"]
```

**Quirks and warnings.**
- Segments are pause-level, not ayah-level; a segment can span several ayahs or part of one. Use `start_span` / `end_span`.
- Non-Quran segments are present and flagged: isti'adha, bismillah, and "sadaka" closings (`has_quran` false; spans null). (first rows)
- The paper says "Moshaf 25.0 was excluded due to poor segmentation", but the dataset ships a `moshaf_25.0` config (7,922 segments, 16.9 h) and the card's totals include it. This project excludes it.
- The authors' own test set is mushafs 19.0, 29.0 and 30.0 (paper, Table 8). The card does not define train/test splits. This project uses 5.0 and 13.0 for test and 19.0 for validation.
- One reciter has four mushafs (0.0 to 0.3) and two others have two each, so a split by mushaf is not a split by speaker. Use `reciter_id`.
- Rare rules (imala, rawm, tasheel) and letter-specific attributes are under-represented (paper, limitations).
- Several of these reciters are widely used in other Quran corpora (studio_b-style collections, studio_f, studio_d, studio_e), so overlap with other sets by speaker is likely. Exact overlap: not verified.
- Card typo: `tafkheem_or_taqeeq` is the real column name (not "tarqeeq").

**Usage terms.** Usage terms are kept with the source files. The rights in the underlying recitation audio (collected from third-party sites listed in `moshaf_metadata.publisher`) are not stated on the card. model_a's code and weights carry their own separate terms, which apply only if they are used.

**Fit for this project.** Train. It is the only large set with sound-level labels that encode tajweed. Remember the labels describe the correct reading, so it teaches correct recitation only; hold out by `reciter_id`, and keep mushafs 5.0 and 13.0 out if section 12's test split is used.

---

## 2. phone (and its 16 kHz mirror)

### phone

**What it is.** Recordings of single ayahs paired with the ayah text. The card describes the content only as "audio recitations of Quranic Ayahs and their corresponding Quranic texts". Who the speakers are and how the audio was collected: not stated on the card. (Another card, section 10, describes this set as "REAL USER PHONE RECORDINGS"; that is a third party's description.)

**Labels.** The card: "The labels were inferred from the Surah/Ayah number of the audio filename and are not validated further." So the label is the text the speaker was supposed to recite, not a transcript of what was said. Expect mistakes, partial recitations, repeats, silence and noise under a clean-looking label.

**Format.**
- Splits: `clean` ("audio files that follow a valid filename format and is not corrupt") and `unclean` ("do not follow a valid filename format and/or is corrupt"). Size tag: 100K<n<1M.
- Columns: `audio` (struct of `bytes`, `path`), `label`, `is_clean` (bool).
- `label` is a **string** in `clean` (Uthmani script with tashkeel) and an **int64 equal to 0** in `unclean` (first rows). The two splits cannot be concatenated without a cast.
- Audio sample rate: not stated. The card says the file name format is `surahNum_ayahNum_id.wav`; the first rows of `clean` show `.flac` paths (`100_10_1086409507.flac`). `unclean` paths look like `<uuid>_<timestamp>_float64_48000_1.wav`.
- Example row (first rows): `label` = `وَحُصِّلَ مَا فِى ٱلصُّدُورِ`, `is_clean` = true, path `100_10_1086409507.flac`.
- Files: `data/clean-000NN-of-00411.parquet` (411 shards), `data/unclean-000NN-of-00332.parquet` (332 shards). First clean shard is about 159 MB for 970 rows. Total size: not stated.

**Loading.** `load_dataset(PHONE_REPO, split="clean")`. Shards appear sorted by file name, so the first shards hold a few ayahs many times over (first rows of shard 0 are all the same ayah); shuffle or sample across shards. Locally the set lives in `data/phone_src`.

**Quirks.**
- Gated (`gated: auto`); the gate text is not in the card metadata: not stated.
- No speaker ids. A speaker-disjoint split is not possible from the data as published.
- 200 clips are reused as the phone holdout clips in section 10; those must be removed from training (ids are the file names, listed in `data/phone_holdout_ids.json`).
- Some `clean` files do not decode: the mirror below found 5,563 of 398,490.

**Usage terms.** Usage terms are kept with the source files. Consent or rights of the people recorded: not stated.

### 16 kHz mirror

**What it is.** A third-party filtered copy of the `clean` split: loaded without decoding, "Validated with `soundfile`", "Removed samples that failed decoding", "Converted valid audio to FLAC, mono, 16 kHz".

**Format.** One split, `clean`; files `data/clean-00000.parquet` to `data/clean-00078.parquet` (79 shards, 5,000 rows each in shard 0, about 434 MB). `stats.json`: 398,490 seen, 392,927 valid, 5,563 bad. `bad_audio.jsonl` lists the rejected files. Columns in shard 0 (first rows): `audio`, `label` (string), `duration` (float, seconds), `is_clean`. The card says "Text is stored in `sentence` and `label`", but shard 0 has **no `sentence` column**.

**Usage terms.** The mirror adds no terms of its own; the upstream terms, kept with the source files, still apply. Not gated.

**Duplicates.** The same recording appears under different ids, so holdout clips must be removed by audio (decoded-sample fingerprint), not by file name.

**Fit for this project.** Train with care, and evaluate. This is the closest thing to real learner audio, but labels are the intended text, not what was said; use it for robustness to real microphones and for weakly supervised training, and filter by an alignment score before trusting any clip. Use the mirror for convenience (16 kHz, decodable). Exclude the 200 phone holdout clips.

---

## 3. studio_d

**What it is.** A large ayah-level corpus: "1400+ hours", "600+" reciters, "365,000+" clips, with automatic word-level timestamps.

**How it was built (card and studio_d's paper).** Fully automatic. Long recordings were collected from "public Qur'anic repositories and archives"; an LLM extracted surah and reciter from titles and descriptions; Whisper Large v3 with a word-timestamp aligner produced word timestamps; an "Ayah Alignment Module" matched transcript segments to canonical ayah text by embedding similarity; a recitation-stop segmenter refined each clip end (after adding a 5-second buffer); near-duplicates for the same reciter and ayah were merged by audio-embedding similarity above 0.9.

**Known label noise (paper, limitations).** "the word-level timestamps are not always precise ... the alignment model used was not originally built for Quranic audio". Some reciters lack ayahs because the pipeline "failed to correctly match the audio to the right ayah". Reciter identity comes from LLM-read metadata, so reciter ids can be wrong or split: not verified. No human verification step is described.

**Format.**
- Configs: `default` (split `train`, `data/train-*.parquet`, 385 shards) and `preview` (`preview.parquet`, 300 rows).
- Columns (first rows): `audio` (struct `bytes`, `path`), `reciter_id` (int32), `surah_id` (int32), `ayah_id` (int32), `text_ar_simple` (no diacritics), `text_ar_uthmani`, `ayah_duration_s` (float), `metadata` (a JSON **string** holding the same fields plus `word_alignments`: list of `{word, start, end}` in seconds, and `audio_filename`).
- Audio: `.wav` per the card; the preview shard's embedded feature metadata declares 16 kHz. Not otherwise stated.
- File name convention: `<set>_spk{reciter_id:04d}_S{surah_id}_A{ayah_id}_{hash}_{segment:06d}.wav`.
- `sheikh_dict.json` maps reciter name to `reciter_id`; `surah_dict.json` is the surah map.
- Shard 0 is about 2.4 GB for 1,000 rows. Total size: not stated on the card (roughly 385 shards of that size if uniform: not verified).
- Example row (card): `"reciter_id": 88, "surah_id": 3, "ayah_id": 82`, `text_ar_uthmani` = `أَفَلَا يَتَدَبَّرُونَ ٱلْقُرْءَانَ ...`.

**Quirks.**
- **`surah_id` is zero-based, `ayah_id` is one-based.** The card's own example (surah_id 3, ayah 82) is the text of 4:82; the first row of shard 0 (surah_id 24, ayah 9) is the text of 25:9; the first preview row (surah_id 77, ayah 30) is 78:30. Add 1 to `surah_id` before joining with any other set. The card does not say this.
- The card claims "complete coverage of all Qur'an"; the paper says "complete coverage of 113 surahs excluding Al-Fatiha".
- Word alignments overlap (a word's `end` can be later than the next word's `start`) and the words in `word_alignments` are undiacritized in the data (first rows), although the card's example shows diacritized words.
- "Some reciters may appear multiple times for the same surah and ayah" (card): intentional repeats, so random splits leak.
- Riwaya is not recorded per clip: not stated. Clips in readings other than Hafs may be present: not verified.
- Speakers overlap heavily with every other professional-reciter set here (the reciter list includes the best-known voices). The paper's alignment evaluation uses five such reciters.

**Usage terms.** Usage terms are kept with the source files. The card also lists prohibited uses: "Mockery or distortion of Qur'anic recitation", "Disrespectful or misleading audio synthesis", "Any application that may cause offense to Muslim communities". Rights in the source recordings: not stated ("collected from public Qur'anic repositories and archives").

**Fit for this project.** Optional training data for speaker variety only, text-level targets only; not for sound-level or timing supervision and not for evaluation. Its non-commercial, research-and-education terms would have to be accepted for the released model, and the card does not say whether a model inherits them.

---

## 4. The mdd_e task's training set

**What it is.** The real-speech training set of the mdd_e mispronunciation-detection shared task. The card is YAML only. From the mdd_e paper and the task page: a subset of voices_a's Arabic portion (read Modern Standard Arabic by volunteers), "augmented ... with samples drawn from Qur'anic recitations", automatically diacritized with an in-house vowelizer, then phonemized. The task page calls it "Native Speech (Pseudo-Labeled)": speakers "are assumed to pronounce the text correctly".

**This is not recitation with tajweed.** It is MSA read speech. First rows confirm audio paths taken from voices_a.

**Format.**
- One config; splits `train` (71,391 rows, 29 shards) and `dev` (2,588 rows, 1 shard). Download 12.2 GB. Paper: 79 h train, 3.33 h dev.
- Columns: `id` (string), `phoneme_ref` (string), `phoneme_aug` (string), `audio` (16 kHz), `sentence` (string), `tashkeel_sentence` (string).
- `phoneme_ref` and `phoneme_aug` were identical in the rows read. The task page documents a `phoneme_mis` column for this set; the actual column is `phoneme_aug`.
- Label alphabet: Latin ASCII transliteration, space-separated, from an MSA text-to-speech phonetizer. 68 phonemes per the paper. Gemination is a doubled symbol (`bb`, `ll`); long vowels are `aa`, `ii`, `uu`; emphatic-context vowels are upper case (`A`, `AA`, `I`, `U`, `UU`); hamza is `<`, ayn is `E`, sheen is `$`, dhal is `*`, tha is `^`. No word boundaries.
- Example row (first rows): `sentence` = `فِيهِنَّ خَيْرَاتٌ حِسَانٌ`, `phoneme_ref` = `f ii h i nn a x A y r aa t H i s aa n`.

**Loading.** `data/train-000NN-of-00029.parquet`, `data/dev-00000-of-00001.parquet`.

**Quirks.** Labels are pseudo-labels: diacritics come from a model, and pause forms are applied (final short vowels dropped), so the phonemes are what an MSA reader would say, not the tajweed reading. No speaker ids. The task organisers state there is "no overlap in speakers or content (sentences) between the training datasets ... and the evaluation datasets".

**Usage terms.** Usage terms are kept with the source files.

**Fit for this project.** Not for training the recitation model as is: different alphabet, no tajweed, mostly non-Quran MSA speech. Useful only if the mdd_e benchmark is a target, in which case a separate output head or a mapping to this alphabet is needed.

---

## 5. The mdd_e task's synthetic set

**What it is.** Synthetic speech from the task organisers' single-speaker text-to-speech systems (the paper says seven in-house voices), with correct and deliberately corrupted text. Errors were injected by changing characters or diacritics using a confusion table, then synthesised.

**Format.**
- One split, `train`: 55,369 rows, 13 shards, 5.9 GB. (Task page: about 80 h; paper: 52 h for the earlier version.)
- Columns: `audio` (16 kHz), `sentence_ref`, `sentence_aug`, `speaker` (string, a voice name), `label` (string; `augmented` in the rows read, other values not verified), `phoneme_ref`, `phoneme_mis`.
- Same Latin alphabet as section 4, **with two differences seen in the first rows**: `phoneme_ref` is wrapped in `<sil>` tokens at both ends, and `phoneme_mis` has leading and trailing spaces and writes gemination as two separate symbols (`l l`, `b b`) where `phoneme_ref` uses one (`ll`, `bb`).
- Example row: `sentence_ref` = `الحَمدُ لِلَّهِ رَبِّ العالَمينَ`, `phoneme_ref` = `<sil> < a l H a m d u l i ll a h i r a bb i l E aa l a m ii n a <sil>`, `phoneme_mis` = ` < a l h a n d a l i l l a E i r a b b i l E aa l a m ii n `.

**Quirks.** Synthetic voices only; the mispronunciations are whatever the TTS produced for corrupted text, not human errors. The text is Quran verses (rows read), read without tajweed.

**Usage terms.** Usage terms are kept with the source files.

**Fit for this project.** Not for training. Synthetic non-tajweed speech in a different alphabet would teach the wrong reading style.

---

## 6. mdd_b

**What it is.** About 2 hours of real human mispronunciations collected for the 2026 round of the mdd_e task (task page). Speakers, collection method and annotation method: not stated. Locally in `data/mdd_b`.

**Format.** One split, `train`: 1,330 rows, one shard (`data/train-00000-of-00001.parquet`, 208 MB). Columns: `id` (string, e.g. `7973_0`), `audio` (16 kHz), `sentence`, `phoneme_ref`, `phoneme_mis`. Same Latin alphabet as section 4. In the rows read the sentence text ends with a stray `@` character.

Example row: `sentence` = `وَاتَّقُوا اللَّهَ الَّذِي تَسَاءَلُونَ بِهِ وَالْأَرْحَامَ@`, `phoneme_ref` ends `... < a r H aa m a`, `phoneme_mis` ends `... < a r H aa m` (final vowel dropped).

**Quirks.** The first three rows are the same sentence by what look like three speakers (id prefix differs), so the set is few sentences by many speakers; split by sentence and by speaker prefix. Gated (`auto`). Whether the "error" in the example is a real mistake or a normal pause form is a judgement the labels make in MSA terms, not tajweed terms.

**Usage terms.** Usage terms are kept with the source files.

**Fit for this project.** Evaluation only, and only as a secondary check, because the reading style is MSA without tajweed and the alphabet differs.

---

## 7. mdd_e

**What it is.** The mdd_e task's test set, split across three repositories (locally in `data/mdd_e/set`):

| part | content | rows | gated |
|---|---|---|---|
| audio | audio only: `ID`, `audio` | 1,642 | no |
| references | `ID`, `Reference_phn` (what should have been said) | 1,642 | no |
| ground truth | `labels_test.csv`: `ID`, `Reference_phn`, `Annotation_phn` (what was said) | 1,642 | yes (auto) |

**How it was built (the mdd_e paper, describing v1).** Quran verses "read aloud by native Arabic speakers", who "were instructed to read the text in MSA at their normal tempo, disregarding Qur'anic tajweed rules, while deliberately producing the specified pronunciation errors", using a recording tool that highlighted the change to make. The task page describes the test set as "98 sentences × 18 speakers ≈ 2 h, with deliberate errors and human annotations". How v2 differs from v1: not stated.

**Format.**
- Audio part: split `test`, `data/test-00000-of-00001.parquet` (220 MB). Audio sample rate not declared in the card. Paths like `00000_00001.wav`. The repository also holds `data/train-00000-of-00001.parquet`, which is not declared in the card and is the same size and has the same first rows as the test file: a duplicate, not a training split.
- IDs look like `00000_00001` (probably speaker_sentence: not verified).
- Alphabet: the Latin set of section 4. Across both columns of `labels_test.csv` 54 distinct symbols occur. `Reference_phn` and `Annotation_phn` differ in 1,532 of 1,642 rows.
- Example row: `Reference_phn` = `< i nn a m aa y a x $ aa ll a h a m i n E i b aa d i h i l E u l a m aa < u`, `Annotation_phn` = `< i nn a m aa y a x $ aa ll a h a m i n E i b aa d i h u l E u l a m < u`.

**Quirks.** The errors are acted, not natural, and the speakers are native Arabic speakers reading without tajweed. Join the three parts on `ID`. The organisers state no speaker or sentence overlap with their training sets.

**Usage terms.** Usage terms are kept with the source files. Do not redistribute the label files.

**Fit for this project.** Evaluation only, as an outside reference point for sound-level error detection. Results need a mapping from the project's sound tokens to this MSA alphabet, and scores will be depressed by the style mismatch (no tajweed).

---

## 8. mdd_c

**What it is.** Recordings of the four verses of Surah 112 by many participants, each labelled correct or containing an error, with the error described in Arabic. Locally in `data/mdd_c`.

**How it was built.** Not stated: who the participants are, how they were recorded, who labelled, and how many labellers. File names give a participant number; 384 distinct participant numbers appear in `metadata.csv`.

**Format.**
- `data/*.wav` plus `metadata.csv`. The card says 1,506 samples (851 error, 655 correct). `metadata.csv` has 1,506 rows, but the repository has **1,500 audio files**: six file names appear twice in the metadata (`ID42V2T`, `ID249V3F`, `ID328V3T`, `ID41V4T`, `ID266V4F`, `ID286V4F`).
- Columns: `file_name` (string), `label` (int: **0 = error, 1 = correct**), `label_name` (`error` / `correct`), `verse_number` (1 to 4), `verse_text`, `error_type` (Arabic; values seen: `تجويد`, `حركات`, or combinations), `error_location` (the word), `error_explanation` (Arabic free text), `error_count` (described as "Error category number"; values 0 to 40, not a count).
- File name pattern: `ID{participant}V{verse}{T/F}.wav`, `T` correct, `F` error.
- Audio: WAV; sample rate not stated (the card's loading example resamples to 16 kHz).
- Example row: `data/ID1V1F.wav,0,error,1,قُلْ هُوَ اللَّهُ أَحَدٌ,تجويد,أَحَدٌ,خطأ في قلقلة حرف الدال,5`.
- Total size: not stated.

**Loading.** The card's `load_dataset(...)` call yields a `train` split from the audio folder; reading `metadata.csv` directly is more reliable given the duplicated rows.

**Quirks.** One label per clip, with a word-level location but no sound-level transcript of what was said. Labels are unverified by any stated process. The label is encoded in the file name, so never feed the path to a model. Deduplicate the six repeated rows (check whether the two rows agree).

**Usage terms.** Usage terms are kept with the source files. Consent of the participants: not stated.

**Fit for this project.** Evaluation only: a small real-learner check of whether mistakes are flagged on the right word in four short verses. Too small and too narrow to train on.

---

## 9. bench_b

**What it is.** The public half of a benchmark for recitation checkers: correct recitation only. "3,910 recordings, 9.2 hours, each with the mushaf text, the phonemes and every legal way of reading its ayah. Most are by 22 professional reciters whose recordings were held out of training." The half with mistakes is private: "Those recordings are not published, so that nobody can train on them."

**How it was built.** Audio "comes from published recitations; the `source_url` column gives the original recording where there is one" (first rows show a recitation archive URL and a video URL). Each row carries a `certification` JSON describing the checks it passed (provenance, riwaya, coverage, and others). Reference phonemes come from the authors' own labeller: details not stated on the card.

**Format.** All audio "16 kHz mono FLAC". Every config has one split, `test`.

| config | recordings | hours | content |
|---|---|---|---|
| `correct_t1p_dev` | 930 | 2.20 | held-out panel, development half: 6 professional reciters |
| `correct_t1p_test` | 1,780 | 4.24 | held-out panel, test half: 16 professional reciters |
| `correct_t1` | 153 | 0.34 | an older held-out reciter |
| `correct_t1x` | 1,044 | 2.37 | 4 reciters whose voices are in the authors' own training mix |
| `correct_t1w_test` | 3 | 0.03 | women: 2 reciters, 3 recordings |

Columns (first rows): `id`, `audio`, `tier`, `group`, `half`, `duration` (float), `ayah` (string `surah:ayah`), `uthmani`, `ref_tj3` (what the audio says), `canonical_tj3` (the written text in the same tokens), `alt_tj3` (list), `clip_is_error`, `identity_control` (bools), `n_error_sites` (int32), `error_classes` (list), `speaker_id`, `speaker_name`, `gender`, `age_group`, `voice_status`, `channel`, `source_dataset`, `source_url`, `seen_by_b7_models` (bool), `labels` (JSON string), `lattice` (JSON string), `certification` (JSON string).

Label alphabet ("tj3"): space-separated tokens with ` | ` between words. Consonants are Arabic letters; vowels are Latin (`a`, `i`, `u`, `aa`, `ii`, `uu`); `ں` marks a hidden noon. Madd length is **not** in the token string; it is in `labels.madd` as allowed lengths per site. Example (first rows, 50:2): `ب a ل | ع a ج i ب uu | ء a ں | ج aa ء a ه u | ...`.

`lattice` lists, per word, the legal forms for continuing and stopping; "A model is never charged for a legal stop or a legal reading." `labels.sites` is empty for correct recitation.

**Loading.** `data/<config>/test-00000-of-00001.parquet`. The scorer is in `scorer/` (`parquet_to_bench.py`, `score_v4_bench.py`, `word_map.json.gz`).

```python
correct = load_dataset(BENCH_B_REPO, "correct_t1p_test", split="test")
```

**Quirks and warnings (card).**
- "Recordings by women are few (three), and there are none by children yet."
- `correct_t1x` is "held out for other models, not for ours" (the authors' models).
- `seen_by_b7_models` marks near-copies of audio in the authors' older training mixes.
- "If you train on any of this data, say so when you report numbers."
- The clips are cut from published recitations by professional reciters; the same recordings may be inside studio_d or studio_e. Check `source_url` and `speaker_name` against training data before trusting a score. Overlap: not verified.
- Only the correct half can be scored locally; the checker score needs the private half, run by the authors on request.
- Gated (`auto`).

**Usage terms.** Usage terms are kept with the source files. Note that its terms treat any model evaluated on it as a derivative, and that upstream audio keeps its own terms.

**Fit for this project.** Evaluation only, never training (the card asks that it stay held out). Before using it, check that its terms fit how the released model will be published, because evaluating on this set already counts under those terms.

---

## 10. bench_a

**What it is.** "A small, leakage-free benchmark (600 clips) for evaluating Arabic ASR on Quranic recitation (Hafs riwayah)". Three sources, 200 clips each. Text-level references only (no sound-level labels). Locally in `data/realworld/bench_a`.

| config | n | content (card) |
|---|---|---|
| studio_b holdout | 200 | 3 studio_b reciters with **zero** training samples (clean studio) |
| recitation-library holdout | 200 | a reciter from a recitation library, "never trained on; references from the canonical mushaf" |
| phone holdout clips | 200 | "**Real phone-mic** recitation held out from training" |

"Every clip is verified absent from **our** training data": the leakage guarantee is relative to the benchmark authors' models, not to this project's.

**Format.**
- Each config has one split, `test`. Layout: `audio/<source>/*.wav` ("16 kHz mono audio") plus `audio/<source>/metadata.jsonl`; root files `benchmark.jsonl`, `score.py`, `results.json`, and a terms file.
- Rows via `load_dataset`: `audio`, `text`, `id`, `source` (`metadata.jsonl` fields: `file_name`, `text`, `id`, `source`).
- `benchmark.jsonl`: 600 rows of `id`, `source`, `reference_text`, `audio` (relative path).
- Reference text style differs by source: phone holdout references are Uthmani with Quranic marks (`وَأَنَّهُۥ لَمَّا قَامَ عَبْدُ ٱللَّهِ ...`); the studio_b and recitation-library references are simple-script with diacritics (`فَكَيْفَ تَتَّقُونَ إِنْ كَفَرْتُمْ ...`). The scorer normalises: "combining marks + Quranic annotation signs + tatweel stripped; hamza forms and alef-maqsura unified", with an extra "alef-insensitive" column.
- Size: about 600 short WAV files; total bytes not stated.

**Quirks.**
- The phone holdout config has 200 metadata rows but **199 audio files**: `11_90_2170378964.wav` is listed twice (file list and `metadata.jsonl`).
- The card's code sample loads an older repository id.
- Revision v1.1 (2026-08-15) corrected the references of ten recitation-library clips where the reciter repeats the ayah (`37_43, 37_83, 24_20, 50_21, 43_46, 6_132, 6_102, 17_72, 6_32, 51_56`). Scores from before and after are not comparable for those clips.
- The phone holdout clips come from phone (file names match its `surah_ayah_id` pattern): remove them from any phone training data (`data/phone_holdout_ids.json`). The studio_b holdout comes from the public studio_b test split, which the card says "overlaps most Quran ASR training sets"; base was trained partly on studio_b (section 17), so that config is not held out for this project unless checked.
- Gated (`auto`) with an agreement.

**Usage terms.** Usage terms are kept with the source files. The access agreement limits use to ASR research and evaluation and forbids redistributing the audio; the stricter reading is the safe one.

**Fit for this project.** Evaluation only: a small text-level accuracy check, with the phone holdout clips the meaningful part. Same terms question as section 9.

---

## 11. learner_b

**What it is.** 159 recordings of one ayah (5:109, `قَالُوا۟ لَا عِلْمَ لَنَآ إِنَّكَ أَنتَ عَلَّٰمُ ٱلْغُيُوبِ`) by 159 different speakers of mixed age and gender, with real mistakes, re-annotated for madd lengths, ghunnah, qalqalah, and a full phonetic transcript of what was actually said. Locally in `data/learner_b`.

**How it was built.** Audio traces back to 11 kHz messaging-app recordings, so it is narrowband (phone-like). Derived from learner_b's parent set (about 1,500 clips: 160 or 161 speakers each recording the same verse about ten times). The authors kept one randomly chosen clip per speaker and re-annotated it. The learner_b paper: "The dataset was annotated by a single hafiz of the Moshaf with Tajweed knowledge"; and in limitations, the benchmark "uses a single annotator and one verse".

**Format.**
- One split, `train` (it is a test set despite the name): 159 rows, `data/train-00000-of-00001.parquet`, 85.9 MB.
- Columns: `audio` (sample rate **not declared**: the card says "sampling rate: None, mono channel"; the parent set's card says that set was downsampled to 16 kHz), `id`, `original_id` (e.g. `s0_1`: speaker 0, take 1), `gender` (string), `age` (int64), `qalo_alif_len`, `qalo_waw_len`, `laa_alif_len`, `separate_madd`, `allam_alif_len`, `madd_aared_len` (int64, 0 to 8 harakat), `noon_moshaddadah_len` (0 = partial, 1 = complete), `noon_mokhfah_len` (0 = plain noon, 1 = partial, 2 = complete), `qalqalah` (0 / 1), `phonetic_transcript` (string, QPS as in section 1), `sifat` (list of structs as in section 1).
- Example row (first rows): `original_id` = `s0_1`, `age` = 14, `separate_madd` = 3, `phonetic_transcript` = `قَاالُۥۥ لَاا عِلمَ لَنَااا ءِننننَكَ ءَںںںتَ عَللَاامُ لغُيُۥۥبڇ`.

**Quirks.**
- Same alphabet as section 1, so it scores expert-style output directly. Here the transcript records what the learner said (a 3-count madd is written with three vowel symbols), unlike section 1 where it is the prescribed reading.
- The parent set had mislabelled and duplicated ids (listed on its card); what was fixed in the benchmark: not stated.
- The parent set holds the other takes by the same 159 speakers. Training on it leaks speakers and the exact sentence.
- The card says the verse is 5:109; the parent card quotes a different verse text (2:32). The benchmark card and the paper agree on 5:109.

**Usage terms.** Usage terms are kept with the source files. Consent of the speakers: not stated.

**Fit for this project.** Evaluation only. It is the one public set with real learner mistakes labelled in the same sound alphabet as the main training data; single verse and single annotator limit how far a result generalises.

---

## 12. The aligned expert set

**What it is.** A derivative of section 1: the same audio "restricted to segments of at most 30 seconds, relabeled in" studio_b's spelling, "with a start and end time for every word, produced by CTC forced alignment". Text-level labels plus word timings; **no phoneme or sifat labels**.

**How it was built (card).** Labels come from each segment's word span, rewritten in the simple spelling with diacritics; "isti'adha and bismillah get fixed text and sadaka segments are dropped". Timings come from a FastConformer CTC model trained on Quran recitation, "aligned against the assimilated spelling it was trained on and mapped back word for word. Segments over 30 s, that fail to align or align with a low score are skipped".

**Splits.**

| split | clips | hours | reciters | selection |
|---|---|---|---|---|
| train | 260,636 | 758 | 20 (25 mushafs) | every segment up to 30 s that aligned, minus validation |
| validation | 2,572 | 7.5 | same 20 | 1% of the same reciters' segments |
| test (unseen) | 20,604 | 60 | 2 (mushafs 5.0, 13.0) | reciters never trained on |

**Format.**
- Not a `datasets` repository: tar archives. `train/moshaf_<id>.tar` (25 files), `validation.tar`, `test_unseen.tar`, `meta/train.tsv`, `meta/validation.tsv`, `meta/test_unseen.tsv`, `meta/manifests.tar` (per-Parquet manifests and skip lists).
- Each clip: 16 kHz mono FLAC with a `.wrd` file, "one line per word: `start end word`, with start and end in **samples**".
- Manifest columns (TSV, verified from the header): `wav_path`, `tg_path`, `raw_text`, `reciter`, `duration` (s), `align_score` (mean log-probability per frame), `reciter_id`, `moshaf_id`, `style`, `match_ratio`, `segment`.
- Example manifest row: `.../test_unseen/audio/moshaf_13.0/001.0001.flac`, `raw_text` = `بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ`, `duration` 6.560, `align_score` -0.0239, `style` murattal.
- Total size: not stated.

**Quirks.**
- Manifests hold absolute paths from the build machine (`/workspace/.../prepared-.../...`); "replace that prefix with the extraction directory".
- Validation shares speakers with train (1% of the same reciters' segments); only `test_unseen` is speaker-disjoint.
- `segment` equals section 1's `segment_index`, so rows can be joined back to the phoneme labels by (`moshaf_id`, `segment`).
- The aligner was itself trained on Quran recitation, possibly including these recordings; timings are model output, not hand-checked.

**Usage terms.** Usage terms are kept with the source files; they follow the expert set's.

**Fit for this project.** Use the split definition and the word timings, not as a separate corpus: it is the same audio as section 1. Adopt mushafs 5.0 and 13.0 as the speaker-disjoint test set and join word timings onto section 1's phoneme labels. Word-timing files live in `data/timings_a`.

---

## 13. learner_a

**What it is.** Crowdsourced recitations by non-Arabic speakers, collected through a mobile prayer app, with crowd and expert labels of whether the recitation is correct. Card: "around 7000 Quranic recitations from a pool of 1287 participants across more than 11 non-Arabic countries, and we have annotated 1166 recitations from the dataset in six categories." Locally in `data/learner_a`.

**How it was built (card and the learner_a paper).** Volunteers recorded short verses in an app; audio was converted to WAV mono 16 kHz with silence removal and noise reduction (paper). Annotation was done by volunteers on a web platform after a training session, with control tasks; the final label is a majority vote, or an expert label for "golden" samples. Reported quality: "a crowd accuracy of 0.77, an inter-rater agreement of 0.63 between the annotators". The paper reports 11.5 hours in total.

**Format.**
- One split, `train`: 6,828 rows, 3 shards (`data/train-0000N-of-00003.parquet`), 1.26 GB.
- Columns: `audio` (16 kHz per the card's example), `Surah` (string, English name such as `Al-Faatihah`), `Aya` (string, the Arabic text; spelling style varies between rows), `duration_ms` (int64), `create_date` (string), `golden` (bool), `final_label` (string or null), `reciter_id` (string; `Unknown` occurs), `reciter_country`, `reciter_gender`, `reciter_age`, `reciter_qiraah` (strings, often null), `judgments_num` (int64), `annotation_metadata` (JSON string with each annotator's label and quality scores).
- `final_label` values: `correct`, `in_correct`, `not_related_quran`, `not_match_aya`, `multiple_aya`, `in_complete`. Most rows have no label (null, `judgments_num` 0).
- Example row (card): `'Surah': 'Al-Faatihah'`, `'Aya': 'أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيْمِ'`, `'final_label': 'in_correct'`, `'reciter_country': 'AT'`, `'judgments_num': 3`.

**Quirks.**
- "correct" means "the pronunciation is correct with the diacritics, **regardless of the rules of Tajweed**". The labels do not judge tajweed.
- Clip-level labels only; no location of the mistake and no transcript of what was said.
- Only about 17% of clips are labelled, and agreement is moderate (0.63).
- Very short clips exist (a 576 ms row labelled `not_related_quran` in the first rows).
- Audio has been denoised and silence-trimmed, so it is not raw phone audio.
- No surah or ayah numbers; match on `Aya` text.

**Usage terms.** Usage terms are kept with the source files. Participant consent terms: not stated.

**Fit for this project.** Evaluation only, internally. It is the best match for the target users (non-Arabic learners) but has only clip-level labels.

---

## 14. studio_e

Access had not been granted when this was written ("awaiting a review from the repo authors"). The card is public and was read in full; no data rows were read. Everything about content below is **card** level.

**What it is.** Ayah-by-ayah audio with word-level timestamps and pause-aware segments, from a project that "unifies recitations on the internet and generates timing data — community-verified results". Badges on the card: 225 recitations, 4 riwayat, "6,150h+". The file list shows 226 configs across `hafs_an_asim/` (1,010 shards), `warsh_an_nafi/` (40), `qalon_an_nafi/` (37), `shubah_an_asim/` (11), plus `mushafs/`.

**How labels were made.** Terms text: "Timing data is produced by forced alignment and human review and may contain errors." The extent of human review per recitation: not stated.

**Format.**
- One config per published mushaf (named by a reciter-and-style slug), single split `train`, one row per ayah. A `mushafs` config is the catalog (one row per mushaf).
- Row schema: `audio` ("Ayah MP3 clip"), `surah` (int32, 1 to 114), `ayah` (int32), `duration_ms` (int32), `text_uthmani` (the exact recited text in a font-oriented Uthmani edition), `segments` (`[[word_from, word_to, start_ms, end_ms]]`), `word_timestamps` (`[[word_idx, start_ms, end_ms]]`, word indices 1-based), `source_url`, `source_offset_ms`.
- Catalog schema includes `slug`, `reciter_id`, `name_en`, `name_ar`, `country`, `riwayah`, `style`, `recording_context`, `recording_year`, `channel`, `audio_category`, `codec`, `container`, `sample_rate_hz`, `channels`, `bitrate_mode`, `bitrate_kbps_nominal`, `total_duration_hours`, `published_at`, `updated_at`.
- Sample rate varies per mushaf (it is a catalog column); audio is MP3, possibly stereo.
- No phoneme labels.

**Loading.** Files are `<riwayah_dir>/<slug>-000NN-of-000MM.parquet`. Filter by `riwayah` in the catalog before loading: the set includes Warsh, Qalun and Shu'bah, which must not be mixed into a Hafs model.

**Quirks (card).**
- "Repeated words are preserved in `text_uthmani` and `word_timestamps`, so the row reflects what the reciter actually recited rather than a plain copy of canonical ayah text." Labels can therefore differ from the canonical ayah.
- Timestamps are relative to the ayah clip.
- The dataset is "constantly expanding": pin a revision.
- The text uses a specific font-oriented Uthmani edition; its code points may differ from the Uthmani text in other sets: not verified.

**Usage terms.** Usage terms are kept with the source files. They cover the timing data only, treat models trained or evaluated on it as derivatives, and forbid public copies of the data.

**Fit for this project.** Do not use until access is granted and its terms are checked against how the released model will be published. Its terms and those of sections 9 and 10 cannot both be met by one released model.

---

## 15. tts_a

The access request for this repository was **rejected** by its authors ("has been rejected by the repo's authors"). The public card page was read; the terms file is served publicly and was read. No data was read. Everything about content is **card** level.

**What it is.** "An ear-verified Quranic recitation corpus for TTS and speech restoration", published "complete, documented, and unmaintained". 64,721 clips, 301.4 h, 16 reciters, Hafs murattal only, 112 surahs, 6,120 unique ayat, mean clip 16.8 s.

**Format (card).**
- `metadata.jsonl` (one row per clip), `data/<reciter>/*.flac` (48 kHz 24-bit masters), `v3_24k/<reciter>/*.flac` (24 kHz mono copies, same file names), `qc_flags.jsonl`. About 150 GB in total, about 56 GB for the 24 kHz copies.
- Fields: `file_name`, `source_id`, `reciter`, `riwaya`, `style`, `surah`, `ayah`, `text_uthmani`, `text_imlaei`, `phoneme_tokens`, `n_tokens`, `duration_s`, `sample_rate`, `bit_depth`, `src_part`, `src_start_s`, `src_end_s`, `boundary`, `align_score`, `recut_match`.
- Label alphabet: a third tajweed inventory, different from sections 1 and 9: "elongation length is part of the token: a two-harakat madd is `ا:2` and a six-harakat madd is `ا:6`, qalqalah is marked (`بڇ`), and shadda/tanwin are carried on the token."

**Quirks (card).**
- "`phoneme_tokens` are canonical prescriptions derived from the text, not measurements of the audio."
- "This is not raw audio." Every clip went through hum removal, dialogue isolation, dereverberation and denoising; the acoustic environment is a product of the cleaning.
- 244 clips (0.38%) are flagged in `qc_flags.jsonl` as disagreeing with their text (extra isti'adha, trailing ayah, long silence, or clipped).
- "roughly 43 % of clips exceed 15 s"; maximum 229.8 s.

**Usage terms.** Usage terms are kept with the source files.

**Fit for this project.** Not at all. Access was refused, and its terms would bind the model to non-commercial share-alike conditions that cannot be combined with the other sets. Processed studio audio is also a poor match for learner recordings.

---

## 16. studio_f

**What it is.** A copy on the platform of an ayah-level recitation corpus, with Uthmani text attached. The card is YAML only: who made the copy, how the text was attached, and what "clean_text" means are **not stated**. studio_d's paper (section 3) describes the original corpus as "226,129 audio-text pairs sourced from 30 renowned reciters", which matches the row count here.

**Format.**
- One split, `train`: 226,129 rows, 53 shards (`data/train-000NN-of-00053.parquet`), 25.6 GB.
- Columns: `audio` (16 kHz declared; embedded files are MP3, paths like `004169.mp3`), `sura` (int64, 1-based), `ayat` (int64), `speaker` (string, a reciter name with underscores), `text` (Uthmani with tashkeel), `clean_text` (identical to `text` in the rows read).
- Example row (first rows): `sura` = 26, `ayat` = 29, `text` = `قَالَ لَئِنِ ٱتَّخَذْتَ إِلَـٰهًا غَيْرِى لَأَجْعَلَنَّكَ مِنَ ٱلْمَسْجُونِينَ`.

**Quirks.**
- Labels are the canonical ayah text attached by ayah number, so they do not reflect repeats, isti'adha or bismillah inside a clip: not verified either way.
- Rows are not grouped by ayah or speaker within a shard (rows 1 and 2 are different surahs, same speaker).
- Professional reciters only; they overlap with the reciters of sections 1, 3 and 14 and with studio_b, which base (section 17) was trained on. Never use as a held-out test.
- Riwaya per speaker: not stated.
- Gated (`auto`).

**Usage terms.** Usage terms are kept with the source files. The upstream terms were not verified (host blocked); until they are read, treat them as unknown.

**Fit for this project.** Hold until the upstream terms are read. If they allow, it is useful as extra text-level training audio (phoneme labels can be generated from `sura`/`ayat` with the section 1 phonetizer, under the assumption that the reading is Hafs and correct); it adds little speaker variety beyond sections 1 and 3.

---

## 17. base

- The pretrained 115M FastConformer Arabic model that is fine-tuned. Usage terms are kept with the source files; they allow commercial and non-commercial use, and the released model must credit it and say it was modified.
- The card says the model "can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset."
- Relevant to evaluation: the card lists studio_b (390 h) among the training data, so any studio_b-derived test clip may have been seen by base.

---

## Cross-cutting notes for training code

**Four label alphabets are in play; none is interchangeable without a mapping.**

| alphabet | used by | script | madd length | gemination | word boundary |
|---|---|---|---|---|---|
| QPS | sections 1, 11 | Arabic symbols | repeated vowel symbol | doubled consonant | space |
| tj3 | section 9 | Arabic consonants, Latin vowels | separate `labels.madd` JSON | not verified | ` \| ` |
| token-with-length | section 15 | Arabic with suffixes (`ا:2`) | in the token | on the token | not stated |
| MSA phonetizer | sections 4 to 7 | Latin ASCII transliteration | none (no tajweed) | doubled symbol, but split in `phoneme_mis` of section 5 | none |

**Text spelling also differs.** Uthmani with Quranic marks (sections 1 `uthmani`, 2, 3 `text_ar_uthmani`, 9, 14, 16, and the phone holdout clips in 10); simple spelling with diacritics (sections 1 `imlaey`, 12, and the other two configs of 10); no diacritics (section 3 `text_ar_simple` and its word alignments). Normalise before comparing or joining. Local text references: `data/text_meta` and `data/text_imlaei.json`.

**Index conventions.** Section 3 uses a zero-based `surah_id`; every other set read here is one-based. Section 12 word times are in samples; sections 3 and 1 use seconds; section 14 uses milliseconds.

**Prescribed versus observed labels.** Sections 1, 12, 15 and 16 label what should have been said. Sections 6, 7, 8, 11 and 13 label what was said or whether it was right. Section 2 labels what the speaker was asked to say. Only the second group can measure mistake detection.

**Type traps.** Section 2: `label` is a string in one split and int64 in the other. Section 3: word alignments are a JSON string inside `metadata`. Sections 9 and 13: `lattice`, `labels`, `certification`, `annotation_metadata` are JSON strings. Section 8: label 0 means error.

**Sample rates.** Declared 16 kHz: sections 1, 4, 5, 6, 9, 10, 12, 16, and the mirror in 2. Not declared: sections 2 (original), 7, 8, 11, 13 (16 kHz per card example and paper). Variable: section 14. 48 kHz / 24 kHz: section 15.

**Leakage map.**
- Section 12 is section 1's audio. Never count them as two corpora.
- The phone holdout clips (section 10) are inside section 2.
- The studio_b holdout config of section 10, and any studio_b clip, may be inside base's training data.
- Professional reciters recur across sections 1, 3, 9, 14, 15 and 16; hold out by reciter identity across corpora, not within one.
- Section 11's speakers have nine more takes each in learner_b's parent set.

---

## Summary table

| dataset | train? | evaluate? |
|---|---|---|
| expert | yes | yes (hold out by reciter) |
| phone | yes, weak labels | yes |
| phone, 16 kHz mirror | yes, as a convenience copy | yes |
| studio_d | optional, text-level only | no |
| the mdd_e task's training set | no | no |
| the mdd_e task's synthetic set | no | no |
| mdd_b | no | secondary |
| mdd_e (audio, references, ground truth) | no | yes, do not redistribute the labels |
| mdd_c | no | yes |
| bench_b | no (held out by request) | yes, if its terms fit the released model |
| bench_a | no | yes, same condition |
| learner_b | no | yes |
| aligned expert set | yes (same audio as expert) | yes (`test_unseen`) |
| learner_a | no | internal only |
| studio_e | not until access and a terms decision | same |
| tts_a | no | no |
| studio_f | hold until upstream terms are read | no |
| base | base weights | n/a |

Usage terms for every source, including which ones restrict a released model or cannot be combined with each other, are kept with the source files.
