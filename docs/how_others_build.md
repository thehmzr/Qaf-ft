# How others build and label recitation data

A playbook drawn from the papers, dataset pages and code read for this project
(details and sources in `datasets.md` and `research_summary.md`). Each section ends with
what this project adopts.

## 1. Expert recitation at scale (expert, 848 hours)

How it is built, almost fully automated:
1. Collect complete recitations by expert reciters (27 mushafs, 22 reciters).
2. Cut the audio at pauses, not at ayah ends, using a model trained to find pauses
   (reported F1 0.9948). Reciters stop mid-ayah, so ayah-based cutting is wrong.
3. Transcribe each piece with an existing recitation model.
4. Match that rough transcript against the Quran text with a moving pointer: search a
   window of about 30 words ahead, accept when at least 85% of letters match. This turns a
   noisy transcript into exact Quran text with start and end positions.
5. Send only the rejected pieces to a person (5,400 of 286,537 were reviewed by hand).
6. Convert the exact text to a sound-level script by rules, with per-reciter settings
   (their madd lengths). Labels are generated from text, not heard from the audio.

Weak points the authors state: only male experts; never saw a mistake; labels say what
should be said, so a reciter's real choice of madd length is assumed, not observed; one
mushaf was badly cut yet still ships.

Adopted: pause-based cutting; "transcribe, then match against the known text" to get exact
labels cheaply; hand review only for what the matcher rejects.

## 2. Three tiers of mistake data (the mdd_e task)

1. Correct speech (about 79 h): assumed correct, so said = expected.
2. Synthetic mistakes (about 80 h): alter the text using a table of common sound
   confusions, then have text-to-speech read it. Stores expected and spoken sounds.
3. Real mistakes (about 2 h): people recorded while making instructed substitutions,
   checked by three annotators.

Every clip has two label rows: what should be said, what was said.
Result that matters: synthetic alone 0.46, real alone 0.67, synthetic then real 0.68,
naive mixing 0.43. A little real data beats a lot of synthetic, and order matters.

Weak points: mistakes are acted, read without tajweed, by native speakers.

Adopted: the two-row label; a confusion table to generate mistakes; training order
"synthetic first, real last"; never mix them naively.

## 3. Crowd labelling of learners (learner_a, about 7,000 clips)

- Collection through a phone app from volunteers in 11+ countries; 1,287 participants.
- Each clip judged by several annotators into six classes (correct, has mistake,
  incomplete, different ayah, several ayahs, unrelated or empty).
- Annotators are graded on hidden control clips with known answers; their accuracy weights
  their vote. Reported crowd accuracy 0.77, agreement between annotators 0.63.
- Only 1,166 of the clips ended up labelled.

Weak points: one label per clip with no location; tajweed ignored by design; agreement is
only moderate, so the labels themselves cap any score.

Adopted: classes for "not a usable clip" (incomplete, wrong ayah, noise) so they are not
confused with mistakes; control clips to measure a labeller's consistency; always report
agreement next to results.

## 4. One verse, many learners, rule-level labels (learner_b)

- About 160 learners each recite the same ayah several times.
- An annotator marks specific tajweed sites: each madd with its length, ghunnah, qalqalah,
  and writes out what was actually said in the sound script.
- learner_b's benchmark subset: 159 clips, 120 of them women, one annotator.

Weak points: one ayah only; one annotator; models tested on it look far better than they
generalise (95-99% reported by simple classifiers).

Adopted: fixed "probe ayahs" that every reciter records, so reciters can be compared on
identical text; marking tajweed at named sites with a measured length, not a vague flag.

- Caution: the 95-99% results reported on learner_b's parent set come from a random clip split with about ten
  takes per speaker, so the same voices are in train and test (speaker leakage).

## 5. Real phone audio with unverified labels (phone, about 393,000 clips)

- Ordinary users reciting into phones; the label is whatever ayah the app expected.
- Nobody checked that the user recited that ayah, finished it, or recited it correctly.
- Teams that train on it first filter: keep a clip only if one or two independent models
  transcribe it close to the expected text (letter error under 0.3, length ratio 0.85-1.15).

Adopted: the same agreement filter before any phone clip is trained on; clips the filter
rejects are kept aside as candidates for hand labelling, since that is where mistakes are.

## 6. Benchmarks that test false flags (bench_a, bench_b)

- A set of correct recitations by reciters held out from training, each with every legal
  way of reading the ayah, so a legitimate variant is never counted as an error.
- A separate set with exactly one planted mistake per recording.
- Score: of 100 recitations, half correct and half with one mistake, how many are judged right.
- A phone channel is simulated by degrading studio audio.

Weak points: mistakes are edited in, not natural; run by a team that also competes.

Adopted (the ideas, not the data): test on correct recitation separately from mistakes;
accept all legal readings; a balanced correct/mistake score; phone-channel augmentation.

## 7. What the winning mistake-detection systems did

- Sound-level output with no language model, so mistakes are not "corrected" away.
- Two training stages; a small head on a frozen or slowly tuned encoder for the real-data stage.
- Agreement filtering: three models transcribe; clips where they disagree are relabelled or dropped.
- Averaging several checkpoints at the end (worth about 0.04 on their score).

## What this means for our own system

| Step | Method taken from |
|---|---|
| Cut recordings at pauses | 1 |
| Get exact labels by matching a rough transcript to the Quran | 1 |
| Filter phone clips by model agreement | 5, 7 |
| Label with two rows: expected and said | 2 |
| Classes for unusable clips | 3 |
| Probe ayahs recited by everyone; tajweed marked at named sites | 4 |
| Control clips and repeat labelling to measure consistency | 3 |
| Generate mistakes from a confusion table | 2 |
| Train synthetic first, real last | 2, 7 |
| Test correct recitation and mistakes separately; accept legal variants | 6 |
| Split every set by reciter | all |

Gaps nobody has filled, which our own labelling targets: natural mistakes located at the
word, across many suras, by ordinary people on phones, including women and children.
