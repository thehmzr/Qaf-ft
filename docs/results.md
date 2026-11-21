# Results

One row per experiment, newest first. Never overwrite a row; add a new one.
False flags: share of correctly recited words flagged. Catch: share of mistakes flagged at that
false-flag rate. Intervals are 95%, from resampling reciters (or clips where there are no
reciter ids). Test set names are fixed in the work plan: `studio-heldout` (reciters 5.0 and
13.0), `studio-val` (19.0), `phone-labelled-v1` (the 67 labels on batch 1: 32 correct, 3 with a mistake, 32 unusable; the
25 "doubted" correct clips were picked by the model as 10-50% off, the 7 controls as close matches).
Word counts: "display words" follow the printed text; "words" follow the sound script's groups,
which join some display words.

| Date | What changed | Data | Test set | False flags | Catch at 1% | Catch at 0.1% | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-21 | False-flag tail on a long correct recitation: segment-free GOP vs current | `full/best.pt` | long recitation, sura 2, one new voice (2,936 test-half words; thresholds from the first half) | GOP: tajweed judged 0.75% flag / 0.31% strong, words and vowels 0.82% / 0.10%. Current: 0.85% / 0.03%, 1.02% / 0.20% | n/a | n/a | Thresholds and test on separate halves. Strong-level tails are a handful of words either way |
| 2026-11-21 | Word score: current (weakest sound / extra sound), baseline for the three word-score rows below | `full/best.pt` | `studio-heldout`, planted, suras 78+, 400 clips (200 per reciter, one shard each) | 1% and 0.1% (in-sample) | 99.5% | 95.3% | AUC 1.000. At 1%: letter, vowel, doubling, ghunnah 1.00, madd 0.99, qalqalah 0.89 (n=9). Words-and-vowels: 100% / 99.4% |
| 2026-11-21 | Word score: segment-free GOP (CTC forward, best one-token edit of the word, top-3 competitors) | `full/best.pt` | `studio-heldout`, planted, suras 78+, 400 clips (200 per reciter, one shard each) | 1% and 0.1% (in-sample) | 81.0% | 51.0% | AUC 0.964. At 1%: vowel 1.00, ghunnah 0.93, letter 0.89, madd 0.65, doubling 0.56, qalqalah 0.00. Words-and-vowels: 82.9% / 51.1% |
| 2026-11-21 | Word score: competing-token margin (worst token) | `full/best.pt` | `studio-heldout`, planted, suras 78+, 400 clips (200 per reciter, one shard each) | 1% and 0.1% (in-sample) | 91.0% | 90.5% | AUC 0.955; close to the weakest-sound part of current alone (91.2 / 90.7); madd 0.70, qalqalah 0.00 |
| 2026-11-21 | Word score: max of rank-normalised current, GOP and margin | `full/best.pt` | `studio-heldout`, planted, suras 78+, 400 clips (200 per reciter, one shard each) | 1% and 0.1% (in-sample) | 99.5% | 96.0% | +0.7 points at 0.1% over current = 3 of 400 words, within noise. max(current,GOP) 95.5%. Words-and-vowels 98.8% at 0.1% (current 99.4%) |
| 2026-11-21 | Speed, this machine (no other load) | `full/best.pt` | 40 phone clips, mean 4.9 s | n/a | n/a | n/a | CPU 258 ms per clip (about 19x real time); GPU backend 42 ms (about 117x); same output as CPU on all 40 clips. 60 s clip timing still to do |
| 2026-11-21 | Reproduce reference numbers on this machine (torch 2.15, training toolkit 3.0, CPU) | `full/best.pt` (step 19,000) | `studio-heldout`, suras 78-114, 400 clips per reciter, one planted mistake each | 1% and 0.1% by construction | 98.3% (97.0-100) | 92.1% (91.7-94.5) | AUC 0.999. 2,798 correct words, 800 planted, 2 reciters. By kind at 1%: letter, vowel, doubling, ghunnah 1.00; madd 0.97; qalqalah 0.84. Same as the run on the GPU notebook service |
| 2026-11-21 | Words-and-vowels mode, thresholds per mode from `studio-heldout` (1,168 clips, 4,158 words) | `full/best.pt` | `phone-labelled-v1`: 25 doubted correct clips (89 words) | tajweed judged 38% (29-49), words and vowels 53% (43-63) at the 1% studio threshold; strong (0.1%) 21% vs 22% | n/a | n/a | 7 control clips: 0% in both, but chosen for decoding well. Mistake clips: 2 of 2 marked words flagged in words-and-vowels mode, 1 of 2 with tajweed judged; correct words in the same clips score higher. Gap is acoustic. Thresholds came from the test reciters; redo from `studio-val` (stage 2.3) |
| 2026-11-20 | Words-and-vowels mode, first run | `full/best.pt` | `phone-labelled-v1`: 25 doubted correct clips | 54% tajweed judged, 56% words and vowels (studio threshold) | n/a | n/a | Different threshold set from the row above |
| 2026-11-20 | First look at phone labels | `full/best.pt` | `phone-labelled-v1`: 25 correct clips (95 display words) + 7 controls | 51% (48 of 95), 19% strong; controls 1 of 26 | n/a | n/a | Studio threshold |
| 2026-11-20 | Full run: all suras, 30,000 steps, about 9 GPU-hours | expert set, all reciters but the three held out | `studio-heldout`, 800 Juz Amma clips, planted | 1% and 0.1% by construction | 98.3% | 92.1% | Token error 0.43% (5,201 segments). Best validation 0.18% at step 19,000; validation swung 0.2-1.5% |
| 2026-11-19 | Pilot: suras 78-114 | expert set, Juz Amma | `studio-heldout`, 800 clips, planted | 1% and 0.1% by construction | 97.0% | 82.5% | Token error 1.2% |
