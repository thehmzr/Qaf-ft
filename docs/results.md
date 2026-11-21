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
| 2026-11-21 | Real mistakes, tajweed judged (`eval/real_mistakes.py`); thresholds from `studio-val` (300 segments, 2,029 words) | `full/best.pt` | `mdd_c`: 397 one-sura clips, 1,250 correct words, 208 word/vowel, 252 tajweed-only | 7.6% (6.1-9.3) at the 1% studio threshold; 1.6% at 0.1% | 44% (37-51) | 5% (2-9) | AUC 0.77 (0.74-0.81). Precision 49% at 1%, 36% at 0.1%. No threshold reaches 90% precision on this set |
| 2026-11-21 | Real mistakes, words and vowels | `full/best.pt` | `mdd_c` as above | 6.2% (4.7-7.9); 1.4% at 0.1% | 45% (38-52) | 5% (2-9) | AUC 0.79 (0.75-0.82). Precision 54% at 1%, 38% at 0.1%. Flags 19% of tajweed-only words (34% tajweed judged) |
| 2026-11-21 | Real mistakes, tajweed judged | `full/best.pt` | `mdd_d`: 441 Hafs chunks, 1,960 correct words, 36 word/vowel, 11 tajweed-only | 23% (20-27); 18% at 0.1% | 22% (10-36) | 8% (0-17) | AUC 0.52 (0.43-0.60), no better than chance. Precision 2%. Many chunks are noisy phone audio and may not hold the whole ayah; labels are sparse |
| 2026-11-21 | Real mistakes, words and vowels | `full/best.pt` | `mdd_d` as above | 22% (19-26); 18% at 0.1% | 19% (7-32) | 8% (0-17) | AUC 0.51 (0.43-0.60). Precision 2% |
| 2026-11-21 | Real mistakes, tajweed judged | `full/best.pt` | `learner_b`: 158 clips of one verse (last 8 words), 941 correct words, 26 word/vowel, 297 tajweed-only; labels from the said transcript | 0.5% (0.1-1.0); 0% at 0.1% | 15% (0-29) | 4% (0-12) | AUC 0.69 (0.59-0.80). Precision 44% at 1% (4 of 9 flags), 100% at 0.1% (1 flag). Only 26 word/vowel mistakes |
| 2026-11-21 | Real mistakes, words and vowels | `full/best.pt` | `learner_b` as above | 0.4% (0.1-0.9); 0% at 0.1% | 8% (0-19) | 4% (0-12) | AUC 0.58 (0.46-0.68). Precision 33% at 1% |
| 2026-11-21 | Speed, this machine (no other load) | `full/best.pt` | 40 phone clips, mean 4.9 s | n/a | n/a | n/a | CPU 258 ms per clip (about 19x real time); GPU backend 42 ms (about 117x); same output as CPU on all 40 clips. 60 s clip timing still to do |
| 2026-11-21 | Reproduce reference numbers on this machine (torch 2.15, training toolkit 3.0, CPU) | `full/best.pt` (step 19,000) | `studio-heldout`, suras 78-114, 400 clips per reciter, one planted mistake each | 1% and 0.1% by construction | 98.3% (97.0-100) | 92.1% (91.7-94.5) | AUC 0.999. 2,798 correct words, 800 planted, 2 reciters. By kind at 1%: letter, vowel, doubling, ghunnah 1.00; madd 0.97; qalqalah 0.84. Same as the run on the GPU notebook service |
| 2026-11-21 | Words-and-vowels mode, thresholds per mode from `studio-heldout` (1,168 clips, 4,158 words) | `full/best.pt` | `phone-labelled-v1`: 25 doubted correct clips (89 words) | tajweed judged 38% (29-49), words and vowels 53% (43-63) at the 1% studio threshold; strong (0.1%) 21% vs 22% | n/a | n/a | 7 control clips: 0% in both, but chosen for decoding well. Mistake clips: 2 of 2 marked words flagged in words-and-vowels mode, 1 of 2 with tajweed judged; correct words in the same clips score higher. Gap is acoustic. Thresholds came from the test reciters; redo from `studio-val` (stage 2.3) |
| 2026-11-20 | Words-and-vowels mode, first run | `full/best.pt` | `phone-labelled-v1`: 25 doubted correct clips | 54% tajweed judged, 56% words and vowels (studio threshold) | n/a | n/a | Different threshold set from the row above |
| 2026-11-20 | First look at phone labels | `full/best.pt` | `phone-labelled-v1`: 25 correct clips (95 display words) + 7 controls | 51% (48 of 95), 19% strong; controls 1 of 26 | n/a | n/a | Studio threshold |
| 2026-11-20 | Full run: all suras, 30,000 steps, about 9 GPU-hours | expert set, all reciters but the three held out | `studio-heldout`, 800 Juz Amma clips, planted | 1% and 0.1% by construction | 98.3% | 92.1% | Token error 0.43% (5,201 segments). Best validation 0.18% at step 19,000; validation swung 0.2-1.5% |
| 2026-11-19 | Pilot: suras 78-114 | expert set, Juz Amma | `studio-heldout`, 800 clips, planted | 1% and 0.1% by construction | 97.0% | 82.5% | Token error 1.2% |
