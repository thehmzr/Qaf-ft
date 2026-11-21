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
| 2026-11-21 | Best of legal reading choices per word (free madd lengths, ghunnah of noon into lam/ra, seen/saad; `qafft/variants.py`) | `full/best.pt` | `studio-heldout` shard, 150 clips from sura 78, planted | 1% and 0.1% by construction | 99.3% (default 99.3%) | 98.7% (default 98.7%) | No cost on planted mistakes; madd catch 96% in both. One reciter, so no interval |
| 2026-11-21 | Best of legal reading choices per word | `full/best.pt` | `phone-labelled-v1`: 32 correct clips, 121 display words | 6.6% both (long-recitation 1% threshold), 0.8% both (0.1%) | n/a | n/a | Best-of lowered only 2 of 121 word scores; no flag removed |
| 2026-11-21 | Best of legal reading choices per word | `full/best.pt` | `long-recitation-1` (sura 2), test half 2,936 words, thresholds from first half | default threshold kept: 0.85% -> 0.37% (1%), 0.03% -> 0.03% (0.1%); own-half thresholds: 1% threshold drops 9.16 -> 5.82 | n/a | n/a | Lowest score came from the default reading for 94% of words, short munfasil for 4%, short aarid for 2% |
| 2026-11-21 | Speed, this machine (no other load) | `full/best.pt` | 40 phone clips, mean 4.9 s | n/a | n/a | n/a | CPU 258 ms per clip (about 19x real time); GPU backend 42 ms (about 117x); same output as CPU on all 40 clips. 60 s clip timing still to do |
| 2026-11-21 | Reproduce reference numbers on this machine (torch 2.15, training toolkit 3.0, CPU) | `full/best.pt` (step 19,000) | `studio-heldout`, suras 78-114, 400 clips per reciter, one planted mistake each | 1% and 0.1% by construction | 98.3% (97.0-100) | 92.1% (91.7-94.5) | AUC 0.999. 2,798 correct words, 800 planted, 2 reciters. By kind at 1%: letter, vowel, doubling, ghunnah 1.00; madd 0.97; qalqalah 0.84. Same as the run on the GPU notebook service |
| 2026-11-21 | Words-and-vowels mode, thresholds per mode from `studio-heldout` (1,168 clips, 4,158 words) | `full/best.pt` | `phone-labelled-v1`: 25 doubted correct clips (89 words) | tajweed judged 38% (29-49), words and vowels 53% (43-63) at the 1% studio threshold; strong (0.1%) 21% vs 22% | n/a | n/a | 7 control clips: 0% in both, but chosen for decoding well. Mistake clips: 2 of 2 marked words flagged in words-and-vowels mode, 1 of 2 with tajweed judged; correct words in the same clips score higher. Gap is acoustic. Thresholds came from the test reciters; redo from `studio-val` (stage 2.3) |
| 2026-11-20 | Words-and-vowels mode, first run | `full/best.pt` | `phone-labelled-v1`: 25 doubted correct clips | 54% tajweed judged, 56% words and vowels (studio threshold) | n/a | n/a | Different threshold set from the row above |
| 2026-11-20 | First look at phone labels | `full/best.pt` | `phone-labelled-v1`: 25 correct clips (95 display words) + 7 controls | 51% (48 of 95), 19% strong; controls 1 of 26 | n/a | n/a | Studio threshold |
| 2026-11-20 | Full run: all suras, 30,000 steps, about 9 GPU-hours | expert set, all reciters but the three held out | `studio-heldout`, 800 Juz Amma clips, planted | 1% and 0.1% by construction | 98.3% | 92.1% | Token error 0.43% (5,201 segments). Best validation 0.18% at step 19,000; validation swung 0.2-1.5% |
| 2026-11-19 | Pilot: suras 78-114 | expert set, Juz Amma | `studio-heldout`, 800 clips, planted | 1% and 0.1% by construction | 97.0% | 82.5% | Token error 1.2% |
