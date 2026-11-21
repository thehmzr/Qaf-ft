# Status

## Approach
Fine-tune of nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 (115M, CC BY 4.0): encoder kept, new
head (2x upsample + linear), CTC loss, 81 sound tokens from the Quran phonetic script with repeats
folded into count tokens. No language model. Mistakes are found by a separate checker
(`qafft/verify.py`): alignment score per expected token plus an extra-sound check, per word.
Current target: words and vowels. Tajweed judging comes last.

## Where things are
- Everything is on `main` (branch `labeler-tajweed` is merged).
- Trained: pilot (suras 78-114) and full run (all suras, 30,000 steps, about 9 GPU-hours).
- Built: evaluation command, mistake generator, labelling tool (batch queue, suggestion review,
  sound-level annotation, tajweed colouring), phone-clip scoring.
- In progress: scoring more phone clips to choose a phone training set; rescoring labelled clips
  on words and vowels only.
- Next: phone fine-tune with a lower late learning rate and averaged checkpoints.
- Open needs: more labelled real mistakes; a full correct recitation in a new voice.

## Numbers
- Token error on two held-out reciters, all suras: 0.43% (pilot 1.2%).
- Checker, 800 held-out clips with one planted mistake each: AUC 0.999; caught at 5% / 1% / 0.1%
  false flags: 99.9% / 98.3% / 92.1%.
- Phone audio: on 25 hand-labelled correct clips the checker flagged 51% of words at the studio
  threshold; 34% of phone clips match their claimed ayah within 10%.
- No WER or CER: the model outputs sound tokens, not text.

Details and sources: `research_summary.md`.
