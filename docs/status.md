# Status

## Approach
Fine-tune of base, the pretrained 115M FastConformer Arabic model: encoder kept, new
head (2x upsample + linear), CTC loss, 81 sound tokens from the Quran phonetic script with repeats
folded into count tokens. No language model. Mistakes are found by a separate checker
(`qafft/verify.py`): alignment score per expected token plus an extra-sound check, per word.
Current target: words and vowels. Tajweed judging comes last.

## Where things are
- Everything up to the words-and-vowels check is on `main`; the phone stage is on branch `phone-finetune`.
- Trained: pilot (suras 78-114) and full run (all suras, 30,000 steps, about 9 GPU-hours).
- Built: evaluation command, mistake generator, labelling tool (batch queue, suggestion review,
  sound-level annotation, tajweed colouring), phone-clip scoring.
- Setup: `scripts/setup.sh` with pinned versions; reference numbers reproduce on a new machine.
- Words-and-vowels mode measured: it does not lower the flags on phone clips; the gap is acoustic.
- In progress: scoring all 79 phone files to choose a phone training set.
- Next: phone fine-tune (code ready): from `full/best.pt`, expert and phone batches mixed, lr 3e-5
  to zero, last checkpoints averaged.
- Open needs: more labelled real mistakes; a full correct recitation in a new voice.

## Numbers
- Token error on two held-out reciters, all suras: 0.43% (pilot 1.2%).
- Checker, 800 held-out clips with one planted mistake each: AUC 0.999; caught at 5% / 1% / 0.1%
  false flags: 99.9% / 98.3% / 92.1%.
- Phone audio: on 25 hand-labelled doubted correct clips the checker flags 38-56% of words
  at the studio threshold, with or without tajweed judged; about a third of scored phone clips
  match their claimed ayah within 5% on words and vowels.
- No WER or CER: the model outputs sound tokens, not text.

Every experiment: `results.md`. Details and sources: `research_summary.md`.
