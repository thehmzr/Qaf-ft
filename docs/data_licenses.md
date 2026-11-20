# Data and model licences

Every dataset and model the project touches, what its licence requires, and how it is used.
Licence fields were read from each Hugging Face page on 2026-11-20. This is a working
register, not legal advice; the full licence text of each source is the authority.

Standing commitments that follow from these licences:

1. The app and every feature powered by the model stay free of charge, with no ads.
2. The model weights are not sold or licensed for a fee.
3. Every source below is credited in the app's About screen and in the model card.
4. No dataset is re-uploaded or redistributed. Training reads from the original location.
5. If a paid feature is ever planned, the model is first retrained without the sources
   marked "free only".

## Used for training

| Source | Licence | What it requires | Use here |
|---|---|---|---|
| nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 (base model) | CC BY 4.0 | Credit; state that it was modified | Starting weights |
| obadx/muaalem-annotated-v3 | MIT (dataset) | Keep the licence notice | Main training set |
| obadx/quran-transcript (text to sound script) | MIT | Keep the licence notice | Label generation |
| tarteel-ai/tlog | CC BY 4.0, access by agreement | Credit; follow the access terms | Phone-audio training, after label filtering |
| FaisaI/tadabur | CC BY-NC 4.0, research and education | Credit; non-commercial only. **Free only** | Reciter variety |
| IqraEval/Iqra_train, Iqra_TTS, Iqra_Extra_IS26 | CC BY-NC 4.0 | Credit; non-commercial only. **Free only** | Mistake examples, second training stage |
| MuazAhmad7/Surah_Ikhlas-Labeled_Dataset | CC BY 4.0 | Credit | Mistake examples |
| Tanzil Quran text (inside quran-transcript) | CC BY 3.0 with terms | Credit and link to tanzil.net; text not altered | Reference text |

## Used for evaluation only (never trained on)

| Source | Licence | What it requires | Use here |
|---|---|---|---|
| IqraEval/QuranMB.v2 and test answers | CC BY-NC 4.0 | Credit; non-commercial. **Free only** | Sound-level detection score |
| obadx/qdat_bench | Apache 2.0 | Keep the notice | Tajweed-rule test |
| Luxbg/quran-muaalem-aligned (unseen reciters split) | MIT | Keep the notice | Held-out reciters |
| RetaSy/quranic_audio_dataset | None stated | No licence granted, so results are reported but the data is not trained on | Learner test |

## On hold: licences that would set the model's own licence

These three licences each require a model trained OR merely evaluated on the data to be
published under that same licence and no other. They exclude one another, so at most one
of them can ever touch the published model. None is used until that choice is made.

| Source | Licence | What using it would bind |
|---|---|---|
| Quran-Lab/quran-tajweed-benchmark, quranic-asr-benchmark | Quran-Lab NPL 1.2 | Model released under NPL 1.2 "with no additional or different terms"; no charge, no ads |
| QUD-Technologies/quranic-universal-ayahs | QUA Dataset License 1.0 | Model published only under the QUA licence; model card and app must credit "QUD Technologies - Qur'anic Universal Audio (https://qud.dev)"; no republishing of derived data; disclose use on request |
| Quran-Lab/QuranTTS | Quran-Lab non-commercial 2.0 | Access request was rejected; not available |

## Not used

| Source | Reason |
|---|---|
| tarteel-ai/everyayah and re-uploads of everyayah.com audio | The recordings' licence is not documented |
| mrmuminov/quran-text_uthmani_openslr-slr132 | No licence stated on the mirror; original page to be checked first |
| obadx/muaalem-model-v3_2 (weights) | Extra conditions in the code repository's licence; used here only to test a design, not shipped |

## Open questions to settle before release

- Tadabur's card adds "for research and educational purposes only", narrower than its CC BY-NC tag.
- The expert set's paper excludes one mushaf (25.0) for poor segmentation, yet it ships in the data; exclude it from training.

- tlog and RetaSy contain voices of private individuals; consent scope for model training is
  not documented on either page.
- The muaalem training set carries an MIT tag, but its audio comes from published
  recitations whose own rights are not listed per recording.
- Licences marked "other" must be read in full; the rows above summarise their dataset pages.
