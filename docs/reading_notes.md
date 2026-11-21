# Reading notes

What has been studied for this project, what each source changed, and what could not be
reached. The full reading list is in `resources.md`.

## 1. The base model's own paper

"Open Automatic Speech Recognition Models for Classical and Modern Standard Arabic"
(NVIDIA, arXiv 2507.13977). Read in full.

What it says:
- Two models. One for Modern Standard Arabic with punctuation, no diacritics. One unified
  model for Modern Standard and Classical Arabic with punctuation and diacritics. The
  second is the checkpoint this project starts from.
- Architecture: FastConformer with two decoders trained together (transducer and CTC,
  CTC weight 0.3), 1,024 word-piece tokens, dropout 0.1, spectral augmentation with
  4 frequency and 20 time masks, AdamW with cosine schedule, 200 epochs, **the five best
  checkpoints averaged**.
- Training data after their cleaning: MASC 433 h clean + 255 h rescued from the noisy part,
  Common Voice 66 h, FLEURS 5 h, and **EveryAyah 392 h** (Quran, professional reciters).
- Starting weights: they compared training from nothing with starting from English,
  Spanish and multilingual models. Any pretrained start beat training from nothing
  (Common Voice error 17.5% from nothing, about 10.2–10.7% from a pretrained start).
- Cleaning recipe: drop clips with symbols outside the alphabet; drop clips shorter than
  0.1 s or longer than 20 s; drop clips with unusually high or low words per second or
  characters per second; run an earlier model and drop clips it disagrees with strongly
  (word error above 60% or character error above 30%). That last step recovered 255 hours
  from data labelled noisy and lowered error on every test set.
- EveryAyah's official test split shared texts with training. After removing test clips
  whose text also appears in training, the test set shrank from about 104 h to 23 h and
  their word error went from 1.34% to 5.73%. Whisper went from 9.4% to 17.2%.

What it changes for this project:

1. **Two of our three held-out reciters were probably heard by the base model.** The
   base model trained on EveryAyah. Our test reciter 13.0 and our validation reciter 19.0
   both have recitations in EveryAyah; reciter 5.0 does not appear there. They were never
   in our own training, but the encoder we started from has likely heard those voices.
   Consequences: report results per reciter; treat reciter 5.0 as the cleaner test; add
   voices that are in no public recitation collection (the friend's recitation, phone
   users).
2. **Memorised text is a real effect, and our planted-mistake test cannot see it.** Every
   ayah's text is in our training data, read by other reciters. Their 1.34% to 5.73% jump
   shows how much a model gains from having seen the text. A model can lean on that and
   hear the familiar ayah even when the reciter slips. Our planted-mistake test changes the
   expected text and leaves the audio correct, so it cannot detect this. Only mistakes in
   the audio can: spliced recordings and real labelled mistakes. That evaluation should
   move earlier in the plan, before more training.
3. **Train on some non-Quran Arabic too.** Speech whose text the model cannot have
   memorised forces it to listen. The paper's own sets (MASC, Common Voice, FLEURS) are
   candidates, with labels converted to sound tokens. Not yet in the plan; add it to the
   phone fine-tune as a small share and measure.
4. **Their filter supports ours, and adds two cheap checks.** Words per second and
   characters per second catch cut-off and padded clips without any model. Add both to
   `phone_filter.py`. Half of our first labelling batch was that kind of clip.
5. **Checkpoint averaging is what the authors did.** It is already planned for the next run.
6. **Licence note.** The checkpoint is released under CC BY 4.0, and the paper's table
   lists EveryAyah as MIT, while the dataset's own page now says the recordings' licensing
   is not documented. We use the released model under its licence and do not train on that
   audio ourselves; the model's history is recorded here so it is not a surprise later.

## 2. Sources from the user's list: relevance

| Source | Verdict | Why |
|---|---|---|
| Speech and Language Processing (Jurafsky and Martin), phonetics and ASR chapters | Core | Best first explanation of phonetics, CTC and alignment |
| A Course in Phonetics (Ladefoged and Johnson) | Core | Makharij and sifat are articulatory phonetics |
| Hugging Face Audio Course | Core, with a caveat | Good on audio basics and data handling; its recipe produces text with Whisper, which is the approach this project avoids |
| Stanford CS224S | Core | Covers CTC, alignment and assessment |
| Audio Signal Processing for ML (video series) | Core | Fast route to spectrograms and mel features |
| NeMo tutorials | Core | The toolkit in use |
| Montreal Forced Aligner | Useful | Reference aligner for checking our timings |
| SpeechBrain, ESPnet tutorials | Optional | Alternative toolkits; read one recipe for comparison |
| CTC paper (Graves and others, 2006) | Core | The training method |
| Conformer, wav2vec 2.0, HuBERT, Whisper papers | Core for the first, background for the rest | Conformer is our architecture family |
| Goodness of Pronunciation (Witt and Young, 2000) | Core | The idea behind the checker |
| L2-ARCTIC and speechocean762 papers | Useful | Show how mispronunciation data is annotated and scored |
| Theory and Applications of Digital Speech Processing; Spoken Language Processing; Automatic Speech Recognition: A Deep Learning Approach | Reference | Consult by chapter, not cover to cover |
| Online DSP books (Stanford CCRMA) | Reference | For the Fourier transform when needed |
| Introduction to Arabic NLP (Habash) | Useful, selected chapters | Orthography and diacritics |
| The Phonology and Morphology of Arabic (Watson) | Useful | The sound system |
| NVIDIA Arabic models paper (2507.13977) | Core | Our base model; see section 1 |
| Open Universal Arabic ASR Leaderboard | Background | Confirms no general model fits; already used |
| MGB-2, Common Voice, MASC, QASR | Low for now | Become relevant for the non-Quran share in training |
| Quran pronunciation error paper (2509.00094) | Core | Source of the sound script and training set |
| IqraEval / QuranMB | Core for method | Data excluded by licence; the recipe is adopted |
| QDAT and the tajweed classification paper | Later | Tajweed mode only |
| Learner recitation dataset (RetaSy) | Core for testing | Only real learner set with judgments; no licence, so not trained on |
| Tadabur | Excluded from training | Non-commercial; its survey of datasets is still worth reading |
| Quran-Lab benchmark | Excluded | Licence would bind the model; its warning about studio results is right |
| Crowd-sourced user recordings (the 25,000-clip set) | Reference | Superseded for us by the larger phone set |
| Tajweed Rules of the Qur'an; Tuhfat al-Atfal; al-Muqaddimah al-Jazariyyah | Core for the domain | Define what the labels mean |
| Real-time recitation tracker repository | Useful later | Ideas for live following |
| quran-timing-helper | Useful | Another approach to word timings |
| Streaming phoneme recogniser model card (Quran-Lab) | Read for its findings | Weights excluded by licence; its notes on durations and children are useful |
| Arabic text-to-speech models and data (SILMA, Habibi, ClArTTS, the QCRI paper) | Not relevant | This project builds a listener. Speech synthesis was considered for generating mistakes and set aside in favour of splicing real audio |
| Diacritisers (CATT, Sadeed) | Not relevant now | The Quran text is already fully diacritised; needed only if non-Quran text is added |

## 3. What has been read, and how

| Source | Read |
|---|---|
| NVIDIA Arabic models paper | Full text, this session |
| Quran pronunciation error paper; QuranMB; IQRA 2026 overview; the 2026 winning system; AraS2P; CR-CTC; learner dataset paper | Full text, by earlier research passes |
| About 60 further papers (see `research_summary.md`) | Abstract level |
| Model and dataset pages of every source in `datasets.md` | Read at the source |
| Statistical learning textbook | All chapters, notes in `docs/notes/` |
| Recitation tracker, timing helper, phonemiser repositories | README level |

## 4. Not read, and why

| Source | Reason | Who can get it |
|---|---|---|
| CTC paper (2006) | Not on arXiv; its host is not reachable from the cloud machine | Free PDF from the authors' pages |
| Goodness of Pronunciation (2000) | Journal article behind a paywall | Library or author copy |
| Jurafsky and Martin chapters | Host not reachable from the cloud machine | Free online |
| Phonetics, Arabic phonology, Arabic NLP and speech processing textbooks | Paid books | Purchase or library |
| Tajweed textbooks and the two classical poems | Not available from a reachable host | Publisher sites; widely printed |
| Course videos | Video cannot be read here | Free online |

108 research papers that are free on arXiv have been downloaded into
`data/papers/` (not in git).
