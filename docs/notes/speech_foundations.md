# Speech foundations: study notes (for Qaf-ft)

Sources read for these notes, with the parts covered:

1. *Speech and Language Processing*, 3rd ed. draft (dated 19 August 2026),
   https://web.stanford.edu/~jurafsky/slp3/ . Ch. 15 "Phonetics and Speech Feature
   Extraction" (15.pdf) and Ch. 16 "Automatic Speech Recognition" (16.pdf), read in
   full. Ch. 17 (TTS) skipped.
2. The CCRMA online books, https://ccrma.stanford.edu/~jos/ : *Spectral Audio Signal
   Processing* (sasp/), *Mathematics of the DFT* (mdft/), *Digital Audio Resampling*
   (resample/). Only the pages on STFT, windows, spectrograms, resolution, aliasing,
   auditory scales and resampling.
3. Hugging Face Audio Course, https://huggingface.co/learn/audio-course : Unit 1
   (audio data, preprocessing), Unit 3 (CTC, seq2seq), Unit 5 (choosing a dataset,
   models, evaluation, fine-tuning).
4. Stanford CS224S *Spoken Language Processing*, Spring 2025 (latest public
   offering), https://web.stanford.edu/class/cs224s/ : syllabus and slides for
   lectures 2, 9, 10, 11 and 14.
5. *Audio Signal Processing for Machine Learning* video series, using the companion
   repository https://github.com/musikalkemist/AudioSignalProcessingForML (lesson
   list plus slide decks 6, 7, 15, 17, 19, 21). The videos themselves were not watched.
6. Toolkit docs: NVIDIA NeMo ASR docs (models, configs, fine-tuning, NeMo Forced
   Aligner) plus NeMo source and tutorials on GitHub; SpeechBrain "Speech
   Recognition From Scratch"; ESPnet2 tutorial and the `asr.sh` template; Montreal
   Forced Aligner (MFA) docs.

Conventions

- Section numbers and URLs point to where a point comes from.
- **(inference)** marks anything that goes beyond what the source says, including
  every application to Qaf-ft and every calculation done here.
- Tajweed facts are domain assumptions to check against the project's own tajweed
  specification. None of the sources mentions Arabic recitation.
- People are named by role only.

---

## 1. Speech and Language Processing (3rd ed. draft)

### Ch. 15 Phonetics and Speech Feature Extraction

**15.1-15.2 Phones and articulation**

- Pronunciation is written as phones (IPA, or ARPAbet for English). TIMIT's
  time-aligned labels were made by automatic alignment and then hand-corrected
  boundaries, the classic forced-alignment workflow.
- Nasals lower the velum and use the nasal tract as a second resonator. Stops have a
  closure, then a release burst. Fricatives are turbulent noise.

**15.3 Prosody.** Prominence is carried by loudness, longer duration and F0
movement.

**15.4 Acoustic phonetics and signals**

- Sampling: at least two samples per cycle are needed. The Nyquist frequency is half
  the sample rate. Most speech information lies below 10 kHz. The telephone network
  passes only frequencies below 4 kHz, so 8 kHz sampling is enough for telephone
  speech, and 16 kHz is common for microphone speech (15.4.2).
- Telephone audio often uses 8-bit mu-law, a log compression with mu = 255
  (Eq. 15.5).
- F0 is the vocal-fold rate. Pitch trackers count pulses in voiced regions and fail
  where there is no voicing. Autocorrelation pitch extraction picks the lag with the
  highest self-correlation.
- RMS amplitude, power, and intensity in dB (Eqs. 15.6-15.8). Vowels show intensity
  peaks.
- Mel scale: m = 1127 ln(1 + f/700) (Eq. 15.9).
- Reading a waveform: vowels are voiced, long and loud. Stops show near-silence then
  a burst. Fricatives look noisy (15.4.4).
- Formants are vocal-tract resonances. F1/F2 identify vowels ([ae] around
  930/1860 Hz, [iy] around 540/2581 Hz). Formants also identify nasals and liquids
  (15.4.5). Source-filter model: the vocal tract shapes the glottal harmonics
  (15.4.6).

**15.5 Log mel spectrum pipeline**

- If test audio is 8 kHz (telephone), training audio must be downsampled to 8 kHz.
  When pooling corpora that include telephone speech, all wideband corpora are
  downsampled to 8 kHz. The chapter says different sampling rates "can't" be combined
  for training and testing (15.5.1). Compare NeMo's advice in section 6.1, which
  differs.
- Windowing: speech is non-stationary, but roughly stationary inside a short window.
  Parameters are window size, stride and shape. Example: 25 ms window, 10 ms stride.
  The rectangular window's hard edges cause problems for Fourier analysis, so the
  Hamming window, 0.54 - 0.46 cos(2 pi n / L), is preferred (15.5.2).
- DFT/FFT (15.5.3). Mel filter bank: triangular filters spaced on the mel scale, fine
  at low frequencies and coarse at high ones. Low frequencies (formants) matter most
  for vowels and nasals. High frequencies (stop bursts, fricative noise) are "less
  crucial" (15.5.4).
- The log of each mel channel reduces sensitivity to level changes, such as the
  mouth moving closer to or further from the microphone. Typical sizes are 80 or 128
  channels. Features are then rescaled, for example to [-1, 1] with zero mean over the
  pretraining set.

**15.6 MFCC.** 39 dimensions (12 cepstral coefficients plus energy, with deltas and
double deltas). The cepstrum separates the source (F0 peak) from the filter (low
quefrencies).

### Ch. 16 Automatic Speech Recognition

**16.1 Task dimensions.** Difficulty varies with vocabulary, speaking style, channel
and noise, and accent or speaker class. Rough WERs (Fig. 16.1): LibriSpeech clean
1.4, Switchboard 5.8, CALLHOME 11, CHiME-6 25.5.

**16.2 Convolution front ends.** A Whisper kernel of 3 frames covers about 45 ms,
enough for formant transitions and stop closures. Stride greater than 1 shortens the
sequence, which narrows the length gap between 10 ms frames and output symbols.

**16.3 Encoder-decoder (AED)**

- An English word averages about 5 letters or 1.3 BPE tokens and lasts about 250 ms
  (25 frames). So 10 ms frames outnumber letters about 5x and tokens about 19x. This
  is why speech models subsample first (16.3.1).
- Whisper: 128 log-mel channels, 25 ms window, 10 ms stride, normalized to [-1, 1];
  30 s = 3000 frames, halved by a stride-2 convolution to one frame per 20 ms.
  HuBERT: convolutions on raw 16 kHz audio, 320x downsampling to 20 ms frames.
- The decoder is a conditional language model, improved by n-best rescoring with an
  external LM and a length normalization (Eq. 16.8).

**16.4 Self-supervised models (HuBERT).** Masked prediction of k-means units. For
fine-tuning, a linear CTC head is added and the CNN layers are frozen.

**16.5 CTC**

- The model emits one symbol per frame, then collapses repeats and removes blanks.
  The blank lets the output contain double letters ("dinner" needs a blank between
  the two n's) and covers silence.
- Strong conditional-independence assumption: the output at time t is independent of
  other outputs given the input. Greedy decoding is the per-frame argmax.
- The most probable alignment is not necessarily the most probable string. A beam
  search that sums alignments of the same prefix approximates the string probability.
- Because of the independence assumption, CTC "does not implicitly learn a language
  model". The chapter calls interpolating an external LM with a length term
  "essential" (Eq. 16.19).
- Training sums over all alignments with the forward-backward algorithm (16.5.2).
- Joint CTC + attention training uses a weighted loss, lambda for attention and
  1 - lambda for CTC (Eq. 16.22).
- Streaming (16.5.4): CTC can emit left to right immediately. Attention needs the
  whole input. RNN-T adds a predictor network conditioned on previous non-blank
  outputs and removes the independence assumption.

**16.6 Evaluation**

- WER = 100 x (I + S + D) / N reference words, using the minimum edit distance. It
  can exceed 100%. Worked example: 6 S, 3 I, 1 D over 13 words = 76.9%.
- NIST sclite produces confusion matrices, per-speaker rates and sentence error rate.
- Normalize text before scoring.
- Significance: use MAPSSWE (Matched-Pair Sentence Segment Word Error). Segments are
  bounded by words both systems get right. Take per-segment error differences
  Z_i = N_A - N_B and test whether their mean is 0. The normal approximation needs
  n > 50 segments. McNemar's test is inappropriate because errors on neighbouring
  words are dependent.
- WER weights all words equally. A better weighting is desirable but no agreed
  alternative exists.

---

## 2. CCRMA online books (DFT, STFT, windows, filters, resampling)

### Sampling and aliasing

- Sampling theorem (*Mathematics of the DFT*, "Sampling Theorem",
  https://ccrma.stanford.edu/~jos/mdft/Sampling_Theorem.html): a signal can be
  rebuilt exactly from its samples if its spectrum is zero for |f| >= fs/2. The
  reconstruction kernel is sinc(t/Ts). Energy above fs/2 folds back as aliasing:
  shifted spectral copies overlap and add.
- Downsampling theorem (*Spectral Audio Signal Processing* (SASP), "Downsampling
  Aliasing"): keeping every M-th sample turns the spectrum into (1/M) times the sum of
  M shifted copies. Before decimating, lowpass to (-pi/M, pi/M).

### STFT and spectrograms

- STFT = FFTs of windowed frames that hop through time. Window length M (often odd),
  FFT size N >= M (often a power of 2, zero-padded), hop R. Sampling at N points is
  lossless if N exceeds the window's nonzero width (SASP "Practical Computation of the
  STFT").
- Classic spectrogram = log-scale (dB) STFT magnitude. M sets frequency resolution.
  Window type trades side-lobe suppression against resolution. N sets spectral
  interpolation. R sets time oversampling. Overlap is typically at least 50%. For a
  periodic Hamming window, R = (M-1)/2 is the largest hop that still allows robust
  perfect reconstruction (SASP "Classic Spectrograms").
- Speech spectrogram example: 20 ms Hamming window, FFT zero-padded to the next power
  of 2 above 4M. The 20 ms choice is short enough to usually hold one phoneme and long
  enough for two periods of a 100 Hz pitch. Long windows resolve harmonics. Short
  windows track pitch and formant changes. Telephone band is nominally 200-3200 Hz,
  usually with three or four formants present (SASP "Spectrogram of Speech").
- Auditory spectrograms: the STFT is a uniform filter bank with one integration time.
  The ear is non-uniform, with wider bands and finer time resolution at high
  frequencies. dB tracks loudness well at high levels but not at low levels (SASP
  "Audio Spectrograms").

### Windows and resolution

- Main-lobe width in multiples of fs/M (SASP "Resolving Sinusoids"): rectangular 2,
  Hann and Hamming 4, Blackman 6. Hamming: peak side lobe about -42.76 dB, 10 dB
  better than Hann, but only -6 dB/octave roll-off because it does not taper to zero
  (SASP "Hamming Window").
- Bark scale (SASP "The Bark Frequency Scale"): 24 critical bands. About 17-18 of
  them lie below 4 kHz and 21-22 below 8 kHz. There is no mel page in SASP.

### Resampling (*Digital Audio Resampling*, https://ccrma.stanford.edu/~jos/resample/)

- Bandlimited interpolation computes signal values at arbitrary times from samples,
  assuming the signal was bandlimited to fs/2 ("Introduction").
- The classic method is to upsample by L, lowpass, then downsample by M. Its filter
  cutoff depends on max(L, M). The page's own method uses one stored lowpass filter
  for any ratio ("Introduction").
- The ideal sinc must be windowed. Hard truncation at the 5th zero crossing gives only
  about 20 dB stopband rejection. A Kaiser taper to the same point gives about 80 dB.
  The Kaiser beta trades stopband attenuation against transition width ("From Theory
  to Practice").
- When downsampling, the filter is stretched to lower its cutoff "to avoid aliasing",
  and more input is needed (about Nz*Fs/Fs' extra samples). Raising Nz improves
  aliasing rejection but places the transition band inside the signal band. For a
  given stopband spec (example: 80 dB), halving the transition width roughly doubles
  the multiply-adds. Example design: Nz = 13, L = 512 table entries per zero crossing,
  27 x 512 = 13,824 taps ("Implementation").

---

## 3. Hugging Face Audio Course

**Unit 1, audio data**
(https://huggingface.co/learn/audio-course/chapter1/audio_data)

- Speech sits below 8 kHz, so 16 kHz sampling is enough. 8 kHz loses high frequencies
  and sounds muffled.
- All examples should share one rate, and fine-tuning data must match the
  pretraining rate.
- Mel spectrogram example: n_mels = 128, fmax = 8000 (40 and 80 are common).
- Caveats: there are two mel scales ("htk" and "slaney"). Some pipelines use
  amplitude instead of power, and some "dB" values are plain logs. Match whatever the
  pretrained model expects.

**Unit 1, preprocessing**
(https://huggingface.co/learn/audio-course/chapter1/preprocessing)

- Upsampling 8 to 16 kHz only estimates values between existing samples, because
  there is no content above 4 kHz.
- Downsampling must filter first. Dropping every other sample causes aliasing. Use
  well-tested libraries.
- Filter clips longer than 20 s to avoid out-of-memory errors.

**Unit 3, CTC** (https://huggingface.co/learn/audio-course/chapter3/ctc)

- Wav2Vec2's CNN gives one hidden state per 20 ms with a 25 ms receptive field: 50
  frames per second, 768 dimensions each.
- CTC works best with small vocabularies, generally under 50 tokens, which is why it
  predicts characters.
- In Wav2Vec2, the blank is the same token as `<pad>`.
- Collapse example: `_ER_RRR_ORR` becomes ERROR. Naive deduplication would give EROR.
- CTC can produce phonetically plausible misspellings. An external LM acts as a
  spellchecker.

**Unit 3, seq2seq.** The decoder is a causal LM. It can repeat or skip words and
decodes more slowly than CTC.

**Unit 5, ASR models**
(https://huggingface.co/learn/audio-course/chapter5/asr_models)

- CTC models are "acoustic-only" and prone to phonetic spelling errors (CHRISTMAUS,
  ROSE for ROAST). They can be fine-tuned with as little as 10 minutes of labelled
  speech.
- Long-form audio is chunked with small overlaps that are stitched
  (chunk_length_s = 30). Segment timestamps come from return_timestamps=True.

**Unit 5, choosing a dataset.** Advice: "match our domain to the conditions we
anticipate at inference time". A model trained on clean audiobooks may struggle in
noise.

**Unit 5, evaluation**
(https://huggingface.co/learn/audio-course/chapter5/evaluation)

- WER = (S + I + D) / N. CER is the character-level version and gives partial credit.
- WER can exceed 100%. A value above 100% usually means something is broken.
- Train on orthographic text, but normalize both references and predictions before
  scoring. Drop references that are empty after normalization.
- Dhivehi baseline: orthographic WER 167.3%, normalized WER 125.7%.

**Unit 5, fine-tuning**
(https://huggingface.co/learn/audio-course/chapter5/fine-tuning)

- Whisper-small on about 7 h of Dhivehi, resampled from 48 to 16 kHz. Batch 16,
  lr 1e-5 constant with 50 warmup steps, 500 steps, best checkpoint chosen by WER.
- Result: normalized WER 14.07% (orthographic 63.90%).

---

## 4. Stanford CS224S Spoken Language Processing (Spring 2025)

Syllabus: https://web.stanford.edu/class/cs224s/semesters/2025-spring/syllabus .
Slides: `.../lecture-slides/224s.25.lecN.pdf`.

**Lec 2, phonetics.** Deep systems learn letter-sound mappings from data better than
hand-engineered phonetics. Phonetics still helps describe and debug errors, such as
how an accent changes sounds.

**Lec 9, ASR introduction.**

- Goals named: verbatim vs clean transcripts; real-time operation with "end-to-end
  latency under 200 ms for natural dialogue flow"; on-device constraints.
- Noisy channel model: argmax_W P(O|W) P(W).
- WER can exceed 100% and treats function and content words alike. Align systems
  against each other for error analysis.
- Typical MFCC front end: 25 ms window, 10 ms shift, pre-emphasis 0.97, 39
  dimensions. "Modern models don't rely heavily on MFCCs."
- HMM-GMM: triphones with decision-tree state tying.
- Forced alignment is the Viterbi path constrained to the known word string. It is
  used iteratively: align, refit, add complexity (monophone, then subphone, then
  triphone, then tied states). Flat start initializes to global means. Starting with
  short or easy utterances helps.

**Lec 10, neural ASR and CTC.**

- CTC assumes input length T >= output length.
- Real CTC output without a lexicon or LM (WSJ) shows phonetic misspellings
  ("REHBILITATION CRU", "LOOGGING"). Switchboard was about 19% CER.
- "Comparing alignments" and "Learning phonemes and timing" slides: phone segments
  from an HMM-GMM aligner are overlaid on average CTC character probabilities. CTC
  character activations are sharp, short events against a background of blank, not
  segment-shaped like HMM states.
- CTC training figure from the original CTC paper: early in training the error is
  driven by the targets alone. Later the network "strongly predicts" the labels as
  spikes, with blank (dashed line) dominating elsewhere.
- Beam search over prefixes. Lexicon-free CTC with a character LM handles OOV words.
- RNN-T: the predictor conditions on previous non-blank tokens. Each token is emitted
  once.
- Joint CTC + attention (λ = 0.2 example) learns alignments faster and is more robust
  in noise (CHiME-4).
- SpecAugment: time warping, frequency masking, time masking (Switchboard 300 h
  table).

**Lec 11, state-of-the-art ASR.**

- Conformer = transformer encoder plus a convolution module. In the ablations, the
  convolution module has the largest effect.
- Whisper long-form: 30 s windows shifted by timestamp tokens, with a `<|nospeech|>`
  tag as voice activity detection. Failure mode: endless repetition.
- Choosing a solution: with tens of hours, fine-tune. A fine-tuned small model can
  beat a larger one out of the box. Whisper tiny runs on modern phones. For
  monolingual ASR, Conformer-type models may beat Whisper.

**Lec 14, beyond English.**

- MMS uses per-language adapters (about 2M parameters each) plus per-language CTC
  heads.
- MMS-lab aligns long chapters with forced alignment and filters out low-quality
  samples.
- Quechua: Whisper out of the box gives 117-326% WER, and about 18-20% after
  fine-tuning.

---

## 5. Audio Signal Processing for Machine Learning (video series)

Companion repository: https://github.com/musikalkemist/AudioSignalProcessingForML.
The series has 23 lessons, from waveforms and loudness through time-domain features,
Fourier transform, DFT, STFT, mel spectrograms and MFCCs to spectral features.

- **Feature extraction pipeline (lesson 6):** framing, windowing, transform,
  per-frame feature, then aggregation. Features use frames because a single sample is
  far shorter than the ear's roughly 10 ms resolution. Discontinuous frame edges cause
  spectral leakage. A Hann window fixes the edges but attenuates them, so frames
  overlap.
- **Time-domain features (lesson 7):** amplitude envelope (per-frame max, sensitive to
  outliers), RMS energy (loudness indicator, less sensitive to outliers, used for
  segmentation), zero-crossing rate.
- **STFT (lesson 15):** output shape is (frame/2 + 1 bins, (len - frame)/hop + 1
  frames). Example: 10,000 samples, frame 1000, hop 500 gives (501, 19). Typical hop
  is 1/2, 1/4 or 1/8 of the frame. Longer frames give better frequency resolution and
  worse time resolution.
- **Mel spectrogram (lesson 17):** STFT, then dB, then mel filters. The filter bank is
  built by spacing points evenly in mel, converting back to Hz, rounding to FFT bins
  and making triangles. Filter matrix shape is (bands, frame/2 + 1). Band count (40,
  60, 90, 128) "depends on the problem".
- **MFCC (lesson 19):** log-amplitude spectrum, mel scaling, then DCT. The DCT gives
  real values, decorrelates bands and compresses. Traditionally 12-13 coefficients
  plus Δ and ΔΔ = 39. Downsides: "not robust to noise", heavy knowledge engineering.
- **Frequency-domain features (lesson 21):** band energy ratio (power below vs above a
  split frequency), spectral centroid (brightness), bandwidth (spread around the
  centroid).

---

## 6. Toolkit recipes

### 6.1 NVIDIA NeMo ASR

**Models**
(https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)

- FastConformer: 8x depthwise-convolution subsampling with 256 channels, kernel size
  9, about 2.4x faster than Conformer. 128 channels are 2.7x faster but lose quality.
  Local attention allows more than 1 hour of audio. Longformer-style attention allows
  up to 70 min. Uses a cosine-annealing scheduler.
- Timestamps: for FastConformer, multiply frame offsets by 8 x window_stride, i.e.
  80 ms per output frame (asr/intro.html).
- Cache-aware streaming Conformer:
  - Fully causal mode: `att_context_size=[left,0]` with `conv_context_size=causal`.
    Zero look-ahead, lower accuracy.
  - Regular look-ahead accumulates across layers. Example: 17 layers x 2 frames x
    10 ms x 4 = 1360 ms.
  - Chunk-aware look-ahead (recommended): causal convolutions, attention sees its own
    chunk.
  - About 6 s of left attention context approaches unlimited-context accuracy.
  - Several look-aheads can be trained at once:
    `att_context_size=[[70,13],[70,6],[70,1],[70,0]]` with `att_context_probs`, then
    switched at inference.
  - Export with `cache_support`.
- Hybrid Transducer-CTC: shared encoder with both heads. Switch with
  `change_decoding_strategy(decoder_type='ctc')`. Export a CTC-only graph with
  `set_export_config({'decoder_type':'ctc'})`.

**Configs and source defaults** (configs.html; `audio_preprocessing.py`;
`examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml`)

- Preprocessor class defaults: 16 kHz, window 0.02 s, stride 0.01 s, Hann,
  `normalize: per_feature`, preemph 0.97, 64 features, log with zero guard 2^-24,
  dither 1e-5, mag_power 2, slaney mel scale and norm, highfreq = Nyquist.
- **Narrowband augmentation built into the preprocessor:** `nb_augmentation_prob`
  (default 0.0) and `nb_max_freq` (default 4000). In training, for the chosen share
  of the batch, all STFT bins above `nb_max_freq` are zeroed before the mel filters
  (`features.py`).
- FastConformer-CTC config: 80 features, 25 ms window, n_fft 512, 18 layers,
  d_model 512, 8 heads. SpecAugment: freq_masks 2, freq_width 27, time_masks 10,
  time_width 0.05 (a fraction). max_duration 16.7 s (set for LibriSpeech). lr 1e-3,
  warmup 15000, min_lr 1e-4.
- Audio augmentors (`perturb.py`):
  - SpeedPerturbation 0.9-1.1 (resamples, so pitch changes too); TimeStretch 0.9-1.1.
  - Gain ±10 dB; Shift ±5 ms; WhiteNoise -90 to -46 dB.
  - Impulse (RIR), NoisePerturbation, RirAndNoise.
  - **TranscodePerturbation** with codecs g711, amr-nb and ogg via sox. g711 =
    8 kHz a-law with `lowpass 3400 highpass 300`; amr-nb at random rate 0-3; ogg at
    quality -1 to 7. Output is resampled back to 16 kHz.
- Tokenizer: `bpe` (SentencePiece) or `wpe`, and an aggregate tokenizer for multiple
  languages.

**Fine-tuning** (`docs/source/asr/fine_tuning.rst`)

- `speech_to_text_finetune.py` with `init_from_pretrained_model` or
  `init_from_nemo_model`. Include/exclude lists allow partial loading.
- With a new tokenizer, exclude the final linear layer (CTC) or decoder/joint
  (Transducer).
- Typical settings: 50-100 epochs for domain adaptation, lr 1e-4 to 1e-5. Replace a
  Noam warmup with a constant or cosine schedule so warmup does not reset to a high
  learning rate. Use SpecAugment when fine-tuning.

**Telephony tutorial** (`tutorials/asr/ASR_for_telephony_speech.ipynb`)

- NeMo's English models were trained on mixed 8 kHz and 16 kHz data: Fisher and
  Switchboard were upsampled to 16 kHz. They therefore handle both narrowband and
  wideband audio.
- At inference all audio is resampled to 16 kHz.
- Recommended for new 8 kHz data: upsample to 16 kHz (offline, because online is
  slow) and fine-tune a 16 kHz model. The alternative, an 8 kHz model, converges
  better when initialized from 16 kHz weights.

**Buffered streaming tutorial** (`tutorials/asr/Streaming_ASR.ipynb`)

- Buffer = chunk plus left and right context. Keep only the tokens from the middle
  chunk. Examples: chunk 1 s in a 3 s buffer, or chunk 4 s with 2 s context on each
  side. A 1 h file at 15 s chunks gives 240 chunks.
- Conformer-CTC stride 4 gives one token per 40 ms (Citrinet stride 8).
- Caveat in the tutorial: it assumes output tokens are well aligned with their audio,
  which "may not always be true for models trained with CTC loss".

**NeMo Forced Aligner** (tools/nemo_forced_aligner.html)

- CTC log-probabilities plus Viterbi give token, word and segment timestamps (CTM and
  ASS files). Works with CTC or hybrid models in CTC mode, not pure Transducer.
- Can align against the model's own transcript (`align_using_pred_text`).
- Buffered chunked streaming option (chunk 1.6 s, buffer 4.0 s; CTC with per_feature
  preprocessor only). `use_local_attention` gives [64,64] context for long audio.
- `minimum_timestamp_duration` exists because timestamps can be "very short".
- Evaluate with mean absolute error on start and end times against ground-truth
  CTMs. Better reference text gives better alignment.

### 6.2 SpeechBrain (brief)

"Speech Recognition From Scratch" tutorial
(https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-recognition-from-scratch.html):

- JSON manifests (id, wav, length, words); SentencePiece unigram, 1000 tokens.
- CRDNN encoder plus attentional decoder, joint CTC/attention (ctc_weight 0.5, CTC for
  the first 5 of 15 epochs).
- On-the-fly augmentation: speed 95/100/105%, noise at 0-15 dB SNR, time/frequency
  dropout. Data sorted by length. The page recommends 100-1000 h.

### 6.3 ESPnet2 (brief)

ESPnet2 tutorial (https://espnet.github.io/espnet/espnet2_tutorial.html) and
`egs2/TEMPLATE/asr1/asr.sh`:

- Stages: 1 data prep; 2 speed perturbation ("0.9 1.0 1.1"); 3 feature
  formatting (16k); 4 remove utterances outside 0.1-20 s; 5 token list (bpe or char);
  6-9 LM; 10-11 ASR training; 12 decoding; 13 scoring (sclite WER/CER/TER).
- Encoder subsampling defaults to 4.

### 6.4 Montreal Forced Aligner

Docs: https://montreal-forced-aligner.readthedocs.io/en/latest/

- `mfa align CORPUS DICTIONARY ACOUSTIC_MODEL OUTPUT` writes TextGrids (word and
  phone intervals). `mfa align_one` handles a single file without speaker adaptation.
  `mfa adapt` adapts a pretrained model. `mfa train CORPUS DICTIONARY OUTPUT` trains a
  new model (workflows/alignment.html, train_acoustic_model.html).
- Training regime (configuration/acoustic_modeling.html): HMM-GMM on MFCCs at a
  10 ms frame shift, run as monophone (40 iterations, 1000 Gaussians), then triphone
  (10,000 Gaussians, 1000 leaves), then LDA (40 dimensions), then SAT (fMLLR speaker
  adaptation). Pronunciation probabilities are estimated after the 2nd and 3rd SAT
  blocks. beam 10, retry_beam 40.
- Phone topology: up to 5 states per phone with early exit, so the **minimum phone
  duration is 10 ms** (one frame), vs 30 ms for a standard 3-state HMM.
- Phone groups let related phones, explicitly including "long/short vowels", share
  data in triphone modelling. Phonological rules cover regular variation. A large
  corpus is not needed for decent alignments.
- Single-speaker mode disables speaker adaptation.
- `--fine_tune` (implementations/fine_tune.html): default boundary resolution is
  about 10 ms. Fine-tuning re-analyses each interval's initial boundary with a 1 ms
  step. The page warns that each frame's features span about 25 ms, so the true
  accuracy limit is unclear.
- Evaluation (implementations/alignment_evaluation.html): reference TextGrids with a
  phone-mapping YAML.
  - Pairwise alignment cost = |Δstart| + |Δend| + 2 if the labels differ.
  - Overlap score = mean cost per utterance.
  - **PER = (I + D + 2S) / N** (substitutions count double, unlike standard edit
    distance).
- The "Concepts" section (features, HMMs, speaker adaptation) is marked "under
  construction" and has no content yet.

---

## 7. Application to Qaf-ft

Recommendations are grouped by stage and ordered within each stage by expected
effect on false flags. All of section 7 is **(inference)** unless a source is cited
as saying it.

### 7.1 Front end and phone fine-tune (current stage)

1. **Decide what "8 kHz audio" means and augment for both cases.** SLP 15.4.2 says
   8 kHz *sampling* passes only content below 4 kHz (telephone, mu-law or a-law). The
   current `degrade()` resamples 16k to 8k to 16k, which removes everything above
   4 kHz, although its docstring says "8 kHz band limit". Wideband phone codecs and
   Bluetooth headsets in wideband mode keep content up to about 7-8 kHz (not stated in
   any source here). Keep the 4 kHz case and add a softer case (lowpass around
   6-7 kHz) so the model sees a spread of cutoffs. **(inference)**
2. **Use NeMo's narrowband switch and codec transcoding instead of only a clean
   resample.** `nb_augmentation_prob` zeroes STFT bins above `nb_max_freq` in the
   preprocessor itself. Randomizing `nb_max_freq` (for example 3.4-7 kHz) is a cheap
   code change. `TranscodePerturbation` adds the real distortions of
   g711/amr-nb/ogg (band edges 300-3400 Hz, quantization, codec smearing), which a
   resample does not produce. **(inference)** on the choice of values; the mechanisms
   are documented in NeMo source.
3. **Expect many empty channels on narrowband input.** With slaney mel filters up to
   8 kHz, about 22% of an 80-band filter bank (about 18 channels) lies above 4 kHz
   (computed here from the slaney mel formula). On 8 kHz-sampled audio those channels
   hold only dither (1e-5) and the log floor (2^-24). Per-feature normalization then
   stretches that near-constant noise to unit variance, so the encoder sees inputs it
   rarely saw in studio training. This is a plausible direct cause of false flags on
   phone audio. Ways to address it: train with band-limit augmentation (points 1-2);
   evaluate a variant that clamps or masks top channels when the input is detected as
   narrowband; or fine-tune on mixed-band data as NeMo's English models do (telephony
   tutorial). **(inference)**
4. **Resample on the phone exactly as in training.** CCRMA *Resampling*: filter
   quality varies widely (rectangular truncation about 20 dB vs Kaiser about 80 dB
   stopband), and the transition band sits inside the signal band. Phones record at
   44.1 or 48 kHz. If the app's resampler rolls off differently from the training one
   near 7-8 kHz, the top mel channels shift systematically. Use one resampler
   implementation, with fixed parameters, in data prep, training augmentation and the
   app. Add a test that compares log-mel features produced by the app path and the
   training path on the same file. **(inference)**
5. **Keep the base model's preprocessor settings fixed.** Window, stride, n_fft, mel
   scale, preemphasis, dither and normalization belong to the checkpoint. The HF
   course warns that htk vs slaney and power vs amplitude differences matter.
   Any reimplementation on device (for example, if features are computed outside the
   ONNX graph) must match NeMo's `per_feature`, `preemph 0.97`, `log_zero_guard 2^-24`
   and slaney filters exactly. Test feature parity before quantization.
   **(inference)**
6. **Expect high-frequency letters to suffer most on phones.** SLP 15.5.4: high
   frequencies carry stop bursts and fricative noise. The SASP speech page: telephone
   bands keep only 3-4 formants. Letters told apart mainly by fricative spectra
   (sibilants, interdentals) and qalqalah bursts should lose more evidence under band
   limiting than vowels and nasals (ghunnah) do, since those live in the formant and
   nasal-murmur region. Break down false flags by letter class and by band-limit
   condition before changing thresholds. **(inference)**
7. **Recheck the ban on speed perturbation.** All three toolkits use 0.9-1.1 speed
   perturbation by default. The project avoids it because labels contain madd counts.
   If madd counts are relative to the reciter's own tempo (harakah units), a uniform
   tempo change keeps the ratios, and TimeStretch (pitch kept) at ±5% might be
   label-safe. If counts are tied to absolute durations, keep the ban. This needs a
   decision from the tajweed specification. **(inference)**
8. **SpecAugment time masks and tajweed.** The FastConformer config uses 10 time
   masks of up to 5% of length. A mask can cover most of a short ghunnah or the tail
   of a madd, which makes the length label partly unobservable. Consider fewer or
   narrower time masks in the later phone fine-tune epochs. Frequency masks are
   harmless and also mimic band loss. **(inference)**
9. **Fine-tuning schedule.** NeMo: lr 1e-4 to 1e-5, no Noam warmup reset, SpecAugment
   on. HF: pick the checkpoint by the held-out metric. CS224S lec 11: with tens of
   hours, fine-tune rather than train. The project's 3e-5 encoder / 1e-4 head with
   cosine decay and EMA is consistent with this.

### 7.2 Frame rate vs token rate (affects alignment and madd)

10. **Check the CTC length constraint for fast reciters.** CTC needs T >= U plus one
    blank between each pair of identical neighbouring tokens (SLP 16.5; CS224S lec 10
    says T >= U). FastConformer gives 80 ms frames (12.5/s). The project upsamples to
    40 ms (25/s). Sound tokens (letters plus short vowels plus long vowels) can exceed
    12 per second in fast recitation, so the upsampling was the right call. Log the
    ratio U/T per utterance and flag cases above about 0.7: at that point CTC is
    forced into a near-diagonal path and loses confidence. **(inference)**
11. **Output frame duration limits duration resolution.** At 40 ms frames, a
    one-frame error is 40 ms, comparable to the shortest consonant durations. MFA
    reaches 10 ms frames, or 1 ms boundary search with `--fine_tune`, although its
    page warns that 25 ms analysis windows blur this. For madd counting, measure
    durations relative to the reciter's local tempo (harakah estimates from nearby
    short vowels), not in absolute frames. **(inference)**

### 7.3 CTC peakiness, forced alignment and duration measurement

12. **Do not read durations off CTC frame labels.** CS224S lec 10 shows CTC
    activations as short spikes with blank dominating, unlike HMM segments. NeMo's
    streaming tutorial warns that CTC tokens may not line up with their audio. A long
    vowel can occupy one or two frames in a `forced_align` path while blank fills the
    rest of its real duration. The number of frames assigned to a token therefore
    measures confidence, not length. Use one of these instead:
    - (a) the span from a token's first frame to the next non-blank token's first
      frame, which includes trailing blanks;
    - (b) acoustic measurement inside that span (voicing from autocorrelation pitch,
      intensity per SLP 15.4.3-15.4.4, nasal energy for ghunnah);
    - (c) an HMM aligner (MFA) for durations, as in point 14.

    Validate whichever is chosen against hand-marked boundaries using NFA/MFA-style
    MAE on start and end times. **(inference)**
13. **Keep length counts in the token set.** Encoding madd length as token identity
    (as the 81-token set does) moves the length decision from timing into
    classification, which suits peaky CTC. Expect confusions between adjacent
    counts. Score them as tajweed-level, not word-level, errors. **(inference)**
14. **Use MFA as a duration reference.**
    - Train an MFA acoustic model with a pronunciation dictionary written in Qaf sound
      tokens, with phone groups for long/short vowel pairs (supported, per the MFA
      train page).
    - Use expert recordings with speaker adaptation (SAT/fMLLR), not single-speaker
      mode, when a reciter has several files.
    - Its 5-state, 10 ms-minimum topology and `--fine_tune` give boundaries for
      calibrating or replacing CTC-based madd and ghunnah measurement.
    - MFA's overlap score and PER (with double-weighted substitutions) can serve as
      alignment-quality checks.

    **(inference)**
15. **Forced alignment against the expected text is biased toward the text.** The
    checker aligns the expected tokens, so Viterbi will place a token somewhere even
    if it was not said (NFA notes that alignment quality depends on reference
    accuracy). Always pair the forced path with a free (greedy) decode, or with the
    margin between the forced token's posterior and the best competing non-blank
    token over the same frames. Large margins against the expected token are the
    evidence of a real mistake. Small or negative margins on phone audio are
    candidates for suppression. **(inference)**

### 7.4 Cutting a long recitation into ayahs

16. **Split on pauses, then match.** For pause candidates, use frame RMS energy (series
    lesson 7; SLP intensity) combined with the CTC blank posterior (long blank runs).
    Then confirm each cut by aligning the expected ayah token sequences. This is the
    same idea as NFA's segment-level timestamps and MMS-lab's chapter alignment
    (CS224S lec 14). **(inference)**
17. **Run long audio in overlapping buffers.** HF Unit 5 and NeMo's buffered tutorial:
    process chunks with left and right context and keep only the middle tokens.
    Examples from the sources: chunk 4 s with 2 s context; NFA defaults 1.6 s chunk in
    a 4 s buffer. FastConformer local attention also handles audio longer than an hour
    in one pass. Prefer cuts that land inside detected pauses so no word is split.
    **(inference)** on the pairing with pauses.

### 7.5 Word-level verifier (logistic regression)

18. **Features per word**, all available from the existing checker. **(inference)**
    - mean and minimum log posterior of the expected tokens over their aligned spans;
    - margin to the best competing non-blank token (point 15);
    - free-decode edit distance to the expected tokens for that word;
    - blank share inside the word span;
    - duration relative to local tempo;
    - U/T ratio (point 10);
    - an input-channel indicator (estimated bandwidth: energy above 4 kHz, a band
      energy ratio per series lesson 21), so the verifier can learn that the same
      posterior means less on narrowband audio.
19. **Train and calibrate on phone audio of correct recitation.** The verifier's main
    job is to cut false flags. Its negatives (no mistake) must include many
    phone-recorded correct readings across devices and codecs. HF Unit 5: match
    training data to inference conditions. **(inference)**

### 7.6 Live tracking with chunked or streaming audio

20. **Start with buffered chunked inference on the current model.** It needs no
    retraining. Latency is roughly chunk length plus right context. **(inference)**
    The alternative, cache-aware streaming, needs fine-tuning with chunk-aware
    look-ahead (causal convolutions, limited `att_context_size`). NeMo shows one model
    can be trained with several look-aheads ([70,13] down to [70,0]) and switched at
    run time, and that about 6 s of left context comes close to full context.
    CS224S lec 9 gives under 200 ms as the dialogue target. Recitation tracking can
    probably accept around 0.5-1 s. **(inference)**
21. **Normalization in streaming.** Per-feature normalization over a whole utterance
    is not available live. Use running statistics, or fixed statistics measured on
    phone audio, and test that switching from offline to streaming features does not
    raise false flags by itself. **(inference)**
22. **Tracking needs alignment, not transcription.** Since the expected text is known,
    live tracking can run forced alignment incrementally against the next few words,
    which tolerates CTC spikes better than greedy decoding. This follows from the
    noisy-channel view (CS224S lec 9), where the text prior here is near-certain.
    **(inference)**

### 7.7 ONNX int8 export to phones

23. **Export the CTC path only, and quantize after a feature-parity check.** NeMo
    exports a CTC-only graph from hybrid models (`decoder_type: ctc`) and adds
    `cache_support` for streaming. CS224S lec 11: small models run on phones.
    **(inference)** for the following:
    - Decide whether the mel front end goes inside the graph or in app code (point 5).
    - Measure the int8 effect on the margins and posteriors the checker uses, not only
      on token error rate, because false flags come from small shifts in posterior
      near thresholds.
    - Re-fit verifier thresholds on int8 outputs.

### 7.8 Evaluation

24. **Metrics.**
    - Token error rate on sound tokens (a PER over the 81 tokens), standard
      (I + S + D) / N as in SLP 16.6. Be explicit if MFA's PER with double-weighted
      substitutions is used for alignment checks instead.
    - Word-flag precision and recall, plus the **false-flag rate on correct phone
      recitations**, the project's key number.
    - Report each by channel condition (studio, phone, headset, codec) and by letter
      class.
    - Normalize text consistently before scoring (HF Unit 5). **(inference)** for the
      choice of metrics.
25. **Significance.** Compare two checkpoints on the same clips with a matched-pairs
    test over many short segments (SLP 16.6, MAPSSWE: more than 50 segments, errors
    independent between segments), for example per-ayah differences in flag counts.
    Do not use McNemar-style tests on per-word errors within a phrase, because errors
    on neighbouring words are dependent (SLP 16.6). **(inference)** for the
    application.

### Top priorities (summary)

1. Band-limit and codec augmentation with varied cutoffs (NeMo `nb_augmentation_prob`
   and `TranscodePerturbation`). Fix the 4 kHz vs "8 kHz" ambiguity in `degrade()`.
2. One shared resampler and a test that the app's features match training features.
3. Stop measuring madd from CTC frame counts. Use span-to-next-token or acoustic
   measurement, calibrated against MFA or hand-marked boundaries.
4. Add posterior margins and an input-bandwidth feature to the word verifier, and
   train it on correct phone recitations.
5. Track the false-flag rate on correct phone audio by channel and letter class, with
   matched-pair tests between checkpoints.

### Access notes

- MFA "Concepts" pages are empty ("under construction"). HMM details were taken from
  the training config and implementation pages.
- The NeMo API reference page renders without class details, so preprocessor and
  augmentor defaults were read from NeMo source on GitHub (main branch).
- The ESPnet TEMPLATE README returned 404, so stages were read from `asr.sh`.
- The videos in the series were not watched. Notes come from the repository's lesson
  list and slide decks.
- Several CCRMA subpages were read only in part, such as the minimum-length bounds
  for resolving sinusoids and the ERB formula.
