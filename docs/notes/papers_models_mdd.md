# Study notes: speech models and mispronunciation detection papers (for Qaf-ft)

Sources read in full (text extracted from the PDF or arXiv HTML; tables read from the
extracted text, figures from captions and prose only):

1. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with
   Recurrent Neural Networks (ICML 2006).
2. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
   (arXiv 2006.11477, v3).
3. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden
   Units (arXiv 2106.07447, v1).
4. Conformer: Convolution-augmented Transformer for Speech Recognition (arXiv 2005.08100);
   Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition
   (arXiv 2305.05084, v6).
5. Robust Speech Recognition via Large-Scale Weak Supervision ("Whisper", arXiv 2212.04356,
   v1, including appendices C to F).
6. Phone-level pronunciation scoring and assessment for interactive language learning
   (Speech Communication 30, 2000; "the GOP paper"); Segmentation-Free Goodness of
   Pronunciation (arXiv 2507.16838, v3, accepted to IEEE TASLP; "the SF-GOP paper").
7. L2-ARCTIC: A Non-Native English Speech Corpus (Interspeech 2018); speechocean762: An
   Open-Source Non-native English Speech Corpus For Pronunciation Assessment
   (arXiv 2104.01378, v2).

Conventions

- Numbers are quoted as printed. "(derived)" = arithmetic on printed numbers.
- **(inference)** = a claim that goes beyond what the paper says, usually an application
  to Qaf-ft. Unmarked text in "What it means for Qaf-ft" restates paper content and links
  it to the project.
- Project facts (81 sound tokens, current checker, phone-audio results) come from
  `docs/research_summary.md` and `qafft/verify.py`. The current token score is the mean
  frame log-probability of each expected token over its Viterbi forced-alignment frames
  (taken from the same CTC model); the word score is the weakest token; an extra-sound
  charge comes from greedy decoding.
- People are named by role only.

Access notes: the GOP paper's university host returned HTTP 403; an archived copy of the
same PDF (10 pages, Speech Communication 30, 95-108) was read instead. Neither corpus paper
reports a numeric inter-annotator agreement; this is stated below rather than filled in.

---

## 1. Connectionist Temporal Classification (ICML 2006)

**Problem.** Train a recurrent network to output a label sequence from unsegmented input,
without frame-level labels and without HMM post-processing.

**Method.**
- Softmax output with one unit per label plus one extra "blank" unit (no label).
- A path is one label-or-blank choice per frame. Path probability = product of the
  per-frame outputs (outputs at different frames are assumed conditionally independent
  given the input).
- A many-to-one map B removes repeated labels, then blanks (B(a-ab-) = B(-aa--abb) = aab).
  The probability of a label sequence is the sum over all paths that map to it.
- Training maximises the log probability of the target sequence (sum over paths) using a
  forward-backward recursion over the target with blanks inserted between labels and at
  both ends (length 2|l|+1). Transitions: stay, advance one, or skip a blank between two
  different labels. A blank is required between two identical labels, so the input must
  have at least U frames plus one per repeated adjacent pair.
- The product of forward and backward variables at (t, s) divided by the output at t gives
  the probability mass of all paths through symbol s at time t; this yields the gradient.
- Decoding: best path (take the most active output each frame, then apply B) is cheap but
  not exact. Prefix search is exact given time but exponential; the paper splits the
  output at frames where blank probability exceeds a threshold (99.99% in the experiment)
  and searches each section separately.

**Data and key numbers.** TIMIT, 61 phones, 4,620 train / 1,680 test utterances, 26 MFCC-based
features. BLSTM with 114,662 weights. Label error rate: context-dependent HMM 35.21%,
BLSTM/HMM hybrid 33.84%, weighted-error hybrid 31.57%, CTC best path 31.47%, CTC prefix
search 30.51%.

**Peaky behaviour.** Figure 1 and section 6: a trained CTC network emits each label as a
spike, separated by strongly predicted blanks. It tends to align each spike with the
right region of input, so approximate segmentation is possible (for example keyword
spotting), but it does not locate label boundaries. Labels that always occur together can
be emitted as a double spike.

**Limitations stated.** No tractable exact decoding. Generalisation is "particularly"
hard for CTC; input noise (std 0.6) mattered more for CTC than for the hybrid. CTC does
not model label dependencies explicitly.

**What it means for Qaf-ft.**
- The project's choice (CTC, no language model) is the setting this paper designed: the
  sequence probability depends only on the label sequence, not on durations.
- Spikes are one or two frames wide. At the encoder's 80 ms frame rate a token's
  "aligned frames" are usually one frame. A mean log-probability over one frame is a
  single noisy number, so the weakest-token rule picks the worst of many single-frame
  draws **(inference)**.
- Blank-threshold splitting (99.99%) is a ready-made way to cut audio into sections at
  confident silences, useful for streaming and for spotting where a clip starts and ends
  **(inference)**.
- The input-noise result supports heavy augmentation in CTC training.

---

## 2. wav2vec 2.0 (arXiv 2006.11477)

**Problem.** Learn speech representations from unlabelled audio, then fine-tune with very
little labelled data.

**Method.**
- Convolutional feature encoder on raw waveform (7 blocks, strides 5,2,2,2,2,2,2; output
  about every 20 ms, receptive field 25 ms), then a Transformer context network with a
  convolutional relative position embedding.
- Encoder outputs are quantised by product quantisation with Gumbel softmax (G=2
  codebooks, V=320 entries each).
- About 49% of time steps are masked (start probability 0.065, span 10 steps; mean span
  299 ms). Loss: for each masked step, pick the true quantised latent among K=100
  distractors from the same utterance, using cosine similarity with temperature 0.1
  (contrastive loss), plus a diversity loss (weight 0.1) that pushes codebook usage to be
  even.
- Fine-tuning: add a random linear output layer, train with CTC, keep the feature encoder
  frozen, train only the output layer for the first 10k updates, use SpecAugment-like
  time and channel masking.

**Data.** Unlabelled: LibriSpeech 960 h or LibriVox 53.2k h (read audiobooks). Labelled:
10 min to 960 h; TIMIT for phones. BASE 95M, LARGE 317M parameters.

**Key numbers.** LARGE, LV-60k: 10 min labelled gives 4.8/8.2 WER (test clean/other) with a
Transformer LM, but 40.2/38.7 with no LM (Table 9). 960 h labelled: 1.8/3.3 with LM;
2.2/4.5 with no LM. TIMIT phone error rate 8.3% (no LM), a 23-29% relative reduction over
the prior best. Ablation: continuous inputs with quantised targets 7.97 WER versus 8.58
for continuous targets (continuous targets let the model use speaker and background
detail). Appendix D: many quantised latents specialise in particular phones without any
labels.

**Limitations.** Pre-training compute is large (LARGE on LV-60k: 128 V100 GPUs, 5.2 days).
All data are read English audiobooks; nothing is tested on phone, noisy or accented
speech. Low-resource numbers lean heavily on the language model.

**What it means for Qaf-ft.**
- With no LM, low-label results are poor (38-40% WER at 10 min). The headline low-resource
  results depend on an LM, which Qaf-ft rules out because it hides mistakes.
- The fine-tuning recipe (freeze the convolutional front end, warm up only the head,
  mask time and channels) transfers directly to the phone fine-tune **(inference)**.
- The contrastive objective's design reason (quantised targets discourage modelling
  speaker and background) points at why self-supervised encoders can be less sensitive
  to channel detail, but the paper never measures channel robustness **(inference)**.

---

## 3. HuBERT (arXiv 2106.07447)

**Problem.** Self-supervised learning for speech has no lexicon of units and no
boundaries; build a BERT-like masked prediction task anyway.

**Method.**
- Offline k-means clustering gives a frame-level pseudo-label per 20 ms frame: first
  100 clusters on MFCCs, then 500 clusters on features from a middle layer of the
  previous HuBERT model (iterative refinement).
- Same encoder shape as wav2vec 2.0. Mask about 8% of frames as span starts (span 10).
  Cross-entropy on the cluster labels, computed on masked frames only (weight alpha=1).
  This forces the model to infer hidden content from context, a combined acoustic and
  language model.
- Cluster quality metrics against forced-aligned phones: phone purity, cluster purity,
  phone-normalised mutual information (PNMI).
- Fine-tune with CTC; convolutional encoder frozen.

**Key numbers.** BASE 95M, LARGE 317M, X-LARGE 964M. 10 min labelled: LARGE 4.7/7.6, X-LARGE
4.6/6.8 (with LM). 960 h: X-LARGE 1.8/2.9. PNMI: MFCC k-means about 0.25-0.29; k-means on
first-iteration layer-6 features 0.56-0.69. Masked-only loss is far more tolerant of bad
targets: with MFCC 100-cluster targets, dev-other WER 17.86 (masked only) versus 96.37
(unmasked only); with good (supervised) targets the gap closes.

**Limitations.** Multi-stage training (cluster, train, re-cluster). Same read-English
data; no phone or noise evaluation. Large compute (LARGE: 128 GPUs).

**What it means for Qaf-ft.**
- "Consistency of targets matters more than correctness" suggests that noisy pseudo-labels
  on phone recitations (for example from a studio-trained teacher, filtered) can still
  teach useful structure **(inference)**.
- The masked-prediction "language model" inside the encoder is a mild form of the
  auto-correction risk: a context model may fill in the expected sound. For a mistake
  checker this needs testing on planted and real mistakes before adoption **(inference)**.

---

## 4. Conformer (arXiv 2005.08100) and Fast Conformer (arXiv 2305.05084)

### Conformer

**Problem.** Transformers model global context, CNNs model local patterns; combine them
parameter-efficiently.

**Method.** Convolutional subsampling (10 ms to 40 ms), then blocks of: half-step
feed-forward, multi-head self-attention with relative positional encoding, a convolution
module (pointwise conv + GLU, depthwise conv, batch norm, Swish, pointwise conv), a
second half-step feed-forward, final layer norm. In words: x + FFN/2, then + attention,
then + convolution, then layer-norm(x + FFN/2). SpecAugment (frequency mask F=27, ten time
masks up to 5% of the utterance each).

**Key numbers.** LibriSpeech, transducer decoder: Conformer(L) 118.8M, 2.1/4.3 WER without LM,
1.9/3.9 with LM; 10.3M model 2.7/6.3. Ablation: removing the convolution block raised
test-other from 4.3 to 4.9; relative position embedding mattered most (5.6 without).
Kernel size 32 best; 65 worse.

**Limitations.** Read English only; transducer decoder; quadratic attention cost.

### Fast Conformer

**Problem.** Conformer is slow and memory-hungry on long audio.

**Method.** 8x downsampling at the start (10 ms to 80 ms) with depthwise separable
convolutions, 256 subsampling channels, conformer conv kernel 9 instead of 31. Optional
limited-context attention (128 steps each side, about 10 s) plus one global token, added
by short fine-tuning. Because CTC needs more frames than target labels, character targets
become too long at 80 ms, so CTC models switch to SentencePiece BPE (128-1024 tokens).

**Key numbers.** Encoder about 2.8x faster, 2.9x fewer multiply-adds. LibriSpeech CTC:
Conformer 5.74 versus Fast Conformer 5.64 test-other (115M, 51.5 GMACs). 25k h set, CTC:
Fast Conformer 4.19 test-other, 9.00 MCV. Long audio with limited context + global token:
Earnings-21 18.26 (Conformer) to 11.85. Maximum audio on one A100: 15 min (Conformer) to
675 min. XL/XXL models were initialised from wav2vec 2.0-style self-supervised
pre-training to stabilise training. Adding 40k h of extra data "facilitated enhanced
accuracy and noise robustness" (Figure 3: WER across SNR).

**Limitations.** No streaming (cache-aware) results in this paper; noise robustness is
shown only for the largest models and only by figure; no phone-channel test.

**What it means for Qaf-ft.**
- The project encoder runs at 80 ms per frame. With CTC, a token sequence needs at least
  one frame per token plus one blank frame between identical neighbours; the checker
  already checks this (`need` in `token_scores`). Fast recitation with many short sounds
  per second may approach the limit, which would show up as forced-alignment failures,
  not as low scores **(inference)**.
- At 80 ms a short vowel often occupies one frame, so vowel scores are especially
  sensitive to channel and timing jitter. This argues for alignment-free scores (section 6)
  rather than single-frame averages **(inference)**.
- The paper itself attributes noise robustness to more and more varied data, not to the
  architecture. This supports the phone fine-tune plan over an architecture change.
- Limited-context attention with a global token is the paper's long-audio method; it is
  not cache-aware streaming. Streaming needs a separate conversion.

---

## 5. Whisper: Robust Speech Recognition via Large-Scale Weak Supervision (arXiv 2212.04356)

**Problem.** Fine-tuned models look superhuman in-distribution but fail on other datasets.
Train on a very large, diverse, weakly labelled set and evaluate zero-shot.

**Method.**
- 680,000 h of audio with internet transcripts (117,000 h in 96 other languages; 125,000 h
  translation). Filtering removes machine-generated transcripts (all-caps, no commas and
  similar cues), language-mismatched pairs and low-quality sources; fuzzy de-duplication.
- Encoder-decoder Transformer on 80-channel log-Mel, 30 s windows. The decoder is an
  audio-conditioned language model; it is trained to condition on previous transcript text
  (50% of the time). Special tokens select language, task, timestamps; a no-speech token
  is trained on speechless segments.
- No data augmentation in the original models; Large V2 adds SpecAugment, stochastic depth
  and BPE dropout.

**Robustness numbers.** At equal LibriSpeech clean WER (2.7), Whisper Large V2 makes 55.2%
fewer errors on average across 12 other English datasets than wav2vec 2.0 Large (no LM)
(Table 2; for example Common Voice 29.9 to 9.0, CHiME-6 65.8 to 25.5). Additive noise
(Figure 5): models trained on LibriSpeech, including wav2vec 2.0 and HuBERT variants,
degrade fastest; NVIDIA Conformer models are best at low noise; Whisper is best below
10 dB SNR of pub noise. Arabic: 739 h of Arabic ASR data; Common Voice 9 Arabic WER 53.8%
(Large V2). Model sizes 39M to 1,550M. Dataset-scaling runs evaluated an exponential moving
average of weights (rate 0.9999).

**Hallucination and failure modes.** The limitations section lists repeat loops, skipping
the first or last words of a segment, and "complete hallucination" (a transcript
unrelated to the audio), described as failures of sequence-to-sequence models, language
models and text-audio alignment. Long-form decoding needs heuristics: beam search (5),
temperature fallback when average log-probability is below -1 or the gzip compression
ratio exceeds 2.4, voice activity detection by combining no-speech probability 0.6 with
average log-probability -1, previous-text conditioning, and constraining the first
timestamp to 0-1 s. Early models also invented speaker names, removed by fine-tuning on
transcripts without speaker labels. The paper suspects its robustness is partly due to
the strong decoder language model.

**Text normalisation.** WER is computed after heavy normalisation, built by inspecting
Whisper's own errors; for some datasets WER drops by up to 50%. The paper warns the
normaliser may be overfitted to Whisper's style. For non-English text the normaliser
replaces every character whose Unicode category starts with M, S or P with a space
(appendix C). Arabic short vowels, shadda and sukun are category M, so Whisper's Arabic
WERs ignore diacritics entirely (derived).

**Limitations stated.** Decoding failures above; poor low-resource languages; zero-shot
only (fine-tuning not studied); unclear whether robustness comes from encoder or decoder.

**What it means for Qaf-ft.**
- The decoder language model is the auto-correction mechanism the project must avoid: it
  is designed to "resolve ambiguous audio" from text context, which is exactly what turns
  a mistaken vowel into the expected one. This agrees with `research_summary.md` (text
  models keep 1.2-4.9 of 100 planted mistakes).
- Published Whisper Arabic WERs say nothing about vowel accuracy, because the normaliser
  deletes diacritics.
- The transferable lesson is data, not architecture: robustness came from diverse
  recording conditions plus careful transcript filtering. The FastConformer paper says the
  same.
- Whisper's heuristic gates (average log-probability, no-speech probability, compression
  ratio) are a template for an utterance-level "can this clip be scored?" gate in Qaf-ft
  **(inference)**. Batch 1 of the phone labels was dominated by unusable clips (24
  incomplete, 5 different ayah, 3 noise of 60), so such a gate removes false flags before
  any word is scored.
- The Whisper encoder alone (without its decoder) could in principle be fine-tuned with a
  CTC head, but the paper offers no evidence on that, and it is 30 s-window, non-streaming
  and large **(inference)**.

---

## 6a. Goodness of Pronunciation (Speech Communication 30, 2000)

**Problem.** Score each phone of a non-native utterance for a computer-assisted language
learning system, with a known text, and validate against human judges.

**Method.**
- GOP(p) = |log P(p | O_p)| / NF(p): the log posterior of the expected phone p given its
  acoustic segment, divided by the segment's frame count. The posterior is the
  likelihood under p divided by the sum over all phones of their likelihoods (equal
  priors). The sum is approximated by its maximum.
- Numerator from forced alignment of the known phone sequence; denominator from an
  unconstrained phone loop, summing per-frame log-likelihoods over the same frames (so
  several loop phones can contribute). A threshold rejects a phone; "the choice of
  threshold depends on the level of strictness required".
- Speaker normalisation by one global MLLR transform of the Gaussian means, limited so it
  cannot learn the speaker's error patterns.
- Phone-dependent thresholds: T1(p) = alpha * mean_p + beta from native GOP statistics
  (some phones, such as fricatives, score worse acoustically); T2(p) from the judges'
  normalised rejection counts for that phone.
- Error modelling: recognition networks with expected substitutions from the learner's
  first language (Spanish phone models added), giving an extra penalty
  GOP2 = GOP1 + K * GOPe.

**Evaluation measures.** Frame-level comparison after forced alignment, smoothed by a
15-frame Hamming window. Strictness S = rejected phones / all phones. Agreement A =
1 - normalised city-block distance between rejection vectors (all frames). Cross-
correlation CC = normalised dot product of rejection vectors (only frames rejected by at
least one side; judged most important). Phone correlation PC = correlation of
per-phone rejection counts.

**Data.** 10 intermediate learners (Spanish, Italian, Japanese, Korean first languages),
40 common + 80 varying read sentences each; 20 calibration sentences from one speaker
labelled by all six phonetician judges. Native British English monophone HMMs.

**Key numbers.** Judge strictness mean 0.18, std 0.05. Between judges: A 0.85-0.95,
CC 0.40-0.65, PC 0.70-0.85, strictness gap 0.03-0.14; means A 0.91, CC 0.47, PC 0.78,
dS 0.06. Baseline GOP mean A 0.88, CC 0.47, PC 0.60. A lenient judge (strictness 0.07-0.13)
gave much worse machine agreement (CC 0.12 for one speaker). Adaptation plus judge-based
thresholds raised PC from 0.62 to 0.72; with error modelling (Spanish speakers only)
A 0.90, CC 0.49, PC 0.78, equal to the human means. The first phone of a sentence was
wrongly rejected in the example, "probably due to bad acoustic modelling at the sentence
beginning".

**Limitations.** Tiny data; frame-level measures; thresholds tuned per speaker within one
standard deviation of judge strictness; substitutions only.

**What it means for Qaf-ft.**
- Human judges agree only moderately at phone level (CC 0.47). A 90-95% precision goal
  must be checked against human-to-human agreement on Qaf-ft's own labels; the machine
  cannot be scored more precisely than the labels are consistent **(inference)**.
- Phone-dependent thresholds are the classic fix for "some sounds always score low". The
  current weakest-token rule uses raw log-probabilities, so words containing intrinsically
  low-scoring tokens are flagged more often. Normalising each token's score by its class
  mean and spread on known-correct phone audio (the T1 idea) before taking the minimum is
  cheap **(inference)**.
- The edge effect (first phone rejected) matches a phone-audio risk: clipped starts and
  ends. Word position should be a verifier feature **(inference)**.
- Strictness is a free parameter; the field reports it. Qaf-ft's fixed false-flag rate is
  the same idea expressed on correct words.

## 6b. Segmentation-Free Goodness of Pronunciation (arXiv 2507.16838)

**Problem.** GOP needs phone segments from forced alignment, which are unreliable for
mispronounced, child or accented speech and do not match where CTC models put their
activations. Classic GOP also handles only substitutions.

**Method.**
- Peakiness has two parts: peaky over time (blank most of the time; measured by blank
  coverage BC = average blank posterior) and peaky over states (one output near 1 per
  frame; measured by conditional entropy of alignment paths, ConEn).
- GOP-SA (self-alignment): the average log posterior of the target over the frames where
  the *same* CTC model aligns it (instead of an external aligner).
- GOP-SF (segmentation-free): log p(target | all audio, left context, right context) =
  log p(full canonical sequence | audio) - log p(left context, any, right context | audio).
  Both terms are CTC losses: GOP-SF = CTC loss(L_SDI) - CTC loss(L_C), where L_SDI replaces
  the target by "any sequence of phones" in a graph. Variants restrict the replacement:
  SD = any single phone or nothing (substitution or deletion), S = any single phone. A
  uniform prior over segmentations is assumed so it cancels.
- GOP-SF-Norm divides by the expected number of frames occupied by the target (sum of
  normalised forward variables over the central graph nodes), floored at 1.
- Cost: two forward passes per target phone over graphs of size 2|L|+1 and 2|L|+|V|,
  linear in utterance length.
- Feature vector FGOP: LPP (whole-utterance CTC log posterior), the log posterior ratio of
  the canonical sequence versus every single-phone substitution or deletion at that
  position (|V|+1 values), and expected occupancy; fed to SVR or GOPT.

**Data.** CMU Kids (9.1 h, children 6-11, 90.2% correct phones, 3.2% substitutions, 2.3%
deletions, 4.3% insertions) with real and simulated errors; speechocean762. Acoustic
models: wav2vec2-large-xlsr-53 fine-tuned on LibriSpeech 100 h with CE, CTC, entropy-
regularised CTC, or equal-spacing CTC.

**Key numbers (CMU Kids, AUC averaged over phones).** Real errors: CE-trained DNN GOP 0.796;
CE GOP-Avg 0.851; CTC GOP-SA 0.905; CTC GOP-SF-S 0.891; CTC GOP-SF-SD 0.914; CTC GOP-SF-SDI 0.859.
Simulated errors: 0.988-0.989 for CTC SA/SF-S. Peakiness: plain CTC has the highest blank
coverage (88.62%) and lowest ConEn (2.83), the best phone error rate (11.46%), and the best
GOP-SA (0.909); GOP-SF is about equally good (0.913-0.915) for all CTC variants, i.e.
insensitive to peakiness. Standard GOP with the external aligner collapses for a model
whose spikes drift (equal-spacing CTC: 0.580). Context: 8-14 phones of context is as good
as the full utterance; zero context (equals GOP-SA) is worse. speechocean762 phone PCC:
scalar GOP-CTC-SF-SD 0.450 (classic TDNN GOP 0.361); FGOP-CTC-SF-Norm with SVR 0.581, with
GOPT 0.648 (best prior 0.693 with a much heavier model); children 0.635, adults 0.651.

**Limitations.** English read speech; phone level only; insertion variant not properly
evaluated; scalar GOP still needs thresholds; uniform segmentation prior is an assumption.

**What it means for Qaf-ft.**
- The current checker's token score is GOP-SA (mean log posterior over self-aligned CTC
  frames), with no normalising denominator. The paper shows SA is good for peaky CTC
  models but SF-SD is significantly better on real errors (0.914 vs 0.905, non-overlapping
  95% intervals).
- Simulated errors overstate performance (0.989 vs 0.91 for the same method), the same
  warning as the project's planted-mistake results.
- The project's extra-sound check plays the role of the insertion path, which the paper
  found hard to use in a scalar score. Keeping it separate, as a verifier feature, is
  consistent with the paper's results **(inference)**.
- FGOP-style vectors fed to a simple classifier beat scalar GOP by a wide margin, which
  supports the planned word-level verifier.

---

## 7a. L2-ARCTIC (Interspeech 2018)

**Problem.** No open non-native English corpus with phone-level error annotation for
mispronunciation detection (and voice or accent conversion).

**Annotation scheme.** 10 speakers (Hindi, Korean, Mandarin, Spanish, Arabic; one male,
one female each), about one hour each of CMU ARCTIC prompts, 11,026 utterances, 11.2 h,
quiet room, studio USB microphone at 20 cm, 44.1 kHz. Montreal Forced Aligner gives word
and phone boundaries. About 150 utterances per speaker (100 shared + 50 chosen for that
first language's difficulties) were hand-corrected: boundaries, actual phones (ARPAbet)
and tags for substitution, deletion and addition, with free comments (IPA allowed).
Three annotators (applied linguistics PhD students); automated consistency scripts flagged
problems for fixing. 1,499 utterances annotated: 5,199 substitutions, 1,048 deletions, 497
additions. Licence CC BY-NC 4.0.

**Agreement.** Not reported in the paper.

**Metrics and baseline.** Classic GOP (Kaldi triphone GMM, LibriSpeech 960 h), one global
threshold, substitutions only. 206 utterances to choose the threshold range; 1,293 test
utterances, 41,353 phones, 4,415 (10.7%) substitutions. Precision = correctly predicted
errors / predicted errors; recall = correctly predicted errors / true errors. At threshold
-4.2 (log GOP) precision = recall = 0.29.

**What it means for Qaf-ft.**
- The field's phone-level metrics are precision and recall on the "mistake" class,
  computed at a chosen threshold, with prevalence printed (10.7%). Precision of 0.29 is the
  classic-GOP floor on accented read speech.
- Its 50 first-language-targeted sentences per speaker are a model for a Qaf-ft test set
  enriched with likely errors, but enrichment changes prevalence and therefore precision;
  report both enriched and natural rates **(inference)**.
- Non-commercial licence: research reference only.

## 7b. speechocean762 (arXiv 2104.01378)

**Problem.** An open corpus for pronunciation assessment, free for commercial use, with
children and adults, scored at sentence, word and phone level.

**Data.** 5,000 utterances, 250 Mandarin-speaking learners (half children; gender 1:1),
20 sentences each, about 6 h. Recorded on **mobile phones** (Apple, Samsung, Xiaomi,
Huawei) held 20 cm from the mouth in a quiet 3x3 m room. Train/test split by speaker,
125 each.

**Annotation scheme.** Five experts scored every utterance independently. Phone: 2 correct,
1 heavy accent, 0 incorrect or missed. Word: accuracy 0-10 (with bands such as "no more than
30% of phones wrong" = 4-6) and stress 10/5. Sentence: accuracy, completeness (share of
words pronounced), fluency, prosody. Multiple valid canonical pronunciations ("fast" as
F AA S T or F AE S T) were handled by each expert choosing the closest, then a vote fixed
one canonical sequence before scoring. The scoring tool warns on inconsistent scores (high
word score with all low phone scores). Most word and phone scores are 8-10 (imbalanced).

**Agreement.** Not reported numerically in the paper.

**Metrics and baseline.** Kaldi recipe: native TDNN acoustic model, forced alignment, GOP and
GOP features, one regressor per phone. MSE and Pearson correlation (PCC) with the human
phone score: GOP value 0.69 / 0.25; GOP features with SVR 0.16 / 0.45. Imbalance handled by
adding good samples of *other* phones as low-score samples of the current phone.

**What it means for Qaf-ft.**
- It is a public phone-recorded pronunciation set, so a quick sanity check of any GOP
  implementation on phone-microphone audio exists (English, so only for method checks)
  **(inference)**.
- The canonical-sequence vote is directly relevant: where a Quran word has more than one
  acceptable realisation (for example optional stops, or pause forms), the expected token
  sequence must accept all of them, or correct readers get flagged **(inference)**.
- "Other phones' good samples as negatives" is the same idea as the project's planted
  text-side mistakes.

---

## Application to Qaf-ft

Priorities run from cheapest-with-largest-expected-effect to more costly. Everything
here is **(inference)** unless it restates a result above.

### (a) Word scores from CTC posteriors

1. **Add a word-level segmentation-free score next to the current weakest-token score.**
   For each word w: score = CTC loss(expected sequence with w replaced by "any token
   sequence", or by "any single token or nothing" per position) minus CTC loss(expected
   sequence). This is the SF-GOP paper's GOP-SF applied at word or token level. It sums over
   all alignments, so it no longer depends on which single 80 ms frame the spike landed on,
   and it is insensitive to peakiness (paper: 0.913-0.915 AUC across CTC variants). Cost is
   one extra forward pass per word on a graph of about 2|L|+81 nodes; an ayah of 40 tokens
   and 200 frames is trivial. Start with the SD variant per token (best on real errors),
   then take the word minimum, and compare with the current score on the phone labels.
2. **Normalise token scores by token class before the minimum.** Fit mean and spread of each
   token's score on known-correct phone audio (and separately studio audio), z-score, then
   take the word minimum. This is the GOP paper's phone-dependent threshold T1 and costs
   nothing at inference.
3. **Use competitor margins, not only the target's own probability.** For each token, the
   log ratio between the expected sequence and its best single-token substitution or
   deletion (the FGOP "LPR" vector) separates "the audio fits another token better" (a
   mistake) from "the audio fits nothing well" (noise, phone channel). The second case
   should push toward "unsure", not "flag".
4. **Handle blank explicitly.** Report expected occupancy (frames) per token and blank mass
   around it. Very low occupancy plus high blank suggests a dropped or swallowed sound;
   low target probability with high competitor probability suggests a substitution;
   both low suggests bad audio. The SF-GOP Norm variant divides by occupancy floored at 1;
   with plain CTC this matters little (paper), so occupancy is better as a feature than as
   a divisor.
5. **Keep the extra-sound charge separate.** The insertion path made the SF-GOP scalar score
   worse (0.859); the project's separate extra-sound check is the better design, used as a
   verifier feature.
6. **Forced-alignment scores remain useful for feedback** (where in the word the problem
   is), while the segmentation-free score decides whether to flag.

### (b) How results are reported, and the 90-95% precision goal

- Field practice in these papers: precision and recall on the mistake class at one threshold
  with prevalence stated (L2-ARCTIC, P = R = 0.29 at 10.7% prevalence); threshold-free AUC
  averaged over phones (SF-GOP); Pearson correlation with ordinal expert scores
  (speechocean762); agreement measures against human-to-human ceilings (GOP paper). The
  common precision/recall/F1 at phone level with true/false acceptance and rejection
  counts comes from later papers not read here.
- Qaf-ft's "share caught at a fixed false-flag rate" is recall at a fixed false-positive rate
  on correct words: it does not depend on how common mistakes are. Precision does:
  precision = r*pi / (r*pi + f*(1 - pi)), with r recall, f false-flag rate, pi share of
  words that are wrong.
- Precision at recall 0.8 (derived):

  | false-flag rate f | pi = 2% | pi = 5% | pi = 10% |
  |---|---|---|---|
  | 0.1% | 0.94 | 0.98 | 0.99 |
  | 0.3% | 0.85 | 0.93 | 0.97 |
  | 1% | 0.62 | 0.81 | 0.90 |
  | 5% | 0.25 | 0.46 | 0.64 |

  False-flag rate needed at recall 0.8: for 90% precision, 0.18% / 0.47% / 0.99% at
  pi = 2% / 5% / 10%; for 95% precision, 0.09% / 0.22% / 0.47%. The phone-audio pilot's 19%
  strong flags on correct words gives about 18% precision at pi = 5%.
- So the goal translates to a false-flag budget of roughly 0.1-0.5% of correct words on real
  phone audio, consistent with the 0.3% target already in `research_summary.md`. Recommend
  reporting, per release: recall at 0.1/0.3/1% false flags, precision at the measured
  prevalence of the labelled phone set (with intervals by reciter), and the "unsure" rate.
- Measure labeller agreement before promising a precision figure. The GOP paper's judges
  agreed at CC 0.47 on phones; the RetaSy set reports 0.63 between raters at clip level.
  Have a second labeller relabel a subset blind; precision against one labeller cannot
  meaningfully exceed the agreement between two. Word and vowel mistakes are likely more
  agreed-on than accent judgements, but this needs a number.
- Precision targets should apply to "flag" only; borderline words go to "unsure". The GOP
  paper's strictness spread (0.13-0.25 between judges) shows that a borderline band exists
  in the labels themselves.

### (c) Self-supervised encoders or Whisper versus the current FastConformer

- None of these papers tests phone-microphone recitation. Direct evidence on robustness:
  Whisper Figure 5 shows LibriSpeech-fine-tuned wav2vec 2.0 and HuBERT models degrading
  faster in noise than NVIDIA Conformer models, which in turn are best at low noise; Whisper
  wins below 10 dB SNR because of its training data. The FastConformer paper attributes its
  own noise robustness to extra data.
- Conclusion: switching to a wav2vec 2.0 or HuBERT style encoder is not justified by these
  papers for phone robustness; their advantage is low-label learning, and their low-label
  results lean on a language model. The SF-GOP paper's numbers are with wav2vec2-xlsr, so
  the scoring method itself is encoder-agnostic.
- Whisper should not be the checker: decoder language model (auto-correction), hallucination
  modes, 30 s windows, size, and its Arabic evaluation strips diacritics. It agrees with
  `research_summary.md` ("not suitable"). One nuance: its encoder could serve as an offline
  teacher feature extractor, but no paper here supports that.
- Where self-supervision may help: continued self-supervised pre-training of the existing
  encoder on unlabelled phone recitations (the FastConformer paper already uses wav2vec
  2.0-style pre-training for its large models), or HuBERT-style pseudo-label training on
  phone audio. Both are costly; try data-side fixes first.

### (d) Robustness to phone audio

1. **Train on phone audio and diverse channels** (the common lesson of Whisper and
   FastConformer). The planned studio + phone mix with EMA fits this; Whisper also evaluated
   EMA weights (0.9999). Follow wav2vec 2.0's fine-tuning habits: freeze the convolutional
   subsampling at first, warm up the head, then unfreeze.
2. **Augment heavily** for CTC (the CTC paper's input-noise finding; SpecAugment in all
   encoder papers): additive noise at low SNR, codec and band-limiting, reverberation, gain
   changes, clipping, and time masking.
3. **Gate the utterance before scoring words.** Use utterance-level features (whole-sequence
   CTC log posterior per frame, as in FGOP "LPP"; blank share; free-decode versus expected
   edit distance; estimated SNR; clip coverage of the ayah) to classify clips as scorable,
   incomplete, different ayah or too noisy, as Whisper does with log-probability and
   no-speech thresholds. Batch 1 shows this is where most phone false flags start.
4. **Treat edges specially.** Pad, trim confidently, and give the verifier word-position
   features; the GOP paper saw edge rejections, and Whisper drops first or last words.
5. **Calibrate on phone audio.** Fit token-class normalisation and verifier calibration on
   known-correct phone recitations, not studio ones; thresholds set on studio reciters
   flagged 51% of correct phone words.
6. **Accept alternative realisations** in the expected sequence (the speechocean762 vote):
   any pronunciation an expert would accept must be in the graph, or it becomes a false flag.

### Top five, in order

1. Utterance gate (scorable / incomplete / wrong ayah / noisy) before word scoring.
2. Word-level segmentation-free score (GOP-SF-SD style) plus competitor margins, compared
   with the current weakest-token score on the labelled phone clips.
3. Token-class normalisation and threshold calibration on known-correct phone audio.
4. Verifier on per-word features (SF score, SA score, margins, occupancy, blank share,
   extra-sound charge, position, utterance quality) with pass / unsure / flag; report recall
   at 0.1-1% false flags and precision at measured prevalence.
5. Phone fine-tune with diverse-channel augmentation and EMA; keep FastConformer.
