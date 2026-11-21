# What we are building

A full explanation of the project for someone who writes software but is new to machine
learning. It goes from the problem, to every part of the system, to how the parts are
trained and tested. Numbers quoted here are measured unless marked as a target.

--------------------------------------------------------------------------------

## 1. The problem

A person recites the Quran into a phone. The app should follow along and tell them, word
by word, when they said something wrong: a different word, a missing or extra word, a
wrong letter, or a wrong vowel. It must almost never tell them they were wrong when they
were right.

That last requirement is the hard one. A checker that is wrong once in a hundred words
sounds good, but a page of the Quran has a few hundred words, so the user would see false
corrections on nearly every page. A recitation app that corrects correct recitation loses
trust immediately, and for this text in particular, a false correction is worse than a
missed one.

### Why ordinary speech recognition is not enough

Speech-to-text systems are built to produce sensible text. They learn which words usually
follow which, and they use that knowledge to repair unclear audio. For dictation that is a
feature. For recitation checking it is a defect: the Quran is a fixed, famous text, so a
system that has seen it will "hear" the correct ayah even when the reciter made a mistake.
Measured on a public benchmark: of 100 planted mistakes, text-producing models reproduced
only 1 to 5 in their output. The mistake was silently corrected away.

So the first design decision is: **the model must not know the text.** It should write
down the sounds it hears and nothing else. Comparing with the expected ayah is a separate,
later step.

--------------------------------------------------------------------------------

## 2. The system in one picture

    microphone
        │  audio, 16,000 numbers per second
        ▼
    ┌──────────────────────┐
    │  LISTENING MODEL     │   a neural network; does not see the text
    │  audio → sounds      │
    └──────────────────────┘
        │  for every 1/25 of a second: how likely is each of 81 sounds
        ▼
    ┌──────────────────────┐        expected ayah ──► rule-based converter ──► expected sounds
    │  CHECKER             │◄──────────────────────────────────────────────────────┘
    │  compare and score   │
    └──────────────────────┘
        │  a score for every expected word
        ▼
    ┌──────────────────────┐
    │  DECISION            │   pass / unsure (say nothing) / flag
    └──────────────────────┘
        │
        ▼
    the app highlights the word and says what kind of mistake it was

Around this sit the tools that make and improve it:

    recordings ─► scoring and filtering ─► labelling tool ─► labelled data
                                                               │
                         training on a GPU  ◄──────────────────┘
                               │
                               ▼
                       new listening model ─► evaluation ─► results log

The rest of this document explains each box.

--------------------------------------------------------------------------------

## 3. Sound tokens: what the model writes

### 3.1 From text to sounds
Written Arabic does not map one-to-one to what is said. Letters merge into the next word,
a noon is hidden before certain letters, a vowel is held for two or four or six counts.
The rules of tajweed describe exactly how the written text turns into sound.

An open, rule-based converter applies those rules to the Uthmani text and writes the
result in a phonetic script. This is plain programming, not machine learning: the same
text always gives the same sounds.

Example, the first ayah of Surah Al-Ikhlas:

    text     قُلْ هُوَ ٱللَّهُ أَحَدٌ
    sounds   ق ُ ل   ه ُ و َ   ل ل َ ا ا ه ُ   ء َ ح َ د ڇ

The lam of the divine name is doubled, the alif is long (written twice), and the final
dal carries qalqalah (the mark ڇ), because the reciter stops on it.

### 3.2 Folding repeats
The script writes length by repetition: a four-count vowel is the symbol four times. We
fold each run into one token with a count:

    ل ل → ل2      ا ا → ا2      ۦ ۦ ۦ ۦ → ۦ4

Two reasons. Sequences get shorter, which matters because the model produces a limited
number of outputs per second. And no two neighbouring tokens are ever equal, which the
training method handles more easily.

### 3.3 The vocabulary
Running the converter over all 6,236 ayahs, with the usual choices of vowel length, gives
81 distinct tokens: 28 consonants plain and doubled, three short vowels, long vowels with
their counts, nasal sounds with their counts, and a few marks. A typical ayah is about 69
tokens.

Because the tokens carry tajweed detail, the same model can later support tajweed
checking. For the main product we ignore that detail when judging (section 6.4).

--------------------------------------------------------------------------------

## 4. The listening model

### 4.1 What a neural network is, briefly
A neural network is a very large formula with adjustable numbers called weights; ours has
about 115 million. Given an input it computes an output. Training means showing it many
inputs with the right outputs and nudging the weights, a little each time, so its outputs
get closer to right. Nobody writes the rules; the weights come to encode them.

### 4.2 From sound wave to features
A microphone gives 16,000 numbers per second: the air pressure over time. That is hard to
learn from directly. The first step converts it into a picture of which frequencies are
present at each moment: 100 times per second, 80 numbers describing the energy from low
to high pitch, spaced the way human hearing is. This is called a mel spectrogram. It is
fixed arithmetic, not learned.

### 4.3 The encoder
The encoder is the learned part. Ours is a FastConformer: 17 layers, each combining

- convolution, which looks at a short stretch of time and is good at local patterns such
  as the burst of a consonant; and
- self-attention, which lets every moment look at every other moment in the clip and is
  good at context, such as a vowel coloured by the consonant three sounds back.

It first shrinks the time axis by eight, from 100 frames per second to 12.5, which makes
it fast. The output is, for every frame, 512 numbers summarising what is being said there.

### 4.4 The head
Two small layers we added:
- one doubles the frame rate back to 25 per second, because fast recitation can have more
  sounds than 12.5 frames per second can hold;
- one turns each frame's 512 numbers into 82 probabilities: one for each of the 81 tokens
  and one for "blank", meaning nothing new here.

### 4.5 Starting from an existing model
Training 115 million weights from nothing needs thousands of hours of audio. Instead we
start from a model that NVIDIA trained on about 1,100 hours of Arabic and released under
a licence that only asks for credit. Its encoder already knows Arabic sounds. We keep it,
attach our new head, and continue training on recitation with our tokens. This is called
fine-tuning.

### 4.6 How it learns without timings: CTC
Our data says "this clip contains these tokens in this order". It does not say when each
token happens. The CTC method removes the need.

The model emits one guess per frame, including blanks. A frame-by-frame output such as

    _ ق ق _ ُ _ _ ل ل _

collapses to "ق ُ ل" by merging repeats and deleting blanks. Many different frame
outputs collapse to the same sequence. CTC adds up the probability of all of them and
trains the model to make that total high for the correct sequence. The model works out
the timing by itself.

A consequence we use everywhere: after training, the model's output is a table of
probabilities, frames down and tokens across. That table is richer than a transcript. It
says not only what was most likely, but how sure the model was, at every moment.

### 4.7 What has been trained
- A pilot on suras 78–114 (about 27 hours): 1.2% of tokens wrong on reciters never seen.
- A full run on all suras, about 9 hours on a free Kaggle GPU: 0.43% of tokens wrong on
  two reciters never seen in training.

Training data so far is an open set of 848 hours by expert reciters, labelled with sound
tokens by the converter. Three reciters are kept out of training entirely so the model
can be tested on voices it has never heard.

--------------------------------------------------------------------------------

## 5. The checker

The checker takes the probability table and the expected tokens of the ayah, and gives
each expected word a score: how badly does the audio fit this word?

### 5.1 Question one: is each expected sound there?
Forced alignment finds the best way to lay the expected tokens along the frames: which
frames belong to which token. It is the same arithmetic as CTC, run to find the single
best path. Then, for each token, look at the probability the model gave it on its frames.
High means the sound was clearly there. Low means the model heard something else.

A word's score is its weakest sound. One wrong letter makes the whole word wrong.

### 5.2 Question two: was something said that should not be there?
Question one cannot notice extra sounds. So we also decode freely (take the most likely
token at each frame, collapse), line that sequence up with the expected one, and look for
sounds that have no partner. Each is charged to the neighbouring word, weighted by how
confident the model was. Adding this took one kind of mistake from never caught to caught
84–88% of the time.

### 5.3 The word score
The larger of the two answers. Bigger means worse.

### 5.4 From score to decision
A threshold turns a score into a decision. Where to put it is the central trade-off:

- low threshold: catches more mistakes, flags more correct words;
- high threshold: fewer false flags, more missed mistakes.

We set it from recordings known to be correct, by voices the model did not train on: find
the score that only 1 in 1,000 correct words exceeds, and use that. Then measure how many
real mistakes still get caught.

There are three outcomes, not two. Between "clearly fine" and "clearly wrong" is a band
where the app says nothing. Silence is always safer than a wrong correction.

### 5.5 Why 1 in 1,000
With independent words, the chance of at least one false flag is 1 − (1 − rate)^words.

| False flags per correct word | 30-word ayah | 300-word session |
|---|---|---|
| 5% | 79% | ~100% |
| 1% | 26% | 95% |
| 0.1% | 3% | 26% |

### 5.6 Words and vowels, not tajweed
Most users are not trained reciters. They may not hold a vowel for the full count or
nasalise where the rules say. Judging that would flag them constantly for things they are
not trying to do. So before scoring, tokens that differ only in tajweed detail are merged:
any length of the same long vowel is one class, hidden and plain noon are one class, the
qalqalah mark is ignored. What remains: which letter, which short vowel, whether a long
vowel is present, whether a letter is doubled.

Tajweed checking becomes an optional mode later, for reciters who want it.

--------------------------------------------------------------------------------

## 6. What has been measured

### 6.1 On studio recitation with planted mistakes
800 clips from two unseen reciters; in each, one mistake planted in the expected text
while the audio stays correct.

| False flags allowed | Mistakes caught |
|---|---|
| 5% of correct words | 99.9% |
| 1% | 98.3% |
| 0.1% | 92.1% |

Wrong letters, wrong vowels, dropped doubling and dropped nasal sounds: all caught at 1%.
Wrong vowel length 97%. Missing qalqalah 84%.

This is an easy test. It shows the scoring logic works. It does not show the product works.

### 6.2 On real phone recordings
The user labelled 67 clips of ordinary people.

- Of 60 clips the model had doubted: 32 were unusable (cut off, another ayah, noise),
  25 were correct, 3 had a real mistake.
- On those 25 correct clips, the checker flagged 54% of the words. Ignoring tajweed made
  no difference (56%). So the model is mishearing letters and vowels on phone audio.
- On 7 clips the model had scored as perfect, the user agreed on all 7.

### 6.3 What that tells us
The model is excellent on studio reciters and poor on phones. It has only ever heard
experts in good recording conditions. The fix is not in the checker; the model needs to
train on real phone audio. That is the next stage.

--------------------------------------------------------------------------------

## 7. Data: the real bottleneck

### 7.1 What exists publicly
- Expert recitation with exact text: plenty (hundreds of hours, open).
- Ordinary people on phones: plenty of audio (about 393,000 clips), but the labels were
  taken from file names and never checked. About half the clips in our sample were not a
  complete recitation of the claimed ayah.
- Real mistakes, marked at the word: almost none. The largest sets cover one sura or one
  ayah, or say only "this clip has a mistake somewhere".

### 7.2 What we therefore build ourselves
1. **A filter.** The model scores each phone clip against the ayah it claims to be. Close
   matches are trusted and used for training. Clear mismatches are dropped. The middle
   goes to a person.
2. **A labelling tool.** It plays a clip, shows the ayah, shows what the model suspects,
   and the labeller confirms, rejects or adds. For each mistake it records what was said,
   sound by sound, next to what should have been said. That pairing is what the strongest
   published systems train on.
3. **Generated mistakes.** A program makes realistic single mistakes (a commonly confused
   letter, a wrong vowel, a shortened long vowel) for testing, and later for training.
4. **Long recordings cut into ayahs.** A full correct recitation by a new voice is split
   at pauses and matched to the text, giving thousands of known-correct words.

### 7.3 The loop
    model scores recordings
       ├─ sure it matches  → training data
       ├─ sure it does not → dropped
       └─ unsure           → labelling tool → person decides
                                   │
                                   ▼
                          better training and test data
                                   │
                                   ▼
                              retrain, measure, repeat

Each round, the model should be unsure about less.

### 7.4 Keeping the test honest
- Whole reciters are held out, never single clips: clips from one voice resemble each other.
- Test data is fixed and named before training on anything similar.
- Results are reported with a range obtained by resampling reciters.
- Results on planted mistakes are never presented as results on real ones.
- When two labellers, or one labeller twice, disagree, that disagreement is the ceiling.

--------------------------------------------------------------------------------

## 8. Training stages

| Stage | Data | Why | State |
|---|---|---|---|
| 1 | Expert reciters, all suras | Learn the sounds of recitation | Done |
| 2 | Add filtered phone clips | Learn ordinary voices, rooms, microphones | Next |
| 3 | Generated mistakes | Learn that expected and said can differ | Later |
| 4 | Real labelled mistakes, last | Learn what real mistakes sound like | Later |
| 5 | The decision layer | Calibrated pass / unsure / flag | Later |

The order of 3 and 4 is deliberate: in published work, artificial mistakes followed by
real ones beat mixing them, and a couple of hours of real mistakes helped more than
eighty hours of artificial ones.

Training runs on free GPU sessions (Kaggle, about 30 hours a week). A run is a notebook
that downloads data piece by piece, trains, tests on the held-out reciters and saves the
best checkpoint.

--------------------------------------------------------------------------------

## 9. From model to app

### 9.1 Live following
In the app, audio arrives continuously. The model must work on a stream, in chunks of
about a second, and a tracker must keep its place in the sura: moving forward as words
arrive, coping with repeats, going back, and stopping mid-ayah. A flag is shown only once
the word is finished, so a self-correction is not punished.

### 9.2 On the phone
The model is converted to a portable format (ONNX), its weights stored in 8 bits instead
of 32 to make it about four times smaller, and run by a small engine inside the app. The
goal is that it runs on the phone itself: no server, no network, recordings never leave
the device. If 115 million weights prove too heavy for older phones, a smaller "student"
model is trained to imitate the large one.

### 9.3 What the app receives
For each word: pass, unsure or flag; for a flag, the kind of mistake and what was heard.

--------------------------------------------------------------------------------

## 10. What makes this different

- It hears sounds, not text, so it cannot paper over mistakes.
- It is judged on false flags first, at one per thousand words, not on average accuracy.
- It has a third answer, silence.
- It is tested on ordinary phone voices with human labels, and reports that separately
  from studio results.
- Its data and licence position are clean: only sources that leave the licence of the
  model a free choice.
- It comes with the tools that made it, so it can keep improving from the user's own labels.

--------------------------------------------------------------------------------

## 11. Where things stand

| Part | State |
|---|---|
| Sound tokens and converter | Done |
| Listening model, studio quality | Done: 0.43% token error on unseen reciters |
| Checker and words-and-vowels mode | Done |
| Evaluation tools | Done; refinements planned |
| Labelling tool | Done and in use |
| Phone audio quality | The main gap: 54% of words flagged on hard correct clips |
| Real-mistake data | 3 clips; needs hundreds |
| Decision layer with calibrated thresholds | Not built |
| Live following | Not built |
| Phone export | Not started |
| Tajweed mode | Deferred by choice |

--------------------------------------------------------------------------------

## 12. Words used here

| Word | Meaning |
|---|---|
| Model | The neural network: a formula with millions of adjustable weights |
| Weights | The adjustable numbers; training changes them |
| Training | Showing examples and nudging weights toward right answers |
| Fine-tuning | Continuing the training of an existing model on new data |
| Checkpoint | A saved copy of the weights |
| Frame | One time slice of audio; the model outputs 25 per second |
| Token | One unit of output: a sound, possibly with a length |
| Blank | The "nothing new" output that CTC uses between sounds |
| CTC | Training method that needs the token sequence but no timings |
| Alignment | Deciding which frames belong to which token |
| Score | How badly the audio fits an expected word |
| Threshold | The score above which a word is flagged |
| False flag | A correct word marked wrong |
| Catch rate | Share of real mistakes that get flagged |
| Held-out | Data from voices the model never trained on |
| Label | What a person says is true about a clip |
| Overfitting | Doing well on training data and worse on new data |
| Quantise | Store weights with fewer bits to shrink the model |
| Distil | Train a small model to imitate a large one |
