# Learning resources

A verified reading and course list for building a Quran recitation speech model: a FastConformer encoder fine-tuned with CTC (NVIDIA NeMo, PyTorch) that outputs phoneme and vowel-length tokens, a forced-alignment checker with goodness-of-pronunciation style word scores, the data pipeline around it, statistical evaluation, and export to a phone (ONNX, 8-bit quantisation, sherpa-onnx or ONNX Runtime in Flutter).

Compiled 2026-11-21. The reader is assumed to be a software-engineering graduate who is new to machine learning.

## How each entry was checked

Every entry was checked on the web on the compile date. The "Verified" column says how:

- **opened**: the page itself was fetched and its title matched.
- **search**: the exact URL and title appeared in web search results, but the page was not fetched.
- **search (site)**: another page of the same site appeared in search results under the expected title; the exact URL given here was not itself seen. Treat these links as the most likely to need a correction.

Only a small set of hosts (arXiv, Hugging Face, Kaggle, Tanzil, EveryAyah, pub.dev and one Microsoft Research page) could be fetched directly from the environment used to compile this list. Everything else is marked "search" or "search (site)". Resources that could not be confirmed either way were left out.

Costs: "free" means free to use; "free to read online" means the full text is on the web but a print edition is sold; "free PDF from the author/publisher" means the PDF is offered by the rights holder; "paid" entries link to the publisher or an official seller only.

Levels are relative to this project: beginner needs no ML background, intermediate assumes the start-here path, advanced is reference material and research papers.

## Start here

Twelve resources to do first, in this order. Times are rough, for focused study with the exercises.

| # | Resource | Roughly | What to take from it |
|---|---|---|---|
| 1 | [Machine Learning Crash Course](<https://developers.google.com/machine-learning/crash-course>) (Google) | 15 hours | Loss, overfitting, train and test splits, classification thresholds. |
| 2 | [But what is a Neural Network?](<https://www.3blue1brown.com/lessons/neural-networks/>) series (3Blue1Brown) | 4 hours | A picture of what a network and gradient descent are doing. Watch the linear algebra and calculus series alongside if rusty. |
| 3 | [Learn the Basics](<https://docs.pytorch.org/tutorials/beginner/basics/intro.html>) (PyTorch) | 5 hours | Tensors, datasets, autograd, a training loop. |
| 4 | [Practical Deep Learning for Coders](<https://course.fast.ai/>), lessons 1 to 5 (fast.ai) | 25 hours | Fine-tuning a pretrained model, the habit of looking at data and errors. |
| 5 | [Neural Networks: Zero to Hero](<https://karpathy.ai/zero-to-hero.html>) (Andrej Karpathy) | 20 hours | Backpropagation and a transformer written by hand. |
| 6 | [Hugging Face Audio Course](<https://huggingface.co/learn/audio-course/chapter0/introduction>), units 1, 2, 3 and 5 | 20 hours | Audio data, spectrograms, CTC models, fine-tuning and scoring a recogniser. |
| 7 | [Speech Processing for Machine Learning](<https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html>) (Haytham Fayek) | 2 hours | Exactly how log-mel features are computed. |
| 8 | [Sequence Modeling with CTC](<https://distill.pub/2017/ctc/>) (Awni Hannun) | 3 hours | The blank token, alignments, the forward algorithm. Read it twice. |
| 9 | [Forced Alignment with Wav2Vec2](<https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html>) (torchaudio tutorial) | 3 hours | The alignment trellis and backtracking, which the checker is built on. |
| 10 | [NeMo ASR documentation](<https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html>) and the [tutorials index](<https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html>) (NVIDIA) | 10 hours | Configs, manifests, fine-tuning a CTC model on a new token set. |
| 11 | [A Recipe for Training Neural Networks](<https://karpathy.github.io/2019/04/25/recipe/>) (Andrej Karpathy) | 1 hour | A debugging routine to follow on every training run. |
| 12 | [A Classical Primer on Tajwid](<https://rissc.jo/wp-content/uploads/2018/11/A_Classical_Primer_on_Tajwid_EN.pdf>) (Royal Aal al-Bayt Institute) | 6 hours | The rules the token sequences encode: noon and meem rules, madd lengths, articulation points. |

After these, the most useful next steps are the ROC and calibration pages in area 13, the goodness-of-pronunciation papers in area 8, and the quantisation documentation in area 15.

## Contents

1. [Maths foundations for ML](#1-maths-foundations-for-ml) (11)
2. [Python for data and ML](#2-python-for-data-and-ml) (9)
3. [Machine learning fundamentals and statistical learning](#3-machine-learning-fundamentals-and-statistical-learning) (8)
4. [Deep learning: general, training practice and debugging](#4-deep-learning-general-training-practice-and-debugging) (15)
5. [Sequence models and Transformers](#5-sequence-models-and-transformers) (12)
6. [Digital signal processing and audio basics](#6-digital-signal-processing-and-audio-basics) (12)
7. [Speech recognition: classic and modern, with CTC](#7-speech-recognition-classic-and-modern-with-ctc) (23)
8. [Pronunciation assessment and mispronunciation detection](#8-pronunciation-assessment-and-mispronunciation-detection) (21)
9. [Phonetics and phonology, including Arabic](#9-phonetics-and-phonology-including-arabic) (11)
10. [Tajweed and Quran recitation science](#10-tajweed-and-quran-recitation-science) (17)
11. [Toolkits and official documentation](#11-toolkits-and-official-documentation) (29)
12. [Data work: datasets, labelling, label noise](#12-data-work-datasets-labelling-label-noise) (18)
13. [Evaluation and statistics for ML](#13-evaluation-and-statistics-for-ml) (19)
14. [ML engineering](#14-ml-engineering) (14)
15. [Model compression and on-device deployment](#15-model-compression-and-on-device-deployment) (25)
16. [Streaming and real-time speech recognition](#16-streaming-and-real-time-speech-recognition) (11)
17. [Licensing and ethics of data](#17-licensing-and-ethics-of-data) (14)
18. [Communities, lecture courses, newsletters and research venues](#18-communities-lecture-courses-newsletters-and-research-venues) (17)
19. [Foundational and directly relevant research papers](#19-foundational-and-directly-relevant-research-papers) (35)

Total: 321 resources.

## 1. Maths foundations for ML

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Essence of linear algebra (series preview and lessons)](<https://www.3blue1brown.com/lessons/eola-preview/>) | 3Blue1Brown (Grant Sanderson) | video series | beginner | free | search | Visual intuition for vectors and matrices, the objects every network layer manipulates. |
| [The Essence of Calculus](<https://www.3blue1brown.com/lessons/essence-of-calculus/>) | 3Blue1Brown (Grant Sanderson) | video series | beginner | free | search | Derivatives and the chain rule, the basis of backpropagation. |
| [Seeing Theory: A Visual Introduction to Probability and Statistics](<https://seeing-theory.brown.edu/>) | Daniel Kunin and co-authors, Brown University | free online book | beginner | free | search | Interactive pictures of distributions, confidence intervals and inference. |
| [Think Stats: Exploratory Data Analysis in Python (3rd edition)](<https://allendowney.github.io/ThinkStats/>) | Allen B. Downey | free online book | beginner | free to read online | search | Statistics taught through Python code; a natural entry for a software engineer. |
| [The Matrix Calculus You Need For Deep Learning (arXiv 1802.01528)](<https://arxiv.org/abs/1802.01528>) | Terence Parr, Jeremy Howard | paper | intermediate | free | opened | Exactly the calculus needed to read loss and gradient derivations. |
| [Mathematics for Machine Learning](<https://mml-book.github.io/book/mml-book.pdf>) | Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong | free online book | intermediate | free PDF from the authors | search | One book covering linear algebra, calculus, probability and optimisation for ML. |
| [MIT 18.06 Linear Algebra](<https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/>) | Gilbert Strang, MIT OpenCourseWare | course | intermediate | free | search | The standard full lecture course on linear algebra. |
| [Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares](<https://web.stanford.edu/~boyd/vmls/vmls.pdf>) | Stephen Boyd, Lieven Vandenberghe | free online book | intermediate | free PDF from the authors | search | Applied linear algebra with least squares, useful for features and regression-style scoring. |
| [Statistics 110: Probability](<https://stat110.hsites.harvard.edu/>) | Joe Blitzstein, Harvard University | course | intermediate | free | search | Rigorous but friendly probability course; underpins CTC, HMMs and likelihood ratios. The site also links the free online edition of the companion book. |
| [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](<https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/>) | Gilbert Strang, MIT OpenCourseWare | course | advanced | free | search | Linear algebra aimed directly at data, signals and neural networks. |
| [Convex Optimization](<https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>) | Stephen Boyd, Lieven Vandenberghe | free online book | advanced | free PDF from the authors | search | Reference for the optimisation theory behind gradient methods and logistic calibration. |

## 2. Python for data and ML

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [NumPy: the absolute basics for beginners](<https://numpy.org/doc/stable/user/absolute_beginners.html>) | NumPy project | documentation | beginner | free | search | Arrays, shapes and broadcasting, used everywhere in audio and feature code. |
| [NumPy Learn (curated tutorials and books)](<https://numpy.org/learn/>) | NumPy project | documentation | beginner | free | search | Official list of further NumPy material, sorted by level. |
| [Kaggle Learn](<https://www.kaggle.com/learn>) | Kaggle | course | beginner | free | opened | Short hands-on micro-courses (Python, pandas, intro ML) on the same platform used for free GPUs. |
| [pandas getting started tutorials](<https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html>) | pandas project | documentation | beginner | free | search | Tables of utterances, labels and scores are handled as data frames. |
| [Learn the Basics (PyTorch tutorial series)](<https://docs.pytorch.org/tutorials/beginner/basics/intro.html>) | PyTorch project | tutorial | beginner | free | search | Official first tour of tensors, datasets, autograd and the training loop. |
| [Deep Learning with PyTorch: A 60 Minute Blitz](<https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>) | PyTorch project | tutorial | beginner | free | search | Fast overview of autograd and a first trained network. |
| [Zero to Mastery Learn PyTorch for Deep Learning](<https://www.learnpytorch.io/>) | Daniel Bourke | free online book | beginner | free to read online | search | Code-first PyTorch course with exercises; good for building fluency. |
| [Python Data Science Handbook](<https://jakevdp.github.io/PythonDataScienceHandbook/>) | Jake VanderPlas | free online book | beginner | free to read online | search (site) | NumPy, pandas, Matplotlib and scikit-learn in one place. |
| [Scientific Python Lectures](<https://lectures.scientific-python.org/index.html>) | Scientific Python community | free online book | intermediate | free | search | NumPy, SciPy and plotting in more depth, including signal processing basics. |

## 3. Machine learning fundamentals and statistical learning

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Machine Learning Crash Course](<https://developers.google.com/machine-learning/crash-course>) | Google | course | beginner | free | search | Short practical introduction to loss, generalisation, classification metrics and thresholds. |
| [Intro to Machine Learning](<https://www.kaggle.com/learn/intro-to-machine-learning>) | Kaggle Learn | course | beginner | free | opened | A few hours of guided notebooks on fitting and validating a first model. |
| [scikit-learn User Guide](<https://scikit-learn.org/stable/user_guide.html>) | scikit-learn project | documentation | beginner | free | search (site) | Reference for logistic regression, cross-validation and metrics used in the checker's scoring stage. |
| [An Introduction to Statistical Learning, and The Elements of Statistical Learning (free PDFs listed on the author's publications page)](<https://hastie.su.domains/publications.html>) | Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jerome Friedman, Jonathan Taylor | free online book | intermediate | free PDF from the authors | search | The two standard statistical learning textbooks: resampling, classification and model assessment. |
| [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd edition)](<https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/>) | Aurelien Geron | book | intermediate | paid | search | Practical end-to-end ML projects; useful for workflow habits. A newer PyTorch edition is listed by the same publisher. |
| [CS229: Machine Learning](<https://cs229.stanford.edu/>) | Stanford University | course | intermediate | free to browse (some materials restricted to enrolled students) | search | Mathematical treatment of the core algorithms. |
| [Probabilistic Machine Learning: An Introduction](<https://probml.github.io/pml-book/book1.html>) | Kevin P. Murphy | free online book | advanced | free PDF from the author | search | Modern reference tying probability, decision theory and deep learning together. |
| [Pattern Recognition and Machine Learning](<https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/>) | Christopher M. Bishop | free online book | advanced | free PDF from the publisher | opened | Classic reference; HMM, EM and mixture chapters matter for classic speech models. |

## 4. Deep learning: general, training practice and debugging

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [But what is a Neural Network? (neural networks series)](<https://www.3blue1brown.com/lessons/neural-networks/>) | 3Blue1Brown (Grant Sanderson) | video series | beginner | free | search | Visual explanation of networks, gradient descent and backpropagation. |
| [Neural Networks and Deep Learning](<http://neuralnetworksanddeeplearning.com/>) | Michael Nielsen | free online book | beginner | free to read online | search | Builds a network from scratch and explains why backpropagation works. |
| [Practical Deep Learning for Coders](<https://course.fast.ai/>) | Jeremy Howard, fast.ai | course | beginner | free | search | Top-down course written for programmers; fine-tuning first, theory later. |
| [Neural Networks: Zero to Hero](<https://karpathy.ai/zero-to-hero.html>) | Andrej Karpathy | video series | beginner | free | search | Builds autograd and language models in code, line by line. |
| [MIT 6.S191 Introduction to Deep Learning](<https://introtodeeplearning.com/>) | Alexander Amini, Ava Amini, MIT | course | beginner | free | search | Compact yearly lecture series with labs. |
| [The Little Book of Deep Learning](<https://fleuret.org/francois/lbdl.html>) | Francois Fleuret | free online book | beginner | free PDF from the author | search | Phone-sized summary of the whole field; good for revision. |
| [A Recipe for Training Neural Networks](<https://karpathy.github.io/2019/04/25/recipe/>) | Andrej Karpathy | tutorial | intermediate | free | search | Step-by-step discipline for debugging training: overfit one batch, inspect data, add complexity slowly. |
| [Deep Learning Tuning Playbook](<https://github.com/google-research/tuning_playbook>) | Varun Godbole, George E. Dahl, Justin Gilmer, Christopher J. Shallue, Zachary Nado | tutorial | intermediate | free | search | Systematic method for choosing learning rate, batch size and schedules on a limited GPU budget. |
| [Dive into Deep Learning](<https://d2l.ai/>) | Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola | free online book | intermediate | free to read online | search | Runnable PyTorch code beside the maths for every major architecture. |
| [Understanding Deep Learning](<https://udlbook.github.io/udlbook/>) | Simon J. D. Prince | free online book | intermediate | free PDF from the author | search | Current, clearly illustrated textbook with notebooks. |
| [CS231n: Deep Learning for Computer Vision (course notes)](<https://cs231n.github.io/>) | Stanford University | course | intermediate | free | search | The notes on optimisation, initialisation and babysitting training apply directly to speech models. |
| [NYU Deep Learning (Spring 2021)](<https://atcold.github.io/NYU-DLSP21/>) | Yann LeCun, Alfredo Canziani | course | intermediate | free | search | Lectures with PyTorch practicals from an energy-based viewpoint. |
| [Performance Tuning Guide](<https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>) | PyTorch project | documentation | intermediate | free | search | Mixed precision, data loading and other settings that decide how much fits in a Kaggle session. |
| [Deep Learning](<https://www.deeplearningbook.org/>) | Ian Goodfellow, Yoshua Bengio, Aaron Courville | free online book | advanced | free to read online | search | Theory reference for regularisation, optimisation and sequence modelling. |
| [Deep Learning: Foundations and Concepts](<https://www.bishopbook.com/>) | Christopher M. Bishop, Hugh Bishop | free online book | advanced | free to read online (PDF and print paid) | search | Recent comprehensive textbook covering transformers and probabilistic foundations. |

## 5. Sequence models and Transformers

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Understanding LSTM Networks](<https://colah.github.io/posts/2015-08-Understanding-LSTMs/>) | Christopher Olah | tutorial | beginner | free | search | Clearest picture of recurrent networks, the predecessors of today's speech encoders. |
| [The Unreasonable Effectiveness of Recurrent Neural Networks](<http://karpathy.github.io/2015/05/21/rnn-effectiveness/>) | Andrej Karpathy | tutorial | beginner | free | search | Shows what sequence models learn, with code. |
| [The Illustrated Transformer](<https://jalammar.github.io/illustrated-transformer/>) | Jay Alammar | tutorial | beginner | free | search | Standard visual walkthrough of self-attention, the core of the Conformer encoder. |
| [Let's build GPT: from scratch, in code, spelled out](<https://www.youtube.com/watch?v=kCc8FmEb1nY>) | Andrej Karpathy | video series | beginner | free | search | Writes a transformer in PyTorch from an empty file. |
| [Attention and Augmented Recurrent Neural Networks](<https://distill.pub/2016/augmented-rnns/>) | Chris Olah, Shan Carter (Distill) | tutorial | intermediate | free | search | Interactive explanation of attention mechanisms. |
| [An introduction to sequence-to-sequence learning](<https://lorenlugosch.github.io/posts/2019/02/seq2seq/>) | Loren Lugosch | tutorial | intermediate | free | search | Why attention was introduced, written by a speech researcher. |
| [Transformers from scratch](<https://peterbloem.nl/blog/transformers>) | Peter Bloem | tutorial | intermediate | free | search | Careful derivation of self-attention with PyTorch code. |
| [The Annotated Transformer](<https://nlp.seas.harvard.edu/annotated-transformer/>) | Harvard NLP (Alexander Rush and co-authors) | tutorial | intermediate | free | search | The original paper re-told as working, commented code. |
| [Hugging Face LLM Course (formerly NLP Course)](<https://huggingface.co/learn/llm-course/chapter1/1>) | Hugging Face | course | intermediate | free | opened | Practical use of transformer models, tokenisers and the fine-tuning workflow. |
| [CS224N: Natural Language Processing with Deep Learning](<https://web.stanford.edu/class/cs224n/>) | Stanford University | course | intermediate | free (slides and schedule) | search | Full lecture course on sequence models, attention and transformers. |
| [Sequence Models (Deep Learning Specialization, course 5)](<https://www.coursera.org/learn/nlp-sequence-models>) | Andrew Ng, DeepLearning.AI | course | intermediate | paid subscription (audit options vary) | search | Structured course on RNNs and attention, with a module on speech recognition and audio data. |
| [Formal Algorithms for Transformers (arXiv 2207.09238)](<https://arxiv.org/abs/2207.09238>) | Mary Phuong, Marcus Hutter | paper | advanced | free | opened | Precise pseudocode for every transformer component. |

## 6. Digital signal processing and audio basics

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [But what is the Fourier Transform? A visual introduction](<https://www.3blue1brown.com/lessons/fourier-transforms/>) | 3Blue1Brown (Grant Sanderson) | video series | beginner | free | search | The single idea behind spectrograms, explained visually. |
| [Hugging Face Audio Course, Unit 1: Working with audio data](<https://huggingface.co/learn/audio-course/chapter1/introduction>) | Hugging Face | course | beginner | free | opened | Sampling rate, waveforms, spectrograms and mel features in a few short pages. |
| [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](<https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html>) | Haytham Fayek | tutorial | beginner | free | search | Derives log-mel filter banks step by step in NumPy; these are the model's input features. |
| [Speech Processing course, Module 3: Digital speech signals](<https://speech.zone/courses/speech-processing/module-3-digital-speech-signals/>) | Simon King, University of Edinburgh (speech.zone) | course | beginner | free | search | Videos and notebooks on sampling, the spectrum and the spectrogram for speech. |
| [Preparation Course for Python (PCP) notebooks](<https://www.audiolabs-erlangen.de/resources/MIR/PCP/PCP.html>) | Meinard Muller and co-authors, International Audio Laboratories Erlangen | free online book | beginner | free | search | Notebook course leading from Python basics to the discrete Fourier transform. |
| [Audio Feature Extractions (torchaudio tutorial)](<https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html>) | PyTorch project | tutorial | intermediate | free | search | Spectrogram, mel and MFCC transforms in the same library family the training code uses. |
| [librosa documentation](<https://librosa.org/doc/latest/index.html>) | librosa development team | documentation | intermediate | free | search | Standard Python audio analysis library; handy for inspection, plots and silence splitting. |
| [Introduction to Speech Processing](<https://speechprocessingbook.aalto.fi/>) | Tom Backstrom and co-authors, Aalto University | free online book | intermediate | free to read online | search (site) | Open textbook covering speech production, windowing, mel cepstrum, voice activity detection and recognition. |
| [FMP Notebooks: Python Notebooks for Fundamentals of Music Processing](<https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html>) | Meinard Muller, International Audio Laboratories Erlangen | free online book | intermediate | free | search | Excellent notebooks on the STFT and on dynamic time warping, a relative of forced alignment. |
| [Spectrogram features for audio and speech analysis (arXiv 2603.14917)](<https://arxiv.org/abs/2603.14917>) | Ian McLoughlin and co-authors | paper | intermediate | free | opened | Recent review of spectrogram and mel features and their parameters. |
| [TorchAudio 2.1: Advancing speech recognition, self-supervised learning, and audio processing components for PyTorch (arXiv 2310.17864)](<https://arxiv.org/abs/2310.17864>) | Jeff Hwang and co-authors | paper | intermediate | free | opened | Describes the audio library's feature, alignment and decoding components. |
| [MIT RES.6-007 Signals and Systems](<https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/>) | Alan V. Oppenheim, MIT OpenCourseWare | course | advanced | free | search (site) | Classic lectures for the theory of sampling and linear systems. |

## 7. Speech recognition: classic and modern, with CTC

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Speech and Language Processing (3rd edition draft)](<https://web.stanford.edu/~jurafsky/slp3/>) | Daniel Jurafsky, James H. Martin | free online book | beginner | free PDF from the authors | search | Chapters on phonetics, ASR, CTC, HMMs and edit distance are the best single textbook entry point. |
| [Sequence Modeling with CTC](<https://distill.pub/2017/ctc/>) | Awni Hannun (Distill) | tutorial | beginner | free | search | The best explanation of CTC: alignments, the blank token, the forward algorithm and decoding. |
| [Speech.zone courses](<https://speech.zone/>) | Simon King, University of Edinburgh | course | beginner | free | search (site) | Video-based courses on speech processing and HMM recognition for newcomers. |
| [Hugging Face Audio Course, Unit 3: CTC architectures](<https://huggingface.co/learn/audio-course/chapter3/ctc>) | Hugging Face | course | beginner | free | opened | Short practical explanation of CTC models next to sequence-to-sequence ones. |
| [Hugging Face Audio Course: Automatic speech recognition with a pipeline](<https://huggingface.co/learn/audio-course/chapter2/asr_pipeline>) | Hugging Face | course | beginner | free | opened | Running a pretrained recogniser in a few lines, as a first experiment. |
| [Hugging Face Audio Course, Unit 5: Automatic speech recognition](<https://huggingface.co/learn/audio-course/chapter5/introduction>) | Hugging Face | course | intermediate | free | opened | Choosing, evaluating and fine-tuning an ASR model end to end. |
| [Fine-Tune Wav2Vec2 for English ASR with Transformers](<https://huggingface.co/blog/fine-tune-wav2vec2-english>) | Hugging Face blog | tutorial | intermediate | free | opened | End-to-end CTC fine-tuning walkthrough, including building a vocabulary. |
| [Fine-Tune XLSR-Wav2Vec2 for low-resource ASR with Transformers](<https://huggingface.co/blog/fine-tune-xlsr-wav2vec2>) | Hugging Face blog | tutorial | intermediate | free | opened | The same CTC recipe for a new language with little data. |
| [Boosting Wav2Vec2 with n-grams in Transformers](<https://huggingface.co/blog/wav2vec2-with-ngram>) | Hugging Face blog | tutorial | intermediate | free | opened | CTC beam search with a language model, explained with code. |
| [Making automatic speech recognition work on large files with Wav2Vec2 in Transformers](<https://huggingface.co/blog/asr-chunking>) | Hugging Face blog | tutorial | intermediate | free | opened | Chunking with stride for long recordings with a CTC model. |
| [CTCLoss (PyTorch documentation)](<https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.loss.CTCLoss.html>) | PyTorch project | documentation | intermediate | free | search | Exact input shapes, length arguments and determinism notes for the loss being trained. |
| [Forced Alignment with Wav2Vec2 (torchaudio tutorial)](<https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html>) | PyTorch project | tutorial | intermediate | free | search | Builds the CTC alignment trellis and backtracking by hand; the heart of the checker. |
| [CTC forced alignment API tutorial (torchaudio)](<https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html>) | PyTorch project | tutorial | intermediate | free | search | Library version of forced alignment with token spans and scores. Check the deprecation notice for the installed version. |
| [Sequence-to-sequence learning with Transducers](<https://lorenlugosch.github.io/posts/2020/11/transducer/>) | Loren Lugosch | tutorial | intermediate | free | search | Clearest explanation of the RNN-Transducer as a follow-up to CTC. |
| [CS224S: Spoken Language Processing](<http://web.stanford.edu/class/cs224s/>) | Andrew Maas, Stanford University | course | intermediate | free (slides) | search | Course on modern speech systems, with lectures on CTC and end-to-end ASR. |
| [Corpus Phonetics Tutorial (arXiv 1811.05553)](<https://arxiv.org/abs/1811.05553>) | Eleanor Chodroff | tutorial | intermediate | free | opened | Practical guide to forced alignment and acoustic analysis with Kaldi-based tools. |
| [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](<https://www.cs.toronto.edu/~graves/icml_2006.pdf>) | Alex Graves, Santiago Fernandez, Faustino Gomez, Jurgen Schmidhuber | paper | advanced | free PDF from the author | search | The original CTC paper (ICML 2006; no arXiv id). |
| [Supervised Sequence Labelling with Recurrent Neural Networks](<https://www.cs.toronto.edu/~graves/preprint.pdf>) | Alex Graves | free online book | advanced | free preprint PDF from the author | search | Book-length treatment of CTC by its inventor, with full derivations. |
| [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](<https://ieeexplore.ieee.org/document/18626/>) | Lawrence R. Rabiner | paper | advanced | paid (IEEE) | search | The classic source for forward-backward and Viterbi, which CTC and forced alignment reuse. |
| [The Application of Hidden Markov Models in Speech Recognition](<https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf>) | Mark Gales, Steve Young | paper | advanced | free PDF from the author | search | Book-length review of classic HMM speech recognition. |
| [Speech Recognition with Weighted Finite-State Transducers](<https://cs.nyu.edu/~mohri/pub/hbka.pdf>) | Mehryar Mohri, Fernando Pereira, Michael Riley | paper | advanced | free PDF from the author | search | Theory behind decoding graphs in Kaldi and k2, and behind constrained alignment graphs. |
| [Automatic Speech Recognition (course page with lecture notes)](<https://cs.nyu.edu/~mohri/asr.html>) | Mehryar Mohri, New York University | course | advanced | free | search | Lecture notes on acoustic models, pronunciation models and transducers. |
| [Montreal Forced Aligner and the state of speech-to-text alignment in 2026 (arXiv 2606.18466)](<https://arxiv.org/abs/2606.18466>) | Michael McAuliffe and co-authors | paper | advanced | free | opened | Recent overview of forced alignment by the authors of the Montreal Forced Aligner. |

## 8. Pronunciation assessment and mispronunciation detection

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Use pronunciation assessment (Azure Speech documentation)](<https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment>) | Microsoft | documentation | beginner | free to read | search | Shows how a production system reports accuracy per phoneme, word and utterance. |
| [Automatic Pronunciation Assessment -- A Review (arXiv 2310.13974)](<https://arxiv.org/abs/2310.13974>) | Yassine El Kheir, Ahmed Ali, Shammur Absar Chowdhury | paper | intermediate | free | opened | Survey of methods, datasets and metrics. |
| [speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment (arXiv 2104.01378)](<https://arxiv.org/abs/2104.01378>) | Junbo Zhang and co-authors | paper | intermediate | free | opened | The standard open benchmark, with a GOP baseline. |
| [speechocean762 dataset card](<https://huggingface.co/datasets/mispeech/speechocean762>) | Hugging Face dataset repository (mispeech) | tool | intermediate | free | opened | The benchmark data itself, with phoneme, word and sentence scores. |
| [speechocean762 repository](<https://github.com/jimbozhang/speechocean762>) | Junbo Zhang and co-authors | tool | intermediate | free | search | Corpus description and scoring details. |
| [Kaldi (see the egs/gop_speechocean762 recipe)](<https://github.com/kaldi-asr/kaldi>) | Kaldi project | tool | intermediate | free | search (site) | Reference implementation of goodness-of-pronunciation features and scoring. |
| [L2-ARCTIC: A Non-Native English Speech Corpus](<https://psi.engr.tamu.edu/wp-content/uploads/2018/08/zhao2018interspeech.pdf>) | Guanlong Zhao and co-authors, Texas A&M University | paper | intermediate | free PDF from the authors | search | Non-native English corpus with annotated mispronunciations; the usual detection benchmark. |
| [Phone-level pronunciation scoring and assessment for interactive language learning](<https://mi.eng.cam.ac.uk/~sjy/papers/wiyo00.pdf>) | Silke M. Witt, Steve J. Young | paper | advanced | free PDF from the author (journal version paid) | search | The paper that defined goodness of pronunciation (GOP). |
| [Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment (arXiv 2205.03432)](<https://arxiv.org/abs/2205.03432>) | Yuan Gong and co-authors | paper | advanced | free | opened | GOP features fed to a transformer for phoneme, word and utterance scores (GOPT). |
| [GOPT code repository](<https://github.com/YuanGongND/gopt>) | Yuan Gong and co-authors | tool | advanced | free | search | Working code for GOP feature extraction and scoring. |
| [Segmentation-free Goodness of Pronunciation (arXiv 2507.16838)](<https://arxiv.org/abs/2507.16838>) | Xinwei Cao and co-authors | paper | advanced | free | opened | GOP defined directly on CTC models, with self-alignment and alignment-free variants. |
| [Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge (arXiv 2506.02080)](<https://arxiv.org/abs/2506.02080>) | Aditya Kamlesh Parikh and co-authors | paper | advanced | free | opened | Restricts substitutions in alignment-free GOP using phoneme classes. |
| [Explore wav2vec 2.0 for Mispronunciation Detection](<https://www.isca-archive.org/interspeech_2021/xu21k_interspeech.html>) | Xiaoshuo Xu and co-authors (Interspeech 2021) | paper | advanced | free | search | Self-supervised encoders used for mispronunciation detection. |
| [A Study on Fine-Tuning wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis](<https://www.isca-archive.org/interspeech_2021/peng21e_interspeech.html>) | Linkai Peng and co-authors (Interspeech 2021) | paper | advanced | free | search | CTC phoneme recognition fine-tuned for detection and diagnosis. |
| [Phonological Level wav2vec2-based Mispronunciation Detection and Diagnosis Method (arXiv 2311.07037)](<https://arxiv.org/abs/2311.07037>) | Mostafa Shahin, Julien Epps, Beena Ahmed | paper | advanced | free | opened | Detecting errors at the level of articulatory features rather than whole phonemes. |
| [Automated detection of pronunciation errors in non-native English speech employing deep learning (arXiv 2209.06265)](<https://arxiv.org/abs/2209.06265>) | Daniel Korzekwa | paper | advanced | free | opened | Thesis-length treatment, including generating synthetic errors for training. |
| [Towards a Unified Benchmark for Arabic Pronunciation Assessment: Quranic Recitation as Case Study (arXiv 2506.07722)](<https://arxiv.org/abs/2506.07722>) | Yassine El Kheir and co-authors | paper | advanced | free | opened | Phoneme set, pipeline and public test set for Quran mispronunciation detection. |
| [Iqra'Eval: A Shared Task on Qur'anic Pronunciation Assessment](<https://aclanthology.org/2025.arabicnlp-sharedtasks.61>) | Yassine El Kheir and co-authors (ArabicNLP 2025) | paper | advanced | free | search | Shared task with open data and baselines for the same problem. |
| [IqraEval organisation page (datasets and baselines)](<https://huggingface.co/IqraEval>) | IqraEval organisers | tool | advanced | free | opened | Training and test sets and leaderboards for Quran mispronunciation detection. |
| [IQRA 2026: Interspeech Challenge on Automatic Pronunciation Assessment for Modern Standard Arabic (arXiv 2603.29087)](<https://arxiv.org/abs/2603.29087>) | Yassine El Kheir and co-authors | paper | advanced | free | opened | Follow-up challenge with real mispronunciation data. |
| [Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning (arXiv 2509.00094)](<https://arxiv.org/abs/2509.00094>) | Abdullah Abdelfattah, Mahmoud I. Khalil, Hazem Abbas | paper | advanced | free | opened | A phonetic script for Quran recitation and a model for detecting recitation errors; among the closest published work to this project. |

## 9. Phonetics and phonology, including Arabic

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Interactive IPA Chart](<https://www.internationalphoneticassociation.org/IPAcharts/IPA_charts_TI/IPA_charts_TI.html>) | International Phonetic Association | tool | beginner | free | search | The reference chart of speech sounds with audio; needed to define a phoneme inventory. |
| [Seeing Speech](<https://www.seeingspeech.ac.uk/>) | University of Glasgow and partner universities | tool | beginner | free | search | MRI and ultrasound videos of each sound being articulated. |
| [Essentials of Linguistics (2nd edition)](<https://ecampusontario.pressbooks.pub/essentialsoflinguistics2/>) | Catherine Anderson and co-authors, eCampusOntario | free online book | beginner | free (open textbook) | search (site) | Open textbook with clear chapters on phonetics and phonology. |
| [Standard Arabic phonology](<https://en.wikipedia.org/wiki/Standard_Arabic_phonology>) | Wikipedia contributors | community | beginner | free | search | Quick overview of Arabic consonants, emphatics and vowel length, with references. |
| [Help:IPA/Arabic](<https://en.wikipedia.org/wiki/Help:IPA/Arabic>) | Wikipedia contributors | community | beginner | free | search | Letter-to-IPA table for Arabic. |
| [Links to Phonetics Resources](<https://www.internationalphoneticassociation.org/content/links-phonetics-resources>) | International Phonetic Association | community | intermediate | free | search | Curated list of charts, recordings and articulation videos. |
| [Praat: doing phonetics by computer](<https://www.fon.hum.uva.nl/praat/>) | Paul Boersma, David Weenink, University of Amsterdam | tool | intermediate | free | search (site) | Standard tool to view spectrograms, formants and durations, e.g. to check vowel-length labels. |
| [UCLA Phonetics Lab Archive](<http://archive.phonetics.ucla.edu/>) | UCLA Department of Linguistics | tool | intermediate | free | search (site) | Recordings and word lists across languages. |
| [PHOIBLE 2.0](<https://phoible.org/>) | Steven Moran, Daniel McCloy (eds.) | tool | intermediate | free | search | Database of phoneme inventories with distinctive features; useful when fixing the Arabic token set. |
| [Arabic (Illustrations of the IPA)](<https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/arabic/D54098F7BBE7821861A80617B872FA24>) | Robin Thelwall, M. Akram Sa'adeddin | paper | intermediate | paid or institutional (Cambridge University Press) | search | The IPA's own short description of Arabic sounds. |
| [The Phonology and Morphology of Arabic](<https://global.oup.com/academic/product/the-phonology-and-morphology-of-arabic-9780199226696>) | Janet C. E. Watson | book | advanced | paid | search | Standard academic reference on Arabic sound structure. |

## 10. Tajweed and Quran recitation science

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Tanzil documents (text types, licence and usage)](<https://tanzil.net/docs/>) | Tanzil project | documentation | beginner | free | opened | Explains the verified Uthmani and simple Quran texts used as the basis for expected token sequences. |
| [Tanzil Quran text download](<https://tanzil.net/download/>) | Tanzil project | tool | beginner | free (see the text licence terms) | opened | Download page for the verified Quran text in several orthographies. |
| [EveryAyah](<https://everyayah.com/>) | EveryAyah.com | tool | beginner | free | opened | Verse-by-verse recitation audio from many reciters. |
| [Quran.com](<https://quran.com/>) | Quran Foundation | tool | beginner | free | search (site) | Text with word-by-word audio for reference listening. |
| [Qaidah Nooraniya (original edition)](<https://islamicbookstore.com/b12072.html>) | Noor Muhammad Haqqani; Al Furqaan (bookseller listing) | book | beginner | paid | search | The beginner primer for letters, vowels and joining rules. No free edition from the publisher could be confirmed. |
| [A Classical Primer on Tajwid (Tuhfat al-Atfal with points from the Muqaddimah of Ibn al-Jazari)](<https://rissc.jo/wp-content/uploads/2018/11/A_Classical_Primer_on_Tajwid_EN.pdf>) | Sulayman al-Jamzuri; Royal Aal al-Bayt Institute for Islamic Thought | free online book | beginner | free PDF from the publisher | search | English translation of the classical beginner poem on noon, meem and madd rules. |
| [Tuhfat al-Atfal: structure and topics of the poem](<https://surahquran.com/Tajweed/tohfat-alatfal-en.html>) | Sulayman al-Jamzuri (page by surahquran.com) | tutorial | beginner | free | search | Outline of the poem with links to the Arabic text. |
| [Tajweed Rules of the Qur'an (3-part set)](<https://darussalam.com/tajweed-rules-of-the-quran-3-part-set>) | Kareema Carol Czerepinski; Dar Al-Khair | book | intermediate | paid | search | The standard English tajweed textbook series, including makharij and sifat. |
| [Murshid al-Qari, Part 2: Commentary of the Tuhfah al-Atfal](<https://duai.org.za/wp-content/uploads/2023/02/Murshid-al-Qari-Part-2-Commentary-of-the-Tuḥfah-al-Aṭfal-by-Jamzuri.pdf>) | Dar al-Ulum al-Arabiyyah al-Islamiyyah (DUAI) | free online book | intermediate | free PDF from the publisher | search | Word-by-word English commentary on the beginner poem. |
| [Al-Muqaddimah al-Jazariyyah: structure and chapters of the poem](<https://surahquran.com/Tajweed/aljazariah-en.html>) | Ibn al-Jazari (page by surahquran.com) | tutorial | intermediate | free | search | Outline of the classical poem on makharij, sifat and the main rules, with the Arabic text. |
| [Jazariyah Sharh (English commentary on the Jazariyyah)](<https://al-tanzil.co.za/books/jazariyah-sharh.pdf>) | Muhammad Saleem Gaibie | free online book | intermediate | free PDF (hosted at al-tanzil.co.za; confirm terms before redistribution) | search | Accessible English commentary on the poem that defines articulation points and attributes. |
| [The Jazariyyah Simplified](<https://bookstore.zaytuna.edu/products/the-jazariyyah-simplified>) | Ibn al-Jazari; Zaytuna College bookstore | book | intermediate | paid | search | Translation and commentary of the Jazariyyah with the Arabic text. |
| [Makharij al-huruf and sifat al-huruf charts (Arabic)](<https://jarirbooksusa.com/72097.html>) | Ayman Rushdi Suwayd; Dar al-Ghawthani (bookseller listing) | book | intermediate | paid | search | Illustrated reference charts for articulation points and letter attributes. |
| [quran-tajweed (tajweed annotation for the Quran text)](<https://github.com/cpfair/quran-tajweed>) | cpfair | tool | intermediate | free (open data; check the repository licence) | search (site) | Machine-readable tajweed rule positions for every verse; a check on rule-derived token sequences. |
| [The Quranic Arabic Corpus](<https://corpus.quran.com/>) | Kais Dukes, University of Leeds | tool | intermediate | free (GNU licence with terms of use) | search (site) | Word-by-word morphology and a phonetic transcription of the whole text. |
| [King Fahd Glorious Quran Printing Complex](<https://qurancomplex.gov.sa/en/>) | King Fahd Complex, Madinah | tool | intermediate | free | search | Official source of the Madinah Mushaf, with digital copies in several narrations. |
| [Quran-MD: A Fine-Grained Multilingual Multimodal Dataset of the Quran (arXiv 2601.17880)](<https://arxiv.org/abs/2601.17880>) | Muhammad Umar Salman, Mohammad Areeb Qazi, Mohammed Talha Alam | paper | advanced | free | opened | Verse- and word-level text and audio dataset, with a survey of other Quran resources. |

## 11. Toolkits and official documentation

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [PyTorch Tutorials](<https://docs.pytorch.org/tutorials/index.html>) | PyTorch project | tutorial | beginner | free | search | Official worked examples, including audio and deployment. |
| [PyTorch documentation](<https://docs.pytorch.org/docs/stable/index.html>) | PyTorch project | documentation | beginner | free | search (site) | Reference for every tensor and module call. |
| [Hugging Face Audio Course](<https://huggingface.co/learn/audio-course/chapter0/introduction>) | Hugging Face | course | beginner | free | opened | Complete practical course on audio transformers, ASR fine-tuning and evaluation. |
| [Hugging Face Datasets documentation](<https://huggingface.co/docs/datasets/index>) | Hugging Face | documentation | beginner | free | opened | Loading, streaming and publishing audio datasets. |
| [Kaggle Notebooks documentation](<https://www.kaggle.com/docs/notebooks>) | Kaggle | documentation | beginner | free | opened | Sessions, saving outputs and limits on the free training platform. |
| [Efficient GPU Usage Tips](<https://www.kaggle.com/docs/efficient-gpu-usage>) | Kaggle | documentation | beginner | free | opened | How to make the weekly free GPU quota last. |
| [torchaudio documentation](<https://docs.pytorch.org/audio/stable/index.html>) | PyTorch project | documentation | intermediate | free | search (site) | Audio transforms and alignment functions. The library is in maintenance mode, so check version notes. |
| [Create an audio dataset (Datasets documentation)](<https://huggingface.co/docs/datasets/audio_dataset>) | Hugging Face | documentation | intermediate | free | opened | Packaging audio and transcripts as a shareable dataset. |
| [Automatic speech recognition task guide (Transformers)](<https://huggingface.co/docs/transformers/tasks/asr>) | Hugging Face | documentation | intermediate | free | opened | Reference CTC fine-tuning recipe to compare against the NeMo one. |
| [Wav2Vec2 model documentation (Transformers)](<https://huggingface.co/docs/transformers/model_doc/wav2vec2>) | Hugging Face | documentation | intermediate | free | opened | API of the most common CTC model family, including phoneme tokenisers. |
| [NVIDIA NeMo: Automatic Speech Recognition (ASR) documentation](<https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html>) | NVIDIA | documentation | intermediate | free | search | Official guide to NeMo ASR models, checkpoints and fine-tuning. |
| [NeMo ASR models reference (Conformer, Fast-Conformer, cache-aware streaming)](<https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html>) | NVIDIA | documentation | intermediate | free | search | Describes the exact encoder family being fine-tuned and its config files. |
| [NeMo ASR configuration files (NeMo-Speech documentation)](<https://docs.nvidia.com/nemo/speech/latest/asr/configs.html>) | NVIDIA | documentation | intermediate | free | search | Meaning of each section of the training YAML: data, preprocessor, augmentation, encoder, optimiser. |
| [NeMo tutorials index](<https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html>) | NVIDIA | tutorial | intermediate | free | search | Links to notebooks for CTC fine-tuning on a new language, streaming and more. |
| [NeMo repository (tutorials/asr notebooks, e.g. ASR confidence estimation)](<https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_Confidence_Estimation.ipynb>) | NVIDIA | tool | intermediate | free | search | Source, example configs and notebooks; issues are where maintainers answer questions. |
| [NeMo Forced Aligner (NFA) documentation](<https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html>) | NVIDIA | documentation | intermediate | free | search | Token, word and segment timestamps from NeMo CTC models. |
| [nvidia/parakeet-ctc-0.6b model card](<https://huggingface.co/nvidia/parakeet-ctc-0.6b>) | NVIDIA | tool | intermediate | free | opened | A FastConformer CTC checkpoint with usage and training details. |
| [nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 model card](<https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0>) | NVIDIA | tool | intermediate | free | opened | An Arabic FastConformer checkpoint and its data and licence notes. |
| [PyTorch Lightning documentation](<https://lightning.ai/docs/pytorch/stable/>) | Lightning AI | documentation | intermediate | free | search (site) | NeMo models are Lightning modules; trainer flags, checkpoints and precision live here. |
| [Hydra documentation](<https://hydra.cc/docs/intro/>) | Meta Open Source | documentation | intermediate | free | search (site) | NeMo configs are Hydra and OmegaConf YAML; explains overrides. |
| [sherpa-onnx documentation](<https://k2-fsa.github.io/sherpa/onnx/index.html>) | k2-fsa (Next-gen Kaldi) | documentation | intermediate | free | search | On-device speech runtime with mobile bindings. |
| [k2-fsa organisation page (exported models and demos)](<https://huggingface.co/k2-fsa>) | k2-fsa (Next-gen Kaldi) | tool | intermediate | free | opened | Ready-made ONNX models and demo spaces to test the phone runtime. |
| [SpeechBrain documentation and tutorials](<https://speechbrain.readthedocs.io/en/latest/index.html>) | SpeechBrain project | documentation | intermediate | free | search | Readable PyTorch speech toolkit with from-scratch tutorials. |
| [Montreal Forced Aligner documentation](<https://montreal-forced-aligner.readthedocs.io/en/latest/>) | Michael McAuliffe and contributors | documentation | intermediate | free | search | The standard HMM-based aligner; a useful independent check on neural alignments. |
| [Silero VAD](<https://github.com/snakers4/silero-vad/releases>) | Silero Team | tool | intermediate | free | search | Small voice activity detector with ONNX export, for pause-based segmentation. |
| [ESPnet documentation](<https://espnet.github.io/espnet/>) | ESPnet project | documentation | advanced | free | search (site) | Research toolkit with many reference recipes. |
| [icefall documentation](<https://k2-fsa.github.io/icefall/>) | k2-fsa (Next-gen Kaldi) | documentation | advanced | free | search | Recipes for CTC, transducer and Zipformer models that export to sherpa-onnx. |
| [k2 documentation (core concepts)](<https://k2-fsa.github.io/k2/core_concepts/index.html>) | k2-fsa (Next-gen Kaldi) | documentation | advanced | free | search | Differentiable finite-state graphs; a way to express constrained alignment topologies. |
| [Kaldi documentation](<https://kaldi-asr.org/doc/about.html>) | Kaldi project | documentation | advanced | free | search | Classic toolkit; its docs explain HMM alignment and lattices in detail. |

## 12. Data work: datasets, labelling, label noise

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Introduction to Data-Centric AI](<https://dcai.csail.mit.edu/>) | MIT (course staff) | course | beginner | free | search | Lectures and labs on label errors, dataset curation and data-driven evaluation. |
| [Label Studio documentation](<https://labelstud.io/guide/get_started.html>) | HumanSignal | documentation | beginner | free (open-source edition) | search | Reference design for audio labelling interfaces and review workflows. |
| [A Complete Guide to Audio Datasets](<https://huggingface.co/blog/audio-datasets>) | Hugging Face blog | tutorial | beginner | free | opened | Loading, streaming, resampling and comparing speech datasets. |
| [Audio Dataset (Hub documentation)](<https://huggingface.co/docs/hub/datasets-audio>) | Hugging Face | documentation | beginner | free | opened | Folder and metadata layout for hosting an audio dataset. |
| [Datasheets for Datasets (arXiv 1803.09010)](<https://arxiv.org/abs/1803.09010>) | Timnit Gebru and co-authors | paper | beginner | free | opened | Checklist for documenting how a dataset was collected and what it may be used for. |
| ["Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI](<https://research.google/pubs/everyone-wants-to-do-the-model-work-not-the-data-work-data-cascades-in-high-stakes-ai/>) | Nithya Sambasivan and co-authors, Google Research | paper | beginner | free | search | Shows how small data problems compound downstream. |
| [Data Cascades in Machine Learning](<https://research.google/blog/data-cascades-in-machine-learning/>) | Google Research blog | tutorial | beginner | free | search | Short readable summary of the data cascades study. |
| [Common Voice](<https://commonvoice.mozilla.org/>) | Mozilla | tool | beginner | free | search (site) | Model of a consented, validated, openly licensed speech corpus with contribution guidelines. |
| [Common Voice: A Massively-Multilingual Speech Corpus (arXiv 1912.06670)](<https://arxiv.org/abs/1912.06670>) | Rosana Ardila and co-authors | paper | intermediate | free | opened | Describes crowd validation of recordings. |
| [cleanlab documentation](<https://docs.cleanlab.ai/>) | Cleanlab | documentation | intermediate | free (open source) | search (site) | Library for finding likely label errors from model outputs. |
| [Confident Learning: Estimating Uncertainty in Dataset Labels (arXiv 1911.00068)](<https://arxiv.org/abs/1911.00068>) | Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang | paper | intermediate | free | opened | Principled way to find mislabelled examples. |
| [Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks (arXiv 2103.14749)](<https://arxiv.org/abs/2103.14749>) | Curtis G. Northcutt, Anish Athalye, Jonas Mueller | paper | intermediate | free | opened | Why test labels must be audited before trusting small differences. |
| [Survey Article: Inter-Coder Agreement for Computational Linguistics](<https://aclanthology.org/J08-4004/>) | Ron Artstein, Massimo Poesio | paper | intermediate | free | search | How to measure agreement between labellers (kappa, alpha). |
| [Librispeech: An ASR Corpus Based on Public Domain Audio Books](<https://www.danielpovey.com/files/2015_icassp_librispeech.pdf>) | Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur | paper | intermediate | free PDF from the author | search | Describes aligning long recordings to known text and filtering bad segments. |
| [Learning from Flawed Data: Weakly Supervised Automatic Speech Recognition](<https://www.danielpovey.com/files/2023_asru_learning_from_flawed_data.pdf>) | Dongji Gao and co-authors | paper | advanced | free PDF from an author | search | Training with CTC-style losses when transcripts contain errors. |
| [MLS: A Large-Scale Multilingual Dataset for Speech Research (arXiv 2012.03411)](<https://arxiv.org/abs/2012.03411>) | Vineel Pratap and co-authors | paper | advanced | free | opened | Segmentation and transcript-matching pipeline for read speech. |
| [CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition (arXiv 2007.09127)](<https://arxiv.org/abs/2007.09127>) | Ludwig Kürzinger and co-authors | paper | advanced | free | opened | Aligning long audio to text with a CTC model and scoring segment confidence. |
| [Learning from Noisy Labels with Deep Neural Networks: A Survey (arXiv 2007.08199)](<https://arxiv.org/abs/2007.08199>) | Hwanjun Song and co-authors | paper | advanced | free | opened | Survey of training methods that tolerate label noise. |

## 13. Evaluation and statistics for ML

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Classification: ROC and AUC (Machine Learning Crash Course)](<https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc>) | Google | tutorial | beginner | free | search | Short visual introduction to thresholds, ROC and AUC. |
| [Hugging Face Audio Course: Evaluation metrics for ASR](<https://huggingface.co/learn/audio-course/chapter5/evaluation>) | Hugging Face | course | beginner | free | opened | Word and character error rate, and text normalisation before scoring. |
| [WER metric (Evaluate)](<https://huggingface.co/spaces/evaluate-metric/wer>) | Hugging Face | tool | beginner | free | opened | Definition and reference implementation of word error rate. |
| [jiwer](<https://github.com/jitsi/jiwer>) | Jitsi and contributors | tool | beginner | free | search | Word, character and token error rate from minimum edit distance. |
| [Metrics and scoring: quantifying the quality of predictions](<https://scikit-learn.org/stable/modules/model_evaluation.html>) | scikit-learn project | documentation | beginner | free | search | Reference for ROC, DET, precision-recall and scoring functions. |
| [Probability calibration](<https://scikit-learn.org/stable/modules/calibration.html>) | scikit-learn project | documentation | intermediate | free | search | Reliability diagrams, sigmoid and isotonic calibration for turning scores into probabilities. |
| [An introduction to ROC analysis](<https://doi.org/10.1016/j.patrec.2005.10.010>) | Tom Fawcett | paper | intermediate | paid (Elsevier) | search (site) | The standard ROC tutorial, including operating points and averaging. |
| [scipy.stats.bootstrap documentation](<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html>) | SciPy project | documentation | intermediate | free | search | Ready-made percentile and BCa bootstrap intervals. |
| [ConfidenceIntervals](<https://github.com/luferrer/ConfidenceIntervals>) | Luciana Ferrer and co-author | tool | intermediate | free | search | Bootstrap confidence intervals for any metric, with support for resampling by condition such as speaker. |
| [Analysis and Comparison of Classification Metrics (arXiv 2209.05355)](<https://arxiv.org/abs/2209.05355>) | Luciana Ferrer | paper | intermediate | free | opened | Argues for cost-based and proper scoring metrics over ad hoc ones. |
| [The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing](<https://aclanthology.org/P18-1128.pdf>) | Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart | paper | intermediate | free | search | Which significance test to use for which metric. |
| [Appendix - Recommended Statistical Significance Tests for NLP Tasks (arXiv 1809.01448)](<https://arxiv.org/abs/1809.01448>) | Rotem Dror, Roi Reichart | paper | intermediate | free | opened | Companion table mapping tasks and metrics to tests. |
| [Statistical Significance Tests for Machine Translation Evaluation](<https://aclanthology.org/W04-3250/>) | Philipp Koehn | paper | intermediate | free | search | The paired bootstrap test widely reused for comparing two systems. |
| [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning (arXiv 1811.12808)](<https://arxiv.org/abs/1811.12808>) | Sebastian Raschka | paper | intermediate | free | opened | Hold-out, cross-validation and bootstrap explained with their pitfalls. |
| [Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing](<https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>) | Yoav Benjamini, Yosef Hochberg | paper | advanced | paid or institutional (Wiley) | search (site) | The original false discovery rate procedure for multiple testing. |
| [On Calibration of Modern Neural Networks (arXiv 1706.04599)](<https://arxiv.org/abs/1706.04599>) | Chuan Guo and co-authors | paper | advanced | free | opened | Shows neural scores are over-confident and introduces temperature scaling. |
| [Statistical Testing on ASR Performance via Blockwise Bootstrap (arXiv 1912.09508)](<https://arxiv.org/abs/1912.09508>) | Zhe Liu, Fuchun Peng | paper | advanced | free | opened | Why resampling must respect correlated utterances such as same-speaker blocks. |
| [Modeling Dependent Structure for Utterances in ASR Evaluation (arXiv 2209.05281)](<https://arxiv.org/abs/2209.05281>) | Zhe Liu, Fuchun Peng | paper | advanced | free | opened | Extends blockwise bootstrap intervals for error rates with dependent utterances. |
| [With Little Power Comes Great Responsibility (arXiv 2010.06595)](<https://arxiv.org/abs/2010.06595>) | Dallas Card and co-authors | paper | advanced | free | opened | Statistical power: how large a test set must be to detect a given improvement. |

## 14. ML engineering

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [The Good Research Code Handbook](<https://goodresearch.dev/testing.html>) | Patrick Mineault | free online book | beginner | free | search | Project layout, testing and documentation for research code. |
| [Cookiecutter Data Science](<https://cookiecutter-data-science.drivendata.org/>) | DrivenData | tool | beginner | free | search | A widely used standard project structure. |
| [pytest: Get Started](<https://docs.pytest.org/en/stable/getting-started.html>) | pytest-dev | documentation | beginner | free | search | Test framework for data and model code. |
| [Visualizing Models, Data, and Training with TensorBoard](<https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>) | PyTorch project | tutorial | beginner | free | search | Simplest built-in experiment logging. |
| [The Full Stack (Full Stack Deep Learning)](<https://fullstackdeeplearning.com/>) | Sergey Karayev, Josh Tobin, Charles Frye and co-instructors | course | intermediate | free | search | Course on shipping ML: troubleshooting, testing, data management, deployment. |
| [Made With ML](<https://madewithml.com/courses/foundations/>) | Goku Mohandas (Anyscale) | course | intermediate | free | search | Engineering practices for ML projects, with code. |
| [Weights & Biases documentation: Experiments overview](<https://docs.wandb.ai/models/track>) | Weights & Biases | documentation | intermediate | free tier for individuals | search | Experiment tracking that works from Kaggle notebooks and with Lightning. |
| [MLflow documentation](<https://mlflow.org/docs/latest/>) | MLflow project | documentation | intermediate | free (open source) | search | Open-source experiment tracking and model registry. |
| [Get Started with DVC](<https://doc.dvc.org/start>) | DVC project | documentation | intermediate | free (open source) | search | Versioning datasets and pipelines alongside Git. |
| [Reproducibility (PyTorch notes)](<https://docs.pytorch.org/docs/main/notes/randomness.html>) | PyTorch project | documentation | intermediate | free | search | Seeds and deterministic settings for repeatable runs. |
| [Hidden Technical Debt in Machine Learning Systems](<https://proceedings.neurips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>) | D. Sculley and co-authors | paper | intermediate | free | search | Why data dependencies and glue code become the main maintenance cost. |
| [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](<https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/>) | Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley | paper | intermediate | free | search | Checklist of tests for data, model and infrastructure. |
| [Data Validation for Machine Learning](<https://research.google/pubs/data-validation-for-machine-learning/>) | Google Research | paper | intermediate | free | search | Schema and distribution checks on data before training. |
| [Designing Machine Learning Systems](<https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/>) | Chip Huyen | book | intermediate | paid | search | Book on data, evaluation and deployment decisions as one system. |

## 15. Model compression and on-device deployment

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [ONNX Concepts](<https://onnx.ai/onnx/intro/concepts.html>) | ONNX project | documentation | beginner | free | search | Concepts of the exchange format: graphs, operators and opsets, plus viewing graphs. |
| [Export a PyTorch model to ONNX (tutorial)](<https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html>) | PyTorch project | tutorial | beginner | free | search | First export, validation and comparison against ONNX Runtime. |
| [torch.onnx documentation](<https://docs.pytorch.org/docs/main/onnx.html>) | PyTorch project | documentation | intermediate | free | search | Official export path from PyTorch to ONNX. |
| [Exporting NeMo Models](<https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/core/export.html>) | NVIDIA | documentation | intermediate | free | search | How NeMo models export to ONNX, including the cache-support option. This is the 24.09 page; check the latest version. |
| [Quantize ONNX models](<https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>) | ONNX Runtime (Microsoft) | documentation | intermediate | free | search | Dynamic and static 8-bit quantisation with calibration. |
| [ONNX Runtime: Deploy on mobile](<https://onnxruntime.ai/docs/tutorials/mobile/>) | ONNX Runtime (Microsoft) | documentation | intermediate | free | search | Android and iOS packaging, execution providers and the ORT format. |
| [Mobile image recognition on Android (ONNX Runtime tutorial)](<https://onnxruntime.ai/docs/tutorials/mobile/deploy-android.html>) | ONNX Runtime (Microsoft) | tutorial | intermediate | free | search | Complete export, quantise and deploy walkthrough for a phone. |
| [sherpa-onnx Flutter documentation](<https://k2-fsa.github.io/sherpa/onnx/flutter/index.html>) | k2-fsa (Next-gen Kaldi) | documentation | intermediate | free | search (site) | Flutter examples and prebuilt apps for the on-device runtime. |
| [sherpa_onnx package](<https://pub.dev/packages/sherpa_onnx>) | k2-fsa (Next-gen Kaldi) | tool | intermediate | free | opened | The Dart and Flutter package for the runtime. |
| [Bind to native code using FFI](<https://docs.flutter.dev/interop/bind-native-code>) | Flutter project | documentation | intermediate | free | search | How a Flutter app packages and calls a native inference library. |
| [Interact with native memory using dart:ffi](<https://docs.flutter.dev/interop/dart-ffi>) | Flutter project | documentation | intermediate | free | search | Pointers, buffers and threading rules when passing audio to native code. |
| [ExecuTorch documentation](<https://docs.pytorch.org/executorch/stable/index.html>) | PyTorch project | documentation | intermediate | free | search | PyTorch's own on-device runtime, another export route. |
| [Quantization Overview (ExecuTorch)](<https://docs.pytorch.org/executorch/stable/quantization-overview.html>) | PyTorch project | documentation | intermediate | free | search | Prepare, calibrate and convert flow for post-training quantisation. |
| [LiteRT documentation](<https://ai.google.dev/edge/litert>) | Google AI Edge | documentation | intermediate | free | search (site) | Google's on-device runtime (formerly TensorFlow Lite), an alternative target. |
| [Guide to Core ML Tools](<https://apple.github.io/coremltools/docs-guides/>) | Apple | documentation | intermediate | free | search (site) | Converting and compressing models for Apple devices. |
| [Hugging Face Optimum documentation](<https://huggingface.co/docs/optimum/index>) | Hugging Face | documentation | intermediate | free | opened | Tools for ONNX export and optimisation of transformer models. |
| [Quantization with Optimum and ONNX Runtime](<https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization>) | Hugging Face | documentation | intermediate | free | opened | Worked dynamic and static quantisation examples. |
| [Distilling the Knowledge in a Neural Network (arXiv 1503.02531)](<https://arxiv.org/abs/1503.02531>) | Geoffrey Hinton, Oriol Vinyals, Jeff Dean | paper | intermediate | free | opened | The original knowledge distillation paper. |
| [MIT 6.5940 TinyML and Efficient Deep Learning Computing](<https://hanlab.mit.edu/courses/2024-fall-65940>) | Song Han, MIT | course | advanced | free | search | Lectures on pruning, quantisation, distillation and on-device inference. |
| [A White Paper on Neural Network Quantization (arXiv 2106.08295)](<https://arxiv.org/abs/2106.08295>) | Markus Nagel and co-authors | paper | advanced | free | opened | Practical guide to post-training quantisation and quantisation-aware training with debugging advice. |
| [A Survey of Quantization Methods for Efficient Neural Network Inference (arXiv 2103.13630)](<https://arxiv.org/abs/2103.13630>) | Amir Gholami and co-authors | paper | advanced | free | opened | Broad survey of quantisation methods. |
| [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (arXiv 1712.05877)](<https://arxiv.org/abs/1712.05877>) | Benoit Jacob and co-authors | paper | advanced | free | opened | Defines the 8-bit affine scheme used by mobile runtimes. |
| [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding (arXiv 1510.00149)](<https://arxiv.org/abs/1510.00149>) | Song Han, Huizi Mao, William J. Dally | paper | advanced | free | opened | Classic pruning-plus-quantisation pipeline. |
| [DistilHuBERT: Speech Representation Learning by Layer-wise Distillation of Hidden-unit BERT (arXiv 2110.01900)](<https://arxiv.org/abs/2110.01900>) | Heng-Jui Chang, Shu-wen Yang, Hung-yi Lee | paper | advanced | free | opened | Distilling a speech encoder into a much smaller one. |
| [Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling (arXiv 2311.00430)](<https://arxiv.org/abs/2311.00430>) | Sanchit Gandhi, Patrick von Platen, Alexander M. Rush | paper | advanced | free | opened | Distillation of an ASR model using pseudo-labels. |

## 16. Streaming and real-time speech recognition

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [An All-Neural On-Device Speech Recognizer](<https://research.google/blog/an-all-neural-on-device-speech-recognizer/>) | Google Research blog | tutorial | beginner | free | search | Readable account of why streaming on-device recognition uses transducers. |
| [Online ASR with Emformer RNN-T (torchaudio tutorial)](<https://docs.pytorch.org/audio/stable/tutorials/online_asr_tutorial.html>) | PyTorch project | tutorial | intermediate | free | search | Shows chunked streaming inference with state carried between chunks. |
| [sherpa-onnx pre-trained models](<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html>) | k2-fsa (Next-gen Kaldi) | documentation | intermediate | free | search | Lists the streaming and non-streaming model types the phone runtime accepts. |
| [icefall: Streaming ASR introduction](<https://icefall.readthedocs.io/en/latest/recipes/Streaming-ASR/introduction.html>) | k2-fsa (Next-gen Kaldi) | documentation | intermediate | free | search | How streaming models are trained and decoded in the k2 ecosystem. |
| [WeNet](<https://github.com/wenet-e2e/wenet>) | WeNet community | tool | advanced | free | search | Production-oriented toolkit built around unified streaming and non-streaming models. |
| [Streaming End-to-end Speech Recognition For Mobile Devices (arXiv 1811.06621)](<https://arxiv.org/abs/1811.06621>) | Yanzhang He and co-authors | paper | advanced | free | opened | The reference on-device streaming transducer system. |
| [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss (arXiv 2002.02562)](<https://arxiv.org/abs/2002.02562>) | Qian Zhang and co-authors | paper | advanced | free | opened | Limited-context attention for streaming. |
| [Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition (arXiv 2010.10759)](<https://arxiv.org/abs/2010.10759>) | Yangyang Shi and co-authors | paper | advanced | free | opened | Chunked attention with memory for low-latency streaming. |
| [Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition (arXiv 2012.05481)](<https://arxiv.org/abs/2012.05481>) | Binbin Zhang and co-authors | paper | advanced | free | opened | Dynamic chunk training so one model serves both modes. |
| [Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition (arXiv 2312.17279)](<https://arxiv.org/abs/2312.17279>) | Vahid Noroozi and co-authors | paper | advanced | free | opened | Cache-aware streaming FastConformer, the streaming form of the encoder in use. |
| [Pruned RNN-T for fast, memory-efficient ASR training (arXiv 2206.13236)](<https://arxiv.org/abs/2206.13236>) | Fangjun Kuang and co-authors | paper | advanced | free | opened | Makes transducer training affordable on small GPUs. |

## 17. Licensing and ethics of data

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Creative Commons Frequently Asked Questions](<https://creativecommons.org/faq/>) | Creative Commons | documentation | beginner | free | search | Answers on attribution, NonCommercial and combining works; needed to decide which recordings can be used. |
| [Reusing CC-Licensed Content](<https://creativecommons.org/reusing-cc-licensed-content/>) | Creative Commons | documentation | beginner | free | search | What a reuser must do under each licence element. |
| [Attribution 4.0 International deed (CC BY 4.0)](<https://creativecommons.org/licenses/by/4.0/deed.en>) | Creative Commons | documentation | beginner | free | search | The plain-language summary of the most common open data licence. |
| [Recommended practices for attribution](<https://wiki.creativecommons.org/Recommended_practices_for_attribution>) | Creative Commons | documentation | beginner | free | search | How to credit sources correctly (title, author, source, licence) in a credits file. |
| [Choose a License: appendix (licence comparison table)](<https://choosealicense.com/appendix/>) | GitHub | documentation | beginner | free | search | Side-by-side permissions and conditions of software and data licences. |
| [SPDX License List](<https://spdx.org/licenses/>) | Linux Foundation SPDX project | documentation | beginner | free | search | Standard identifiers for recording the licence of each dependency and dataset. |
| [Dataset Cards](<https://huggingface.co/docs/hub/datasets-cards>) | Hugging Face | documentation | beginner | free | opened | Standard format for stating licence, provenance and limits of a dataset. |
| [Model Cards](<https://huggingface.co/docs/hub/model-cards>) | Hugging Face | documentation | beginner | free | opened | Standard format for stating a model's training data, intended use and limits. |
| [Model Cards for Model Reporting (arXiv 1810.03993)](<https://arxiv.org/abs/1810.03993>) | Margaret Mitchell and co-authors | paper | intermediate | free | opened | Template for stating intended use and measured performance by group. |
| [Common Voice privacy notice and terms](<https://commonvoice.mozilla.org/en/privacy>) | Mozilla | documentation | intermediate | free | search (site) | Worked example of contributor consent and a CC0 release for voice data. |
| [Biometric data guidance: Biometric recognition](<https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/biometric-data-guidance-biometric-recognition/>) | UK Information Commissioner's Office | documentation | intermediate | free | search (site) | Regulator guidance on when voice recordings count as biometric data, and on consent. |
| [ACM Code of Ethics and Professional Conduct](<https://www.acm.org/about-acm/code-of-ethics>) | Association for Computing Machinery | documentation | intermediate | free | search | General professional baseline, including data minimisation and consent. |
| [General Data Protection Regulation (Regulation (EU) 2016/679)](<https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng>) | European Union (EUR-Lex) | documentation | advanced | free | search | Legal text on consent and on special categories of personal data. |
| [Anonymizing Speech: Evaluating and Designing Speaker Anonymization Techniques (arXiv 2308.04455)](<https://arxiv.org/abs/2308.04455>) | Pierre Champion | paper | advanced | free | opened | What a voice recording reveals about a speaker and how far anonymisation can go. |

## 18. Communities, lecture courses, newsletters and research venues

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Kaggle](<https://www.kaggle.com/>) | Kaggle | community | beginner | free | opened | Free GPU notebooks, public datasets and worked solutions. |
| [fast.ai course forums](<https://forums.fast.ai/>) | fast.ai | community | beginner | free | search (site) | Friendly community for practical deep learning questions. |
| [The Batch](<https://www.deeplearning.ai/the-batch>) | DeepLearning.AI | community | beginner | free | search | Weekly newsletter summarising ML news. |
| [MIT OpenCourseWare](<https://ocw.mit.edu/>) | Massachusetts Institute of Technology | course | intermediate | free | search (site) | Free lecture courses in maths, signals and machine learning. |
| [Statistics 110 lecture videos](<https://stat110.hsites.harvard.edu/youtube>) | Joe Blitzstein, Harvard University | video series | intermediate | free | search | Full probability lecture recordings. |
| [Deep Learning Specialization](<https://www.coursera.org/specializations/deep-learning>) | Andrew Ng, DeepLearning.AI | course | intermediate | paid subscription (audit options vary) | search | Structured five-course sequence with graded exercises. |
| [Ahead of AI](<https://magazine.sebastianraschka.com/>) | Sebastian Raschka | community | intermediate | free (paid tier optional) | search (site) | Newsletter with careful explanations of recent research. |
| [Import AI](<https://importai.substack.com/>) | Jack Clark | community | intermediate | free | search | Long-running weekly research newsletter. |
| [Open ASR Leaderboard](<https://huggingface.co/spaces/hf-audio/open_asr_leaderboard>) | Hugging Face audio team | community | intermediate | free | opened | Tracks accuracy and speed of open speech models. |
| [Hugging Face Daily Papers](<https://huggingface.co/papers>) | Hugging Face | community | intermediate | free | opened | Daily trending papers with linked code and models. |
| [ISCA Archive](<https://www.isca-archive.org/>) | International Speech Communication Association | community | advanced | free | search (site) | Open archive of all Interspeech and ISCA workshop papers, including the SLaTE workshops on speech technology in education. |
| [IEEE Signal Processing Society: Speech and Language Processing Technical Committee newsletter](<https://signalprocessingsociety.org/community-involvement/speech-and-language-processing/newsletter/spring-2024-edition-speech-and>) | IEEE Signal Processing Society | community | advanced | free to browse | search | News from the community behind ICASSP, ASRU and SLT. |
| [arXiv eess.AS (Audio and Speech Processing) recent listing](<https://arxiv.org/list/eess.AS/recent>) | arXiv | community | advanced | free | opened | Daily new speech papers. |
| [arXiv cs.CL (Computation and Language) recent listing](<https://arxiv.org/list/cs.CL/recent>) | arXiv | community | advanced | free | opened | Daily language-technology papers, including Arabic NLP. |
| [arXiv cs.SD (Sound) recent listing](<https://arxiv.org/list/cs.SD/recent>) | arXiv | community | advanced | free | opened | Daily audio papers. |
| [ACL Anthology: Special Interest Group on Arabic Natural Language Processing](<https://aclanthology.org/sigs/sigarab/>) | Association for Computational Linguistics | community | advanced | free | search | All ArabicNLP and WANLP proceedings, including shared tasks on speech. |
| [SIGARAB: Arabic NLP archives](<https://www.sigarab.org/arabic-nlp-archives>) | SIGARAB | community | advanced | free | search | The group that organises the ArabicNLP conference and its shared tasks. |

## 19. Foundational and directly relevant research papers

| Resource | Author or organisation | Type | Level | Cost | Verified | Why it matters here |
|---|---|---|---|---|---|---|
| [Adam: A Method for Stochastic Optimization (arXiv 1412.6980)](<https://arxiv.org/abs/1412.6980>) | Diederik P. Kingma, Jimmy Ba | paper | intermediate | free | opened | The optimiser family behind nearly all training runs. |
| [Decoupled Weight Decay Regularization (arXiv 1711.05101)](<https://arxiv.org/abs/1711.05101>) | Ilya Loshchilov, Frank Hutter | paper | intermediate | free | opened | AdamW, the default optimiser in NeMo fine-tuning configs. |
| [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arXiv 1502.03167)](<https://arxiv.org/abs/1502.03167>) | Sergey Ioffe, Christian Szegedy | paper | intermediate | free | opened | Normalisation used in the Conformer convolution module; matters when freezing or quantising. |
| [Layer Normalization (arXiv 1607.06450)](<https://arxiv.org/abs/1607.06450>) | Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton | paper | intermediate | free | opened | Normalisation used throughout transformer blocks. |
| [Deep Residual Learning for Image Recognition (arXiv 1512.03385)](<https://arxiv.org/abs/1512.03385>) | Kaiming He and co-authors | paper | intermediate | free | opened | Residual connections, which make deep encoders trainable. |
| [Attention Is All You Need (arXiv 1706.03762)](<https://arxiv.org/abs/1706.03762>) | Ashish Vaswani and co-authors | paper | intermediate | free | opened | The transformer architecture. |
| [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin (arXiv 1512.02595)](<https://arxiv.org/abs/1512.02595>) | Dario Amodei and co-authors | paper | intermediate | free | opened | Showed large-scale CTC training working end to end. |
| [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition (arXiv 1904.08779)](<https://arxiv.org/abs/1904.08779>) | Daniel S. Park and co-authors | paper | intermediate | free | opened | The standard augmentation switched on in FastConformer configs. |
| [Jasper: An End-to-End Convolutional Neural Acoustic Model (arXiv 1904.03288)](<https://arxiv.org/abs/1904.03288>) | Jason Li and co-authors | paper | intermediate | free | opened | First of the NVIDIA convolutional CTC model line. |
| [QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions (arXiv 1910.10261)](<https://arxiv.org/abs/1910.10261>) | Samuel Kriman and co-authors | paper | intermediate | free | opened | Small CTC model; a candidate shape for a distilled student. |
| [NeMo: a toolkit for building AI applications using Neural Modules (arXiv 1909.09577)](<https://arxiv.org/abs/1909.09577>) | Oleksii Kuchaiev and co-authors | paper | intermediate | free | opened | The design of the toolkit being used. |
| [Robust Speech Recognition via Large-Scale Weak Supervision (arXiv 2212.04356)](<https://arxiv.org/abs/2212.04356>) | Alec Radford and co-authors | paper | intermediate | free | opened | Whisper; also a lesson in filtering weakly labelled data. |
| [SpeechBrain: A General-Purpose Speech Toolkit (arXiv 2106.04624)](<https://arxiv.org/abs/2106.04624>) | Mirco Ravanelli and co-authors | paper | intermediate | free | opened | Overview of a readable PyTorch speech toolkit. |
| [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates (arXiv 1708.07120)](<https://arxiv.org/abs/1708.07120>) | Leslie N. Smith, Nicholay Topin | paper | intermediate | free | opened | One-cycle learning rate schedules for short GPU sessions. |
| [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (arXiv 1901.02860)](<https://arxiv.org/abs/1901.02860>) | Zihang Dai and co-authors | paper | advanced | free | opened | Source of the relative positional encoding used in Conformer attention. |
| [Sequence Transduction with Recurrent Neural Networks (arXiv 1211.3711)](<https://arxiv.org/abs/1211.3711>) | Alex Graves | paper | advanced | free | opened | The RNN-Transducer, the main alternative to CTC. |
| [Listen, Attend and Spell (arXiv 1508.01211)](<https://arxiv.org/abs/1508.01211>) | William Chan and co-authors | paper | advanced | free | opened | Attention-based encoder-decoder speech recognition. |
| [Citrinet: Closing the Gap between Non-Autoregressive and Autoregressive End-to-End Models for Automatic Speech Recognition (arXiv 2104.01721)](<https://arxiv.org/abs/2104.01721>) | Somshubra Majumdar and co-authors | paper | advanced | free | opened | CTC with sub-word tokens and squeeze-excitation. |
| [Conformer: Convolution-augmented Transformer for Speech Recognition (arXiv 2005.08100)](<https://arxiv.org/abs/2005.08100>) | Anmol Gulati and co-authors | paper | advanced | free | opened | The encoder architecture behind FastConformer. |
| [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition (arXiv 2305.05084)](<https://arxiv.org/abs/2305.05084>) | Dima Rekesh and co-authors | paper | advanced | free | opened | The exact encoder being fine-tuned: 8x subsampling and its trade-offs. |
| [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations (arXiv 2304.06795)](<https://arxiv.org/abs/2304.06795>) | Hainan Xu and co-authors | paper | advanced | free | opened | Token-and-duration transducer used in recent NVIDIA models. |
| [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition (arXiv 2206.00888)](<https://arxiv.org/abs/2206.00888>) | Sehoon Kim and co-authors | paper | advanced | free | opened | Efficiency-focused redesign of Conformer for CTC. |
| [Zipformer: A faster and better encoder for automatic speech recognition (arXiv 2310.11230)](<https://arxiv.org/abs/2310.11230>) | Zengwei Yao and co-authors | paper | advanced | free | opened | Efficient encoder widely deployed through sherpa-onnx. |
| [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (arXiv 2006.11477)](<https://arxiv.org/abs/2006.11477>) | Alexei Baevski and co-authors | paper | advanced | free | opened | Self-supervised pre-training then CTC fine-tuning with little labelled data. |
| [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units (arXiv 2106.07447)](<https://arxiv.org/abs/2106.07447>) | Wei-Ning Hsu and co-authors | paper | advanced | free | opened | The other main self-supervised speech encoder. |
| [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale (arXiv 2111.09296)](<https://arxiv.org/abs/2111.09296>) | Arun Babu and co-authors | paper | advanced | free | opened | Multilingual self-supervised models covering Arabic. |
| [Scaling Speech Technology to 1,000+ Languages (arXiv 2305.13516)](<https://arxiv.org/abs/2305.13516>) | Vineel Pratap and co-authors | paper | advanced | free | opened | MMS, built largely from read religious texts, with a long-audio forced alignment method. |
| [WhisperX: Time-Accurate Speech Transcription of Long-Form Audio (arXiv 2303.00747)](<https://arxiv.org/abs/2303.00747>) | Max Bain and co-authors | paper | advanced | free | opened | VAD segmentation plus phoneme-model forced alignment for word timestamps. |
| [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (arXiv 2109.11680)](<https://arxiv.org/abs/2109.11680>) | Qiantong Xu, Alexei Baevski, Michael Auli | paper | advanced | free | opened | CTC phoneme recognition with articulatory mapping across languages. |
| [Why does CTC result in peaky behavior? (arXiv 2105.14849)](<https://arxiv.org/abs/2105.14849>) | Albert Zeyer, Ralf Schlüter, Hermann Ney | paper | advanced | free | opened | Explains spiky CTC posteriors, which affect alignment boundaries and GOP scores. |
| [End-to-End Speech Recognition: A Survey (arXiv 2303.03329)](<https://arxiv.org/abs/2303.03329>) | Rohit Prabhavalkar and co-authors | paper | advanced | free | opened | Up-to-date map of CTC, transducer and attention models. |
| [WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit (arXiv 2102.01547)](<https://arxiv.org/abs/2102.01547>) | Zhuoyuan Yao and co-authors | paper | advanced | free | opened | Design choices for deployable streaming models. |
| [ESPnet: End-to-End Speech Processing Toolkit (arXiv 1804.00015)](<https://arxiv.org/abs/1804.00015>) | Shinji Watanabe and co-authors | paper | advanced | free | opened | Toolkit paper explaining hybrid CTC/attention training. |
| [Lhotse: a speech data representation library for the modern deep learning ecosystem (arXiv 2110.12561)](<https://arxiv.org/abs/2110.12561>) | Piotr Żelasko and co-authors | paper | advanced | free | opened | Data manifests and dynamic batching for speech. |
| [QASR: QCRI Aljazeera Speech Resource -- A Large Scale Annotated Arabic Speech Corpus (arXiv 2106.13000)](<https://arxiv.org/abs/2106.13000>) | Hamdy Mubarak and co-authors | paper | advanced | free | opened | Large Arabic speech corpus and its lightly supervised alignment. |
