# ISLP study notes: Chapters 1-3 (for Qaf-ft)

Source: `refs/islp_text/islp.txt`, lines 416-8734 (pdftotext -layout extraction of
"An Introduction to Statistical Learning with Applications in Python").
Coverage: Ch. 1 Introduction (pp. 1-13), Ch. 2 Statistical Learning (pp. 15-67),
Ch. 3 Linear Regression (pp. 69-134), including labs; exercises skimmed.

Conventions used here

- "p. N" is the book page from the running header. "§" is the book section.
- Everything under a heading "Application to Qaf-ft" is this note-taker's
  application of the book's principles. The book says nothing about speech,
  Arabic or recitation. Points that go beyond what the book states are marked
  **(inference)**. Tajweed facts are domain assumptions to be checked against the
  project's own tajweed specification, not book content.
- People are named by role only (the user, the reciter, the listener, the authors).
- Chapters 1-3 contain no numbered "Algorithm" boxes. Procedures (KNN classifier,
  KNN regression, forward/backward/mixed selection) are given in prose.

Extraction caveats (see also the last section)

- All figure graphics are garbled; only captions, axis labels and prose survive.
  No curve values are read off figures in these notes.
- The error term epsilon is printed as `"` and "not equal" as `%=` in the text file.
- Summation signs and fraction bars are broken across lines. Equations below were
  reassembled from the fragments; all are standard forms and the fragments agree.

---

## Chapter 1: Introduction (pp. 1-13)

Chapter 1 has unnumbered headings only.

### An Overview of Statistical Learning (p. 1)

- Statistical learning: "a vast set of tools for understanding data", classed as
  supervised or unsupervised.
- Supervised: build a statistical model for predicting or estimating an output
  from one or more inputs.
- Unsupervised: inputs but no supervising output; one can still learn
  relationships and structure.

### Wage data (pp. 1-2; Figure 1.1, p. 2)

- Men from the Atlantic region of the United States; association of age,
  education and calendar year with wage.
- Wage rises with age, then falls after approximately age 60. Average wage rises
  by approximately $10,000, roughly linearly, between 2003 and 2009, "though this
  rise is very slight relative to the variability in the data". Education levels
  1 (no high school diploma) to 5 (advanced graduate degree); wage rises with level.
- Warning: there is "a significant amount of variability associated with this
  average value, and so age alone is unlikely to provide an accurate prediction
  of a particular man's wage" (p. 1). The most accurate prediction combines age,
  education and year, and should account for the non-linear wage-age relation.
- Predicting a continuous (quantitative) output is a regression problem (p. 2).

### Stock Market data (pp. 2-4; Figures 1.2 p. 3, 1.3 p. 4)

- Smarket: daily movements of the S&P 500 over a 5-year period, 2001 to 2005.
  Goal: predict Up or Down from the past 5 days' percentage changes. A
  categorical (qualitative) output makes this a classification problem.
- Figure 1.2: boxplots of the previous day's percentage change for the 648 days
  the market rose and the 602 days it fell "look almost identical"; same for 2
  and 3 days previous. No simple strategy exists.
- Figure 1.3: a quadratic discriminant analysis model fitted on 2001-2004 and
  used to predict the probability of a decrease on the 2005 data. Predicted
  probabilities lie roughly between 0.46 and 0.52 (axis labels); on average the
  predicted probability of decrease is higher on days the market does decrease.
  Direction is predicted correctly 60% of the time ("approximately 60%" on p. 3).
- Design point: the model is fitted on one period and judged on a later period
  it never saw.

### Gene Expression data (pp. 3-5; Figure 1.4 p. 5)

- Only inputs, no output: a clustering problem.
- NCI60: 6,830 gene expression measurements for each of 64 cancer cell lines.
- Each cell line is summarised by two numbers, Z1 and Z2, the first two
  principal components. "It is likely that this dimension reduction has resulted
  in some loss of information."
- "Deciding on the number of clusters is often a difficult problem." The plot
  suggests at least four groups.
- The cell lines correspond to 14 cancer types (not used to make the left
  panel). Lines of the same type sit near each other, which "provides some
  independent verification of the accuracy of our clustering analysis" (p. 4).

### A Brief History of Statistical Learning (p. 5)

- Least squares: beginning of the nineteenth century, first applied in astronomy.
- Linear discriminant analysis: 1936. Logistic regression: 1940s.
- Generalized linear model: early 1970s (linear and logistic regression are
  special cases).
- By the end of the 1970s methods were "almost exclusively linear" because
  non-linear fitting was computationally hard.
- Classification and regression trees: mid 1980s; generalized additive models
  shortly after. Neural networks gained popularity in the 1980s; support vector
  machines arose in the 1990s.

### This Book (pp. 6-8)

- The companion reference text (The Elements of Statistical Learning) was first
  published in 2001. This book first appeared in 2013, second edition 2021, with
  R labs; this version has Python labs.
- Four premises (pp. 7-8):
  1. Many statistical learning methods are relevant well beyond the statistical
     sciences.
  2. "Statistical learning should not be viewed as a series of black boxes. No
     single approach will perform well in all possible applications."
  3. One needs to know what each cog does, but not how to build the machine.
  4. The reader wants to apply methods to real problems; hence a lab per chapter.

### Who Should Read This Book? (p. 8)

- Expects one elementary statistics course; linear regression background useful.

### Notation and Simple Matrix Algebra (pp. 8-10)

- n = number of distinct data points (observations); p = number of variables
  available for prediction. Wage data: n = 3,000, p = 11.
- p can be in the thousands or millions (modern biological or web advertising data).
- x_ij = value of variable j for observation i. X is the n x p matrix.
- Equation (1.1), p. 9: x_i = (x_i1, x_i2, ..., x_ip)^T, the row of X as a
  column vector. Columns are written x_j (length n).
- y_i = i-th observation of the response; y = (y_1, ..., y_n)^T.
- Typeface rules: length-n vectors bold lower case; other vectors and scalars
  normal lower case; matrices bold capitals; random variables normal capitals.
- Matrix product: (AB)_ij = sum over k of a_ik b_kj; needs columns of A = rows of B.
  Worked example (p. 10): A = [[1,2],[3,4]], B = [[5,6],[7,8]], AB = [[19,22],[43,50]].

### Organization, data sets, website (pp. 11-13; Table 1.1 p. 12)

- Ch. 2 terminology and KNN; Ch. 3-4 linear regression and classification; Ch. 5
  cross-validation and bootstrap; Ch. 6 stepwise, ridge, PCR, lasso; Ch. 7
  non-linear single-input methods and additive models; Ch. 8 trees, bagging,
  boosting, random forests; Ch. 9 SVMs; Ch. 10 deep learning; Ch. 11 survival
  analysis; Ch. 12 unsupervised (PCA, K-means, hierarchical); Ch. 13 multiple testing.
- "A central problem in all statistical learning situations involves choosing
  the best method for a given application" (p. 11).
- "Linear methods often have advantages over their non-linear competitors in
  terms of interpretability and sometimes also accuracy" (p. 11).
- Warning (pp. 11-12): package versions change, so lab results "may no longer
  correspond precisely" to what a later reader obtains.
- Table 1.1 lists 21 data sets (Auto, Bikeshare, Boston, BrainCancer, Caravan,
  Carseats, College, Credit, Default, Fund, Hitters, Khan, NCI60, NYSE, OJ,
  Portfolio, Publication, Smarket, USArrests, Wage, Weekly).
- The symbol that marks harder sections is missing from the extraction (p. 12).

### Application to Qaf-ft (Chapter 1)

1. **Name each sub-task by its statistical type** (pp. 1-4). Transcription at
   diacritic/phoneme level is supervised classification. The per-word mistake
   decision is supervised binary classification with a probability output. Madd
   (vowel-lengthening) duration scoring is supervised regression. Exploring
   reciter styles or recording conditions without labels is unsupervised.
2. **Look at class-conditional feature distributions before modelling**
   (Figure 1.2). For each verifier feature (alignment score, posterior margin,
   duration, entropy) draw boxplots for correct words and for mistaken words on
   held-out data. A feature whose two boxplots "look almost identical" has little
   stand-alone value. **(inference)**
3. **Adopt Figure 1.3 as a standard assessment plot.** Boxplots of the verifier's
   predicted mistake probability, by true class, on data from reciters it never
   saw. Note how weak the separation in the book's example is at 60% accuracy;
   a plot shows what a single accuracy number hides.
4. **Judge on data from a different slice than training** (Figure 1.3 design:
   fit on 2001-2004, test on 2005). For Qaf-ft the analogous slice is reciters
   and recording sources, not random frames or random verses. **(inference)**
5. **An average is not an individual prediction** (p. 1). A reciter's average
   madd length can be well estimated while a single madd instance still varies a
   lot. This anticipates confidence versus prediction intervals in §3.2.2.
6. **Unsupervised checks on data coverage** (pp. 3-5). Project per-reciter
   acoustic or tempo summaries to two principal components and look for groups;
   validate any clusters against known labels (reciter identity, recitation
   style, microphone) as the authors did with cancer types. Expect the number of
   clusters to be hard to settle. Use the result to stratify test sets. **(inference)**
7. **Decide what n is.** The book's n counts "distinct data points". Frames and
   words from one reciter are not distinct in the sense that matters for
   generalising to new reciters; see §3.3.3 on correlated errors. Record both
   word counts and reciter counts in every results table. **(inference)**
8. **Not a black box; no universal winner** (premise 2, p. 7). Compare several
   verifier families on the same held-out data instead of assuming one.
9. **Pin versions and seeds** for all evaluation code (pp. 11-12; lab §2.3.3).

---

## Chapter 2: Statistical Learning (pp. 15-67)

### 2.1 What Is Statistical Learning? (pp. 15-17)

- Advertising data: sales of a product in 200 markets with TV, radio and
  newspaper budgets. Figure 2.1 (p. 16): sales in thousands of units against
  each budget in thousands of dollars, each with its simple least squares line.
- Inputs X (predictors, independent variables, features, variables); output Y
  (response, dependent variable).
- Equation (2.1), p. 15: `Y = f(X) + e`. f is "some fixed but unknown function";
  e is a random error term, "independent of X" with "mean zero". f is the
  systematic information X gives about Y.
- Income data (simulated, 30 individuals): Figure 2.2 (p. 16) income against
  years of education with the true f; Figure 2.3 (p. 18) income against
  education and seniority with the true surface.
- "Statistical learning refers to a set of approaches for estimating f" (p. 17).

#### 2.1.1 Why Estimate f? (pp. 17-20)

Prediction

- Equation (2.2), p. 17: `Y_hat = f_hat(X)`. f_hat is "often treated as a black
  box" if it predicts accurately.
- Reducible error: f_hat is imperfect; can be improved with a better method.
  Irreducible error: Y also depends on e, which cannot be predicted from X.
- Why irreducible error exceeds zero (pp. 17-18): e may contain unmeasured
  variables useful for predicting Y, and unmeasurable variation.
- Equation (2.3), p. 18, with f_hat and X fixed:
  `E(Y - Y_hat)^2 = [f(X) - f_hat(X)]^2 + Var(e)` (reducible + irreducible).
- "The irreducible error will always provide an upper bound on the accuracy of
  our prediction for Y. This bound is almost always unknown in practice" (p. 18).

Inference (pp. 18-20)

- Goal is to understand the association between Y and X1..Xp; f_hat "cannot be
  treated as a black box".
- Three questions: which predictors are associated with the response; what is
  the relationship with each; is a linear equation adequate.
- Examples: direct-marketing response (prediction); Advertising questions and
  brand choice against price (inference); house values (both).
- Trade-off stated on p. 20: linear models give simple, interpretable inference
  but may predict less accurately; highly non-linear approaches may predict well
  but are harder to interpret.

#### 2.1.2 How Do We Estimate f? (pp. 20-23)

- Training data: the n observed points {(x1,y1),...,(xn,yn)} used to fit.
- Parametric methods, two steps: (1) assume a form for f, for example
  equation (2.4), p. 20: `f(X) = b0 + b1 X1 + ... + bp Xp`, which leaves p + 1
  coefficients to estimate; (2) fit, most commonly by (ordinary) least squares.
- Disadvantage: the chosen model "will usually not match the true unknown form
  of f". More flexible models need more parameters and can overfit, which
  "essentially means they follow the errors, or noise, too closely" (p. 21).
- Figure 2.4 (p. 21): linear plane fitted to Income; "not quite right" but
  captures the positive relationships.
- Non-parametric methods (p. 22): no explicit form for f; seek a fit close to the
  data "without being too rough or wiggly". Advantage: can fit many shapes.
  "Major disadvantage": "a very large number of observations (far more than is
  typically needed for a parametric approach) is required".
- Figure 2.5 (p. 22): smooth thin-plate spline, close to the true f.
  Figure 2.6 (p. 23): rough thin-plate spline that "makes zero errors on the
  training data", an example of overfitting; such a fit "will not yield accurate
  estimates of the response on new observations".

#### 2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability (pp. 23-25)

- Flexible methods produce a wide range of shapes; restrictive ones a narrow range.
- Figure 2.7 (p. 24), from high interpretability and low flexibility to the
  reverse: subset selection, lasso; least squares; generalized additive models,
  trees; bagging, boosting; support vector machines; deep learning.
- For inference, restrictive models are preferred.
- Key warning (p. 25): even when only prediction matters, the most flexible
  model is not always best. "We will often obtain more accurate predictions
  using a less flexible method", because of overfitting.

#### 2.1.4 Supervised Versus Unsupervised Learning (pp. 25-27)

- Supervised: each x_i has a response y_i. Unsupervised: x_i only; "in some
  sense working blind".
- Cluster analysis asks whether observations fall into relatively distinct
  groups. Figure 2.8 (p. 26): 150 observations, three groups; easy when well
  separated, and when groups overlap a clustering method "could not be expected
  to assign all of the overlapping points to their correct group".
- With p variables there are p(p - 1)/2 scatterplots, so visual inspection fails.
- Semi-supervised learning (p. 27): responses for only m < n observations,
  arising when predictors are cheap and responses expensive. Out of the book's scope.

#### 2.1.5 Regression Versus Classification Problems (p. 27)

- Quantitative response: regression. Qualitative (categorical) response with K
  classes: classification. The line is not crisp: logistic regression is a
  classification method that estimates class probabilities, so it can be seen as
  regression too. KNN and boosting handle both.
- Method choice depends on the response type; predictor type matters less,
  provided qualitative predictors are properly coded.

### 2.2 Assessing Model Accuracy (pp. 27-39)

- "There is no free lunch in statistics: no one method dominates all others over
  all possible data sets" (p. 27).

#### 2.2.1 Measuring the Quality of Fit (pp. 28-31)

- Equation (2.5), p. 28: `MSE = (1/n) * sum_i (y_i - f_hat(x_i))^2`. Computed on
  training data this is the training MSE.
- "In general, we do not really care how well the method works on the training
  data" (p. 28). Examples: a stock model should predict tomorrow's price, not
  last week's; a diabetes model should work for future patients.
- Equation (2.6), p. 29: test MSE `Ave(y0 - f_hat(x0))^2` over previously unseen
  test observations (x0, y0). Choose the method with the lowest test MSE.
- "There is no guarantee that the method with the lowest training MSE will also
  have the lowest test MSE" (p. 29).
- Figure 2.9 (p. 29): simulated data with a non-linear f; a linear fit and two
  smoothing splines; training MSE (grey), test MSE (red), and Var(e) (dashed) as
  functions of flexibility (degrees of freedom; linear regression has two).
  Training MSE "declines monotonically"; test MSE is U-shaped. The mid-flexible
  fit is near optimal.
- "This is a fundamental property of statistical learning that holds regardless
  of the particular data set at hand and regardless of the statistical method
  being used. As model flexibility increases, the training MSE will decrease,
  but the test MSE may not" (p. 30).
- Overfitting: small training MSE with large test MSE; the procedure "is working
  too hard to find patterns in the training data".
- Precise definition (p. 31): training MSE is almost always below test MSE in
  any case; "Overfitting refers specifically to the case in which a less
  flexible model would have yielded a smaller test MSE."
- Figure 2.10 (p. 31): f close to linear; test MSE falls only slightly, linear
  beats the highly flexible fit. Figure 2.11 (p. 32): f far from linear; both
  curves drop fast before test MSE slowly rises.
- The flexibility at minimum test MSE "can vary considerably among data sets".
  Cross-validation (Ch. 5) estimates test MSE from training data.

#### 2.2.2 The Bias-Variance Trade-Off (pp. 31-34)

- Equation (2.7), p. 32:
  `E(y0 - f_hat(x0))^2 = Var(f_hat(x0)) + [Bias(f_hat(x0))]^2 + Var(e)`.
  Expected test MSE at x0 is the average over many training sets.
- Expected test MSE can never be below Var(e).
- Variance: "the amount by which f_hat would change if we estimated it using a
  different training data set". More flexible methods generally have higher variance.
- Bias: error from approximating a complicated real problem with a simpler
  model. More flexible methods generally have less bias.
- As flexibility rises, bias at first falls faster than variance rises, so test
  MSE falls; later variance dominates and test MSE rises.
- Figure 2.12 (p. 33): squared bias, variance, Var(e) and test MSE for the three
  data sets of Figures 2.9-2.11; the optimal flexibility differs in each.
- Extremes (p. 34): a curve through every training point has very low bias and
  high variance; a horizontal line has very low variance and high bias.
- In real problems test MSE, bias and variance cannot be computed explicitly,
  but "one should always keep the bias-variance trade-off in mind". If the true
  f is linear, linear regression has no bias and is hard to beat.

#### 2.2.3 The Classification Setting (pp. 34-39)

- Equation (2.8), p. 35: training error rate `(1/n) * sum_i I(y_i != y_hat_i)`.
- Equation (2.9), p. 35: test error rate `Ave(I(y0 != y_hat_0))`. A good
  classifier has the smallest test error.
- Bayes classifier: assign x0 to the class j maximising equation (2.10),
  `Pr(Y = j | X = x0)`. It minimises the test error rate on average. With two
  classes: predict class one if Pr(Y = 1 | X = x0) > 0.5.
- Bayes decision boundary: points where the probability is exactly 50%.
- Equation (2.11), p. 36: Bayes error rate `1 - E(max_j Pr(Y = j | X))`, the
  lowest possible test error rate, "analogous to the irreducible error". It is
  above zero when classes overlap.
- Figure 2.13 (p. 36): simulated data, 100 observations in each of two groups.
  The text gives the Bayes error rate as 0.133 on p. 36 and as 0.1304 on p. 37
  (both as printed).
- The Bayes classifier is "an unattainable gold standard" because the
  conditional distribution is unknown for real data.
- K-nearest neighbours (KNN) classifier (pp. 36-37): find the K training points
  nearest x0 (set N0) and estimate, equation (2.12), p. 37:
  `Pr(Y = j | X = x0) = (1/K) * sum over i in N0 of I(y_i = j)`; classify to the
  largest.
- Figure 2.14 (p. 38): K = 3 with six blue and six orange points; neighbours are
  two blue and one orange, giving 2/3 and 1/3; prediction blue.
- Figure 2.15 (p. 38): K = 10 boundary is very close to the Bayes boundary. Test
  error 0.1363 against a Bayes error rate of 0.1304 (p. 37).
- Figure 2.16 (p. 39): K = 1 is overly flexible (low bias, very high variance);
  K = 100 is close to linear (low variance, high bias). Test error rates 0.1695
  and 0.1925 respectively.
- With K = 1 the KNN training error rate is 0 (p. 37).
- Figure 2.17 (p. 39): training error (200 observations) and test error (5,000
  observations) against 1/K. Training error falls steadily as flexibility rises;
  test error is U-shaped with a minimum at approximately K = 10. "The jumpiness
  of the curves is due to the small size of the training data set."
- "Choosing the correct level of flexibility is critical to the success of any
  statistical learning method" (p. 38).

### 2.3 Lab: Introduction to Python (pp. 40-62)

Mostly Python mechanics. Points with statistical or practical weight:

- Lists concatenate under `+`; numpy arrays add element-wise (pp. 41-42).
- `reshape` returns an array sharing memory with the original; modifying one
  modifies the other (pp. 44-45).
- `np.random.normal` gives different results on every call; use
  `np.random.default_rng(seed)` for reproducibility. Example: seed 1303 gives
  [4.09, -1.07] twice (p. 47). Small discrepancies can still appear across
  numpy versions.
- `np.corrcoef` example value 0.69 (p. 46).
- "By default np.var() divides by the sample size n rather than n - 1; see the
  ddof argument" (p. 47). Example values: mean -0.11, variance 2.72, sd 1.65.
- Indexing with two lists picks paired elements, not a submatrix; use
  `np.ix_` or slices. Boolean and integer arrays index differently even when
  equal under `==` (pp. 52-54).
- Loading data (pp. 55-56): view a file before loading. In `Auto`, missing
  values coded `?` made the whole horsepower column load as strings (dtype
  object). Fix with `na_values=['?']`. Shape (397, 9); five rows have missing
  values; after `dropna()` shape is (392, 9). Horsepower sum 40952.0.
- `loc[]` for labels and Booleans, `iloc[]` for integer positions (pp. 57-59).
- Loops and formatting (pp. 59-61): totals 24 and 144; weighted average 10.8.
  A 127 x 5 frame with entries set missing with probability 0.2 shows realised
  missing shares of 16.54%, 25.98%, 29.13%, 21.26% and 22.83%.
- Plots: `subplots`, scatter, contour, heatmap, boxplot, histogram,
  `pd.plotting.scatter_matrix`, `describe()` (pp. 48-50, 61-62). A numeric
  variable with few values (cylinders) may be better treated as qualitative.

### 2.4 Exercises (pp. 63-67), skimmed

- Conceptual 1: when a flexible method should do better or worse (large n small
  p; large p small n; highly non-linear relationship; very high Var(e)).
- 2: classify scenarios as regression or classification, inference or prediction.
- 3: sketch squared bias, variance, training error, test error and Bayes error
  against flexibility. 5-6: pros and cons of flexible and of parametric approaches.
- 7: KNN by hand on six observations with three predictors.
- Applied 8-10: College, Auto and Boston data exploration.
- The book poses these without printed answers.

### Application to Qaf-ft (Chapter 2)

Framing

1. **Prediction and inference are both needed, in different stages** (§2.1.1).
   The CTC acoustic model is a prediction device and may be a black box. The
   verifier and the tajweed scorer must also support inference: which features
   signal a mistake, how madd duration depends on tempo, and what to tell the
   reciter. That argues for restrictive, interpretable models in those stages
   (Figure 2.7, p. 24), and p. 25 says such models often predict better too.
2. **There is an irreducible floor** (eq. 2.3, eq. 2.11). For mistake detection
   the floor comes from disagreement among expert listeners, acceptable variant
   readings, and things the features do not measure. The book says the bound is
   "almost always unknown". Estimate a practical proxy by having two or more
   expert listeners label the same test items and measuring their disagreement;
   do not expect any model, Qaf-ft or a competitor, to beat it. **(inference)**
3. **Labelled mistakes are expensive, audio is cheap** (§2.1.4, p. 27). This is
   the semi-supervised setting the book names and leaves out of scope. One
   practical reading: fit a reference model on verified-correct recitation only
   and treat departures as candidates (see Ch. 3 application). **(inference)**

Training the acoustic model

4. **Never select on training error** (§2.2.1, pp. 28-30). Training loss on the
   training reciters will fall monotonically with capacity, unfrozen layers and
   training steps. Checkpoint, model size and regularisation must be chosen on
   error measured on reciters absent from training. Treating fine-tuning steps
   and adapter size as "flexibility" is an analogy to the book's degrees of
   freedom. **(inference)**
5. **Over-fitting to a few famous reciters.** The book's test observation is
   "previously unseen ... not used to train". A different chapter recited by a
   training reciter shares voice, microphone, room and style with training, so
   error on it behaves like training error. Split by reciter and by recording
   source. Report three numbers side by side: training reciters, held-out
   professional reciters, held-out ordinary reciters (the target population).
   The book's stock example applies directly: we "don't really care" how the
   model does on last week's data. **(inference)**
6. **The text is a closed corpus.** Every verse in any test set has been seen as
   text during training. A flexible model can learn to output the expected verse
   instead of what was said, which is the worst failure for mistake detection.
   This is "working too hard to find patterns in the training data" (p. 30).
   Test sets therefore need real or carefully constructed deviations from the
   canonical text, spoken by unseen reciters. **(inference)**
7. **Diagnose with the book's definition of overfitting** (p. 31). A gap between
   training and held-out error is expected and is not proof. Overfitting means a
   less flexible configuration would give lower held-out error. So run at least
   three flexibility levels and compare held-out-reciter error across them.
8. **Model size through bias and variance** (eq. 2.7). Variance is how much the
   fitted model changes with a different training set. Measure it: fine-tune on
   different subsets of reciters (and seeds) and look at the spread of
   held-out error. Large spread means high variance: reduce flexibility or add
   reciters. Training and held-out error both high and close together suggests
   bias: add flexibility. The optimum "can vary considerably among data sets"
   (p. 31), so do not import a size from someone else's benchmark. **(inference)**

The verifier

9. **Start parametric** (§2.1.2, p. 22). Non-parametric methods need "a very
   large number of observations". True mistakes are rare, so the effective n for
   the mistake class is small. Four features and few positives favour a
   parametric model. Move to something flexible only if held-out error improves
   (Figures 2.9-2.11 show both outcomes are possible).
10. **The 0.5 rule minimises overall error, not false flags** (§2.2.3, p. 35).
    The Bayes classifier with a 0.5 threshold minimises the test error rate
    (2.9). Qaf-ft's requirement is a very low false-flag rate, a different
    criterion. Chapters 1-3 define only the overall error rate; class-specific
    error rates and other thresholds are outside these chapters. Report
    false-flag rate and miss rate separately from the start. **(inference)**
11. **Calibration is what the Bayes rule needs.** The rule depends on the true
    Pr(Y = j | X = x0). A KNN estimate (2.12) can only take values that are
    multiples of 1/K (for K = 3: 0, 1/3, 2/3, 1), so small K cannot express the
    fine probabilities near 0 or 1 that a strict false-flag budget needs.
    **(inference from eq. 2.12)**
12. **Tune every hyperparameter on held-out data** (pp. 37-39). In the book's
    example K = 1 has training error 0 and test error 0.1695; K = 100 gives
    0.1925; K = 10 gives 0.1363 against a floor of 0.1304.

Assessment against competing models

13. **Same unseen test set for every model.** If a competitor was trained on
    public recordings that overlap the test set, its number is partly a training
    error and is optimistic (p. 31: training error is almost always lower). Build
    the comparison set from reciters and sources unlikely to be in any model's
    training data, and state what is known about overlap. **(inference)**
14. **No free lunch** (p. 27). A win on one test set does not transfer to all
    conditions. Report per condition (reciter group, recording quality, tempo,
    rule type) and do not summarise as "better" without naming the population.
15. **Test sets must be large enough for the rate being claimed.** Figure 2.17
    uses 5,000 test observations and still attributes jumpiness to small sample
    size (of the training set). The lab's 127-row example shows a nominal 20%
    rate realised as 16.54% to 29.13%. A false-flag rate near zero needs
    thousands of correct test words to be measured at all. As a rough guide from
    outside the book: with zero false flags in n correct words, rates much below
    about 3/n cannot be claimed. **(inference; last sentence is not from ISLP)**
16. **Data hygiene from the lab** (pp. 55-56). Check dtypes of feature tables,
    how missing values are coded, and how many rows are dropped. Words where
    forced alignment failed are not missing at random; dropping them silently
    flatters the model. Count and report them. **(inference)**
17. **Variance estimates:** `np.var` divides by n by default; set `ddof=1` when a
    sample variance is intended (p. 47).

---

## Chapter 3: Linear Regression (pp. 69-134)

Opening (pp. 69-70): seven questions about the Advertising data: (1) is there a
relationship; (2) how strong; (3) which media; (4) how large is each
association; (5) how accurately can future sales be predicted; (6) is it linear;
(7) is there synergy (interaction). Linear regression "is still a useful and
widely used statistical learning method" and the base for later methods.

### 3.1 Simple Linear Regression (pp. 70-80)

- Equation (3.1), p. 70: `Y ~ b0 + b1 X` ("is approximately modeled as");
  regressing Y on X. b0 intercept, b1 slope; together coefficients or parameters.
- Equation (3.2), p. 71: `y_hat = b0_hat + b1_hat x`.

#### 3.1.1 Estimating the Coefficients (pp. 71-72)

- Residual: `e_i = y_i - y_hat_i`.
- Equation (3.3), p. 71: `RSS = sum_i (y_i - b0_hat - b1_hat x_i)^2`.
- Equation (3.4), p. 71 (least squares estimates):
  `b1_hat = sum_i (x_i - x_bar)(y_i - y_bar) / sum_i (x_i - x_bar)^2`,
  `b0_hat = y_bar - b1_hat x_bar`.
- Advertising, sales on TV (n = 200): b0_hat = 7.03, b1_hat = 0.0475. An extra
  $1,000 of TV advertising is associated with approximately 47.5 more units (p. 72).
- Figure 3.1 (p. 72): the fit with residual segments; it "overestimates the
  trend in the left of the plot". Figure 3.2 (p. 73): RSS contours with the
  minimum at the least squares estimates.

#### 3.1.2 Assessing the Accuracy of the Coefficient Estimates (pp. 72-77)

- Equation (3.5), p. 72: `Y = b0 + b1 X + e`, the population regression line,
  "the best linear approximation to the true relationship". The error term
  absorbs non-linearity, omitted variables and measurement error.
- Footnote 1 (p. 73): "we seldom believe that the true relationship is linear".
- Equation (3.6), p. 73: simulation `Y = 2 + 3X + e` with 100 random X.
  Figure 3.3 (p. 74): ten least squares lines from ten data sets differ, but
  average close to the population line.
- Unbiasedness (p. 74): an unbiased estimator "does not systematically over- or
  under-estimate the true parameter". Least squares estimates are unbiased.
- Equation (3.7), p. 75: `Var(mu_hat) = SE(mu_hat)^2 = sigma^2 / n` (holds when
  the n observations are uncorrelated, footnote 2).
- Equation (3.8), p. 75:
  `SE(b0_hat)^2 = sigma^2 * [1/n + x_bar^2 / sum_i (x_i - x_bar)^2]`,
  `SE(b1_hat)^2 = sigma^2 / sum_i (x_i - x_bar)^2`, with sigma^2 = Var(e).
  Strictly valid when errors have common variance and are uncorrelated.
- "SE(b1_hat) is smaller when the x_i are more spread out; intuitively we have
  more leverage to estimate a slope when this is the case" (p. 75).
- Residual standard error estimates sigma: `RSE = sqrt(RSS / (n - 2))`.
- 95% confidence interval: a range that, over repeated samples, contains the
  true value 95% of the time. Equations (3.9)-(3.11), pp. 75-76:
  `b1_hat +/- 2 SE(b1_hat)`, `b0_hat +/- 2 SE(b0_hat)`.
- Footnote 3 (p. 76): approximate because it assumes Gaussian errors, and the
  factor 2 should be the 97.5% quantile of a t-distribution with n - 2 degrees
  of freedom.
- Advertising: 95% CI for b0 [6.130, 7.935]; for b1 [0.042, 0.053] (p. 76).
- Hypothesis test, (3.12)-(3.13): H0 no relationship (b1 = 0) against Ha (b1 != 0).
- Equation (3.14), p. 76: `t = (b1_hat - 0) / SE(b1_hat)`; t-distribution with
  n - 2 degrees of freedom under H0; close to normal for n above about 30.
- p-value: probability of a value at least as large as |t| if b1 = 0. Typical
  cut-offs 5% or 1%; for n = 30 these correspond to t of around 2 and 2.75 (p. 77).
- Table 3.1 (p. 77), sales on TV:

  | | Coefficient | Std. error | t-statistic | p-value |
  |---|---|---|---|---|
  | Intercept | 7.0325 | 0.4578 | 15.36 | < 0.0001 |
  | TV | 0.0475 | 0.0027 | 17.67 | < 0.0001 |

#### 3.1.3 Assessing the Accuracy of the Model (pp. 77-80)

- Table 3.2 (p. 78): RSE 3.26; R^2 0.612; F-statistic 312.1.
- Equation (3.15), p. 78: `RSE = sqrt(RSS / (n - 2))`; (3.16): `RSS = sum_i (y_i - y_hat_i)^2`.
  RSE is "an estimate of the standard deviation of e", an absolute measure of
  lack of fit in units of Y.
- Example: RSE 3.26 means sales deviate from the true line by about 3,260 units
  on average; mean sales about 14,000 units; percentage error 3,260/14,000 = 23%.
- Equation (3.17), p. 79: `R^2 = (TSS - RSS)/TSS = 1 - RSS/TSS`, with
  `TSS = sum_i (y_i - y_bar)^2`. Proportion of variance explained, between 0 and 1.
- A low R^2 can mean the linear model is wrong, or sigma^2 is high, or both.
- What counts as a good R^2 depends on the field: near 1 in some physics
  problems; "well below 0.1 might be more realistic" in biology, psychology,
  marketing (p. 79).
- Equation (3.18), p. 79: sample correlation Cor(X, Y). In simple regression
  R^2 = r^2 (p. 80).

### 3.2 Multiple Linear Regression (pp. 80-90)

- Table 3.3 (p. 80), separate simple regressions:

  | Model | Term | Coefficient | Std. error | t | p-value |
  |---|---|---|---|---|---|
  | sales on radio | Intercept | 9.312 | 0.563 | 16.54 | < 0.0001 |
  | | radio | 0.203 | 0.020 | 9.92 | < 0.0001 |
  | sales on newspaper | Intercept | 12.351 | 0.621 | 19.88 | < 0.0001 |
  | | newspaper | 0.055 | 0.017 | 3.30 | 0.00115 |

- Separate regressions are unsatisfactory (p. 81): no single prediction, and
  each ignores the other predictors, which misleads when predictors are correlated.
- Equation (3.19), p. 81: `Y = b0 + b1 X1 + ... + bp Xp + e`. b_j is "the
  average effect on Y of a one unit increase in X_j, holding all other
  predictors fixed". (3.20) is the Advertising instance.

#### 3.2.1 Estimating the Regression Coefficients (pp. 81-83)

- (3.21) prediction formula; (3.22) RSS minimised by least squares. Figure 3.4
  (p. 82): with two predictors the fit is a plane.
- Table 3.4 (p. 83), sales on TV, radio, newspaper:

  | | Coefficient | Std. error | t | p-value |
  |---|---|---|---|---|
  | Intercept | 2.939 | 0.3119 | 9.42 | < 0.0001 |
  | TV | 0.046 | 0.0014 | 32.81 | < 0.0001 |
  | radio | 0.189 | 0.0086 | 21.89 | < 0.0001 |
  | newspaper | -0.001 | 0.0059 | -0.18 | 0.8599 |

- Table 3.5 (p. 83), correlations: TV-radio 0.0548; TV-newspaper 0.0567;
  TV-sales 0.7822; radio-newspaper 0.3541; radio-sales 0.5762;
  newspaper-sales 0.2283.
- Lesson (pp. 82-83): newspaper is significant alone but not jointly. It is "a
  surrogate for radio advertising" and "gets 'credit'" for radio's association.
  Analogy: shark attacks and ice cream sales both follow temperature.

#### 3.2.2 Some Important Questions (pp. 83-90)

One: is there any relationship? (pp. 84-86)

- H0: b1 = ... = bp = 0. Equation (3.23), p. 84:
  `F = [(TSS - RSS)/p] / [RSS/(n - p - 1)]`.
  E{RSS/(n - p - 1)} = sigma^2; under H0, E{(TSS - RSS)/p} = sigma^2, so F is
  near 1 under H0 and above 1 otherwise.
- Table 3.6 (p. 85): RSE 1.69; R^2 0.897; F-statistic 570.
- How large F must be depends on n and p. F follows an F-distribution under H0
  with normal errors, approximately so for large n (footnote 6).
- Equation (3.24), p. 85, testing q coefficients:
  `F = [(RSS0 - RSS)/q] / [RSS/(n - p - 1)]`, RSS0 from the model without them.
  Each t-test equals the q = 1 F-test (the t-statistic squared, footnote 7).
- Warning (p. 86): with p = 100 and H0 true, about 5% of individual p-values
  fall below 0.05 by chance, so about five "small" p-values are expected and at
  least one is likely. The F-statistic adjusts for the number of predictors.
- If p > n least squares cannot be fitted and the F-statistic cannot be used.

Two: deciding on important variables (pp. 86-87)

- Variable selection. Criteria named: Mallow's Cp, AIC, BIC, adjusted R^2 (Ch. 6).
- 2^p subsets: 4 for p = 2; 2^30 = 1,073,741,824 for p = 30.
- Forward selection: start from the null model, add the variable giving the
  lowest RSS, repeat. Backward selection: start with all, remove the largest
  p-value, repeat. Mixed selection: forward steps with removal when a p-value
  rises above a threshold.
- Backward selection cannot be used if p > n; forward always can. Forward is
  greedy and may keep variables that later become redundant.

Three: model fit (pp. 88-89)

- In multiple regression R^2 = Cor(Y, Y_hat)^2.
- "R^2 will always increase when more variables are added to the model, even if
  those variables are only weakly associated with the response", because
  training RSS always falls "(though not necessarily the testing data)" (p. 88).
- R^2 with all three media 0.8972; with TV and radio only 0.89719. TV only 0.61.
- RSE: TV and radio 1.681; adding newspaper 1.686; TV only 3.26.
- Equation (3.25), p. 88: `RSE = sqrt(RSS / (n - p - 1))`. RSE can rise when a
  variable is added if the fall in RSS is small relative to the rise in p.
- "Graphical summaries can reveal problems with a model that are not visible
  from numerical statistics." Figure 3.5 (p. 89): the plane overestimates sales
  when budget goes to one medium and underestimates when split, a sign of interaction.

Four: predictions (pp. 89-90)

- Three sources of uncertainty: (1) coefficient estimates (reducible error),
  quantified by a confidence interval; (2) model bias from assuming linearity,
  which the book then ignores, operating "as if the linear model were correct";
  (3) the random error e (irreducible), which prediction intervals add.
- "Prediction intervals are always wider than confidence intervals, because
  they incorporate both the error in the estimate for f(X) (the reducible error)
  and the uncertainty as to how much an individual point will differ from the
  population regression plane (the irreducible error)" (p. 90).
- Example with $100,000 on TV and $20,000 on radio: 95% confidence interval for
  average sales [10,985, 11,528]; 95% prediction interval for one city
  [7,930, 14,580]; both centred at 11,256.
- Confidence interval: uncertainty about the average response over many cities.
  Prediction interval: uncertainty about a particular city.

### 3.3 Other Considerations in the Regression Model (pp. 91-109)

#### 3.3.1 Qualitative Predictors (pp. 91-94)

- Credit data (Figure 3.6, p. 92): response balance; quantitative predictors
  age, cards, education, income, limit, rating; qualitative own, student,
  status, region.
- Two levels: one dummy variable, (3.26); model (3.27). b0 is the mean for the
  0 group, b1 the mean difference. Footnote 10: dummy coding is "one-hot encoding".
- Table 3.7 (p. 93): Intercept 509.80 (SE 33.13, t 15.389, p < 0.0001);
  own[Yes] 19.73 (SE 46.05, t 0.429, p 0.6690). No evidence of a difference.
- Coding choice does not change the fit or predictions, only the interpretation.
  With +1/-1 coding: b0 estimate $519.665, b1 estimate $9.865.
- More than two levels: one fewer dummy than levels, (3.28)-(3.30); the level
  without a dummy is the baseline.
- Table 3.8 (p. 94): Intercept 531.00 (SE 46.32, t 11.464); region[South] -12.50
  (SE 56.68, t -0.221, p 0.8260); region[West] -18.69 (SE 65.02, t -0.287,
  p 0.7740). The prose on p. 94 attributes $18.69 to South and $12.50 to West,
  the reverse of the table as printed.
- Individual dummy coefficients and p-values depend on the coding. Use an F-test
  of all dummies together, which does not; here its p-value is 0.96.

#### 3.3.2 Extensions of the Linear Model (pp. 94-99)

- Two restrictive assumptions: additive (the association of X_j with Y does not
  depend on other predictors) and linear (constant change per unit).
- Interaction term, equation (3.31), p. 95: `Y = b0 + b1 X1 + b2 X2 + b3 X1 X2 + e`;
  rewritten (3.32) as `b0 + (b1 + b3 X2) X1 + b2 X2 + e`, so the slope of X1
  depends on X2.
- Factory example (p. 96): `units ~ 1.2 + 3.4 lines + 0.22 workers + 1.4 (lines x workers)`;
  an extra line adds 3.4 + 1.4 x workers.
- Advertising with interaction (3.33). Table 3.9 (p. 96): Intercept 6.7502
  (SE 0.248, t 27.23); TV 0.0191 (0.002, 12.70); radio 0.0289 (0.009, 3.24,
  p 0.0014); TV x radio 0.0011 (SE printed 0.000, t 20.73, p < 0.0001).
- R^2 96.8% with interaction against 89.7% without: (96.8 - 89.7)/(100 - 89.7)
  = 69% of the remaining variability explained. An extra $1,000 of TV is
  associated with 19 + 1.1 x radio units; of radio, 29 + 1.1 x TV units.
- Hierarchical principle (pp. 96-97): "if we include an interaction in a model,
  we should also include the main effects, even if the p-values associated with
  their coefficients are not significant."
- Qualitative by quantitative interaction: without it, (3.34), parallel lines
  with different intercepts; with it, (3.35), different intercepts and slopes.
  Figure 3.7 (p. 98): students have a lower slope of balance on income.
- Polynomial regression, equation (3.36), p. 98:
  `mpg = b0 + b1 horsepower + b2 horsepower^2 + e`; still a linear model.
- Table 3.10 (p. 99): Intercept 56.9001 (SE 1.8004, t 31.6); horsepower -0.4662
  (0.0311, -15.0); horsepower^2 0.0012 (0.0001, 10.1). R^2 0.688 quadratic
  against 0.606 linear. Figure 3.8 (p. 99): the degree-5 fit "seems
  unnecessarily wiggly".

#### 3.3.3 Potential Problems (pp. 100-109)

"Identifying and overcoming these problems is as much an art as a science" (p. 100).

1. **Non-linearity** (pp. 100-101). If the truth is far from linear, "virtually
   all of the conclusions that we draw from the fit are suspect". Tool: residual
   plot (residuals against the predictor, or against fitted values in multiple
   regression). Ideally no pattern. Figure 3.9 (p. 100): clear U-shape for the
   linear fit of mpg on horsepower; little pattern after adding the square.
   Remedy: transformations such as log X, sqrt X, X^2.
2. **Correlation of error terms** (pp. 101-102). Standard errors assume
   uncorrelated errors. If errors are correlated, estimated standard errors
   "tend to underestimate the true standard errors"; confidence and prediction
   intervals are too narrow; p-values too low; "we may have an unwarranted sense
   of confidence in our model". Extreme case: data accidentally doubled gives
   intervals narrower by a factor of sqrt(2). Common in time series: adjacent
   residuals show "tracking". Figure 3.10 (p. 102): residuals with adjacent
   correlation 0.0, 0.5, 0.9. Also arises outside time series, for example
   members of the same family. "Good experimental design is crucial."
3. **Non-constant variance (heteroscedasticity)** (p. 103). Standard errors,
   intervals and tests rely on Var(e_i) = sigma^2. Sign: a funnel shape in the
   residual plot. Remedy: transform Y with a concave function (log Y, sqrt Y).
   Figure 3.11 (p. 103): funnel removed after log Y. If response i is an average
   of n_i raw observations, use weighted least squares with weights w_i = n_i.
4. **Outliers** (pp. 103-104): y_i far from the predicted value. Figure 3.12
   (p. 104): removing the outlier barely moves the line, but RSE is 1.09 with it
   and 0.77 without; R^2 falls from 0.892 to 0.805 when it is included. Since
   RSE feeds all intervals and p-values, one point matters. Studentized
   residuals (residual over its estimated standard error) above 3 in absolute
   value are possible outliers; the example outlier exceeds 6, the others lie
   between -2 and 2. Removal is justified for recording errors, but "care should
   be taken, since an outlier may instead indicate a deficiency with the model,
   such as a missing predictor".
5. **High-leverage points** (pp. 104-106): unusual x_i. Figure 3.13 (p. 105):
   removing a high-leverage point changes the line much more than removing an
   outlier. "It is cause for concern if the least squares line is heavily
   affected by just a couple of observations." In multiple regression a point
   can be ordinary in each predictor and unusual jointly. Equation (3.37), p. 105:
   `h_i = 1/n + (x_i - x_bar)^2 / sum_i' (x_i' - x_bar)^2`. h_i lies between 1/n
   and 1; the average is (p + 1)/n; a value greatly exceeding that suggests high
   leverage. A point that is both an outlier and high leverage "is a
   particularly dangerous combination!" (p. 106).
6. **Collinearity** (pp. 106-109): predictors closely related to each other.
   Figure 3.14 (p. 106): limit and age unrelated; limit and rating highly
   collinear. Figure 3.15 (p. 107): with collinearity the RSS contours form a
   narrow valley; many coefficient pairs fit equally well. Plausible range of
   the limit coefficient goes from about 0.15-0.20 to about -0.2 to 0.2, an
   eight-fold increase. Table 3.11 (p. 108):

   | Model | Term | Coefficient | Std. error | t | p-value |
   |---|---|---|---|---|---|
   | 1 | Intercept | -173.411 | 43.828 | -3.957 | < 0.0001 |
   | 1 | age | -2.292 | 0.672 | -3.407 | 0.0007 |
   | 1 | limit | 0.173 | 0.005 | 34.496 | < 0.0001 |
   | 2 | Intercept | -377.537 | 45.254 | -8.343 | < 0.0001 |
   | 2 | rating | 2.202 | 0.952 | 2.312 | 0.0213 |
   | 2 | limit | 0.025 | 0.064 | 0.384 | 0.7012 |

   The standard error of the limit coefficient grows 12-fold; its importance is
   "masked". Collinearity lowers the power of the test. Detection: correlation
   matrix, but multicollinearity among three or more variables can hide from
   it; better the variance inflation factor (unnumbered formula, p. 108):
   `VIF(b_j) = 1 / (1 - R^2 of X_j regressed on the other predictors)`.
   Minimum 1; "a VIF value that exceeds 5 or 10 indicates a problematic amount
   of collinearity". Credit example (p. 109): VIFs 1.01, 160.67, 160.59 for age,
   rating, limit. Remedies: drop one variable (R^2 falls only from 0.754 to
   0.75), or combine them, for example the average of standardized versions.

### 3.4 The Marketing Plan (pp. 109-111)

Answers to the seven questions:

1. Relationship: yes, F-statistic p-value very low.
2. Strength: RSE 1.69 against mean response 14.022, roughly 12% error; R^2
   almost 90%.
3. Which media: TV and radio; not newspaper.
4. Size: 95% confidence intervals (0.043, 0.049) for TV, (0.172, 0.206) for
   radio, (-0.013, 0.011) for newspaper. VIFs 1.005, 1.145, 1.145, so
   collinearity does not explain the newspaper interval.
5. Prediction accuracy: prediction interval for an individual response,
   confidence interval for the average response.
6. Linearity: residual plots; transformations.
7. Synergy: the interaction term raises R^2 from around 90% to almost 97%.

### 3.5 Comparison of Linear Regression with K-Nearest Neighbors (pp. 111-115)

- Parametric methods are easy to fit, interpretable and testable, but assume a
  form for f. If the form is far from the truth they perform poorly.
- KNN regression (unnumbered formula, p. 111):
  `f_hat(x0) = (1/K) * sum over x_i in N0 of y_i`.
- Figure 3.16 (p. 112): 64 observations, p = 2; K = 1 interpolates as a rough
  step function; K = 9 is smoother. Small K: low bias, high variance. Large K:
  smoother, possibly biased.
- "The parametric approach will outperform the non-parametric approach if the
  parametric form that has been selected is close to the true form of f" (p. 112).
- Figures 3.17-3.18 (p. 113): truth linear, 50 observations. K = 1 far too
  variable. KNN test MSE is above that of linear regression for all K; close
  only for large K. A non-parametric approach "incurs a cost in variance that is
  not offset by a reduction in bias".
- Figure 3.19 (p. 114): slightly non-linear truth: linear still better for low
  K, but for K >= 4 KNN outperforms. Strongly non-linear truth: KNN
  substantially better for all K.
- Figure 3.20 (p. 115): the strongly non-linear case with extra noise
  predictors, p = 1, 2, 3, 4, 10, 20. KNN wins for p = 1 or 2; mixed for p = 3;
  linear regression superior for p >= 4. Linear test MSE deteriorates slightly;
  KNN's rises more than ten-fold. With 50 training observations spread over 20
  dimensions a point "has no nearby neighbors": the curse of dimensionality.
- "As a general rule, parametric methods will tend to outperform non-parametric
  approaches when there is a small number of observations per predictor" (p. 115).
- Even in low dimension one may prefer linear regression for interpretability
  and available p-values if KNN's test MSE is only slightly lower.

### 3.6 Lab: Linear Regression (pp. 116-127)

- Tools: `statsmodels` (`sm.OLS`), `variance_inflation_factor`, `anova_lm`, and
  from the book's package `load_data`, `ModelSpec` (as `MS`), `summarize`, `poly`.
- `sm.OLS(y, X)` specifies the model; `.fit()` fits it (p. 118).
- Boston data: medv for 506 neighbourhoods. The text says "13 predictors" on
  p. 117 and "12 variables" on p. 122; the column listings differ accordingly
  (both as printed).
- medv on lstat (p. 118): intercept 34.5538 (SE 0.563, t 61.415); lstat -0.9500
  (SE 0.039, t -24.528).
- **Fit then transform** (pp. 118-119): a transform object's `fit()` may compute
  "means and standard deviations for centering and scaling" from the data it is
  given; `transform()` applies the fitted transformation, including to new data
  (`design.transform(new_df)`, p. 120).
- Predictions at lstat 5, 10, 15 (p. 120): 29.80359411, 25.05334734, 20.30310057.
  Confidence intervals `conf_int(alpha=0.05)`: [29.00741194, 30.59977628],
  [24.47413202, 25.63256267], [19.73158815, 20.87461299]. Prediction intervals
  `conf_int(obs=True, alpha=0.05)`: [17.56567478, 42.04151344],
  [12.82762635, 37.27906833], [8.0777421, 32.52845905]. The text summarises
  lstat = 10 as CI (24.47, 25.63) and PI (12.82, 37.28), centred at 25.05.
- Diagnostics (pp. 121-122): residuals against fitted values; leverage from
  `get_influence().hat_matrix_diag`; largest leverage at index 374.
- medv on lstat and age (p. 122): intercept 33.2228 (0.731, 45.458); lstat
  -1.0321 (0.048, -21.416); age 0.0345 (0.012, 2.826, p 0.005).
- All predictors (p. 123): for example indus 0.0135 (p 0.829) and age 0.0036
  (p 0.787) are not significant; lstat -0.5520 (t -10.897); rm 3.6581 (t 8.705).
- `results.rsquared` gives R^2; `np.sqrt(results.scale)` gives RSE (p. 123).
- VIFs (p. 124): crim 1.767, zn 2.298, indus 3.987, chas 1.071, nox 4.369,
  rm 1.913, age 3.088, dis 3.954, rad 7.445, tax 9.002, ptratio 1.797, lstat 2.871.
- Interaction (pp. 124-125): lstat -1.3921 (t -8.313); age -0.0007 (p 0.971);
  lstat:age 0.0042 (t 2.244, p 0.025). The main effect stays despite its p-value.
- Polynomial (p. 125): `poly('lstat', degree=2)` uses orthogonal polynomials by
  default; `raw=True` gives lstat and lstat^2 with identical fitted values.
  Quadratic term t = 13.315.
- Nested model comparison (p. 125): `anova_lm(results1, results3)`: residual df
  503.0 and 502.0; SSR 19168.13 and 14165.61; F = 177.28; p = 7.47e-35. F is the
  square of the quadratic term's t-statistic.
- Qualitative predictors (pp. 126-127): Carseats, 400 locations. ShelveLoc
  (Bad, Medium, Good) is one-hot encoded with the first level dropped to avoid
  collinearity with the intercept. ShelveLoc[Good] 4.8487 (t 31.724);
  ShelveLoc[Medium] 1.9533 (t 15.531); Income:Advertising 0.0008 (p 0.007);
  Price:Age 0.0001 (p 0.424).

### 3.7 Exercises (pp. 127-134), skimmed

- Conceptual: null hypotheses behind Table 3.4; KNN classifier against KNN
  regression; interpreting a model with interactions (3); training against test
  RSS for linear and cubic fits when the truth is or is not linear (4); fitted
  values as linear combinations of responses, eq. (3.38) (5); the line passes
  through (x_bar, y_bar) (6); R^2 = r^2 (7).
- Applied: Auto regressions with intervals and diagnostics (8-9); Carseats with
  qualitative predictors (10); t-statistics without intercept (11-12);
  simulation with varying noise, eq. (3.39) `Y = -1 + 0.5X + e` (13);
  collinearity simulation plus one mismeasured point that may be an outlier, a
  high-leverage point or both (14); Boston simple against multiple regression
  coefficients and cubic terms (15).
- The formula in exercise 11(d) is badly scrambled in the extraction; it was not
  relied on.

### Application to Qaf-ft (Chapter 3)

Chapter 3 is most directly useful for the tajweed duration scorer and for the
design and checking of the verifier's features. The verifier's binary output
itself calls for a classification model (Ch. 4); the feature-level lessons here
still apply.

A. Modelling madd duration against reciter tempo

1. **Set up the regression explicitly.** Observation: one madd instance in
   recitation verified as correct by an expert listener. Response: its duration.
   Predictors: a tempo measure for that reciter and passage (for example the
   typical short-vowel duration), the madd type as a qualitative predictor, and
   position effects such as a stop at verse end. The assumption that correct
   madd length scales with the reciter's own pace is a domain assumption to
   confirm against the project's tajweed specification.
2. **Spread the tempo values** (eq. 3.8, p. 75). The slope's standard error
   shrinks as x is more spread out. If the training reciters all recite at a
   similar slow pace, the slope is poorly determined and fast reciters are
   extrapolation. Collect correct recitation across the full tempo range.
3. **Let slope and intercept differ by madd type** (eq. 3.35, pp. 97-98). An
   additive model (3.34) forces parallel lines. A tempo by type interaction
   gives each type its own line. Keep main effects when including the
   interaction (hierarchical principle). Test the type factor and the
   interaction with an F-test on the whole set of dummies (eq. 3.24; p. 94), not
   by reading individual dummy p-values, which depend on the baseline.
4. **Report the fit in interpretable terms.** Slope with its 95% confidence
   interval (eq. 3.9); RSE in milliseconds and as a share of mean duration, as
   the authors do (23% and 12% in their examples); R^2. If tajweed prescribes a
   ratio, check whether the slope's interval contains it. **(inference)**

B. Confidence interval against prediction interval: the central point for scoring

5. **Flag individual instances with a prediction interval, never a confidence
   interval** (§3.2.2 pp. 89-90; lab p. 120). The reciter's single madd is an
   individual response Y = f(X) + e. In the book's example the confidence
   interval spans 543 units and the prediction interval 6,650, about twelve
   times wider. Using the narrow one would flag a large share of correct
   recitation. In code: `get_prediction(newX).conf_int(obs=True, alpha=...)`.
6. **Use the confidence interval for averages.** Statements about a reciter's
   mean madd length over many instances, or about the population line, are
   average-response statements. This gives two feedback levels: per-instance
   flags (wide band, rare) and per-reciter tendency (narrow band, more
   sensitive). **(inference)**
7. **A 95% band is a 5% false-flag rate.** "95% of intervals of this form will
   contain the true value of Y" (p. 90), so flagging outside it flags about one
   correct madd in twenty. A very low false-flag rate needs a far wider band,
   and then the tails matter. The interval formulas assume Gaussian errors
   (footnote 3, p. 76), constant variance and uncorrelated errors, and ignore
   model bias (p. 90, item 2). Do not trust the nominal level: measure the
   actual share of held-out correct instances from unseen reciters that fall
   outside the band, and set the band from that. **(inference)**
8. **The same distinction applies to reporting Qaf-ft's own accuracy.** A tight
   confidence interval on mean accuracy over many test reciters says little
   about what one new reciter will experience. Also report the spread across
   reciters, including the worst group. **(inference from p. 90)**

C. Diagnostics checklist for the duration model (§3.3.3)

9. **Non-linearity.** Plot residuals against fitted values and against tempo. A
   U-shape means the straight line is wrong; try log or square-root transforms
   or a quadratic term, and compare nested models with `anova_lm` (lab p. 125).
   Avoid high-degree polynomials (degree 5 was "unnecessarily wiggly", p. 99).
10. **Non-constant variance.** Longer madds plausibly vary more in milliseconds;
    look for a funnel. If present, model log duration (p. 103). With a funnel
    left uncorrected, a constant-width band over-flags long madds and
    under-flags short ones. If fitting to per-reciter averages built from
    unequal numbers of instances, use weighted least squares with weights n_i.
11. **Correlated errors: the largest risk to honest assessment.** Madd instances
    from one reciter, one session or one passage share errors, like the book's
    family example; consecutive instances within a recitation can "track" as
    tempo drifts. Consequences stated by the book: standard errors too small,
    intervals too narrow, p-values too low, "unwarranted sense of confidence".
    Actions: plot residuals in time order within a recitation; treat the
    reciter as the sampling unit when computing uncertainty; hold out whole
    reciters. The same warning covers every confidence interval quoted when
    comparing Qaf-ft with competitors: thousands of words from a dozen reciters
    are closer to a dozen observations than to thousands. **(inference)**
12. **Outliers.** Studentized residuals beyond 3 in absolute value are
    candidates. In the training data for the reference model they have two
    sources: alignment or labelling errors (remove) and model deficiency such as
    a missing predictor (fix the model, p. 104). They must be handled, because a
    few outliers inflate RSE (1.09 against 0.77 in the book), which widens every
    band and hides real mistakes. In deployment the roles reverse: a genuine
    lengthening mistake is an outlier relative to the correct-recitation model,
    so the scorer is, in effect, a studentized-residual test. **(inference)**
13. **High leverage.** One or two reciters at extreme tempo can set the slope
    for everyone. Compute leverage (`hat_matrix_diag`; compare with (p + 1)/n),
    refit leaving each reciter out, and see how far the line moves, as the book
    does with observation 41. At scoring time, a reciter whose tempo lies
    outside the training range is in high-leverage territory; prefer to abstain
    from flagging there. Check joint unusualness, since a point can be ordinary
    in each predictor and unusual in combination. **(inference)**
14. **Outlier and high leverage together** is "a particularly dangerous
    combination" (p. 106): a mislabelled recording from an extreme-tempo reciter
    in the training set deserves manual review before anything else.

D. Collinearity among verifier features

15. **Expect it.** Alignment score, posterior margin and entropy are all
    computed from the same CTC posteriors and will be strongly correlated. The
    book's consequences: inflated standard errors (12-fold in Table 3.11),
    unstable coefficients, and a truly useful variable appearing unimportant.
    Compute the correlation matrix and VIFs (problematic above 5 or 10; pairs
    can look fine while three variables are jointly collinear).
16. **What to do.** Do not read individual coefficients as feature importance
    when VIFs are high. Either drop a redundant feature (fit barely changed in
    the book: 0.754 to 0.75) or combine standardized features into one score.
    Coefficients sliding along the "valley" from one training sample to the next
    are a stability risk for a deployed threshold. **(inference)**
17. **Surrogates.** A feature that predicts mistakes on its own may only stand
    in for another, as newspaper did for radio. Word duration alone may proxy
    tempo or word length. Fit features jointly and compare with the one-at-a-time
    results (exercise 15 pattern).

E. Model building and selection

18. **Training fit always improves when a feature is added** (p. 88). R^2 rose
    from 0.89719 to 0.8972 with a useless variable. Never justify a verifier
    feature by training fit; require an improvement on held-out reciters.
19. **Many comparisons produce false discoveries** (p. 86). With many tajweed
    rules, features, chapters and reciter groups, about 5% of null comparisons
    will show p < 0.05. For the competitor comparison, fix one primary metric
    and test set in advance; treat per-rule wins as exploratory.
20. **Standardisation must be fitted on training data only** (lab pp. 118-120).
    Fit scaling constants on the training reciters, freeze them, and apply the
    same transform to validation, test and production inputs.
21. **Qualitative predictors.** Madd type, recitation style and recording source
    enter as dummy variables with a baseline; one fewer than the number of
    levels (p. 93).

F. Parametric against KNN for these models (§3.5)

22. **Duration against tempo is the low-dimensional case where KNN is a fair
    check.** With one or two predictors, compare held-out MSE of the linear
    model with KNN regression over a range of K. If KNN with moderate K clearly
    wins, the relationship is non-linear (Figure 3.19 pattern) and the linear
    form needs a transformation. If linear wins or ties, keep it for
    interpretability (p. 115). Choose K on held-out data only.
23. **Do not use KNN for the verifier.** With four or more features and few
    mistake examples, the book's result is that KNN degrades sharply (linear
    superior for p >= 4 with 50 observations; more than ten-fold MSE increase),
    and "parametric methods will tend to outperform non-parametric approaches
    when there is a small number of observations per predictor".
24. **Interpretability has value in itself here.** The reciter needs a reason
    with each flag ("held for about 1.3 beats where 2 are expected"). A linear
    model with a few coefficients provides it.

---

## Cross-chapter assessment checklist for Qaf-ft

| # | Check | Basis |
|---|---|---|
| 1 | All reported numbers are on reciters and sources absent from training, for every model compared | §2.2.1, pp. 28-31 |
| 2 | Test sets contain real deviations from the canonical text, not only correct recitation | §2.2.1 (inference) |
| 3 | Model size, checkpoint and thresholds chosen on held-out data across at least three flexibility levels | §2.2.1-2.2.2 |
| 4 | Spread of held-out error across reciter subsets and seeds is reported (variance) | §2.2.2, p. 32 |
| 5 | A human-agreement floor is estimated and shown beside model results | eq. 2.3, eq. 2.11 (inference) |
| 6 | False-flag rate and miss rate are reported separately; 0.5 is not assumed | §2.2.3 (inference) |
| 7 | Uncertainty is computed with the reciter as the unit | §3.3.3 item 2 |
| 8 | Per-instance flags use prediction intervals with empirically verified coverage | §3.2.2, p. 90 |
| 9 | Residual, leverage and VIF diagnostics are run and archived for each regression | §3.3.3; lab pp. 121-124 |
| 10 | One primary metric fixed in advance; sub-group results labelled exploratory | §3.2.2, p. 86 |
| 11 | Counts of dropped items (failed alignments, missing features) are reported | lab §2.3.7 (inference) |
| 12 | Seeds and package versions are recorded | pp. 11-12; lab p. 47 |

---

## Discrepancies and unreadable items in the source text

- Bayes error rate for the Figure 2.13 data: 0.133 (p. 36) and 0.1304 (p. 37).
- Table 3.8 against the prose on p. 94: the South and West values are swapped.
- Boston data: "13 predictors" (p. 117) against "12 variables" (p. 122).
- Footnote 3 (p. 76), p. 81 and exercise 11 refer to R; leftovers from the R edition.
- Table 3.9: the interaction's standard error prints as 0.000 (rounded).
- Lab p. 120: the text rounds 12.82762635 to 12.82.
- Figures: graphics unreadable throughout; only captions and axis labels used.
- Symbols: epsilon appears as `"`, "not equal" as `%=`; the marker for harder
  sections (p. 12) is missing.
- Exercise 3.7 no. 11(d): formula scrambled; not transcribed.
