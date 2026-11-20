# ISLP study notes: Chapters 4, 5, 6 (for Qaf-ft)

Source: `refs/islp_text/islp.txt`, lines 8735-18740 (pdftotext extraction of James, Witten, Hastie, Tibshirani, Taylor, *An Introduction to Statistical Learning with Applications in Python*, Springer 2023). Page numbers are the book's own (running headers). Chapter 4 = pp. 135-199, Chapter 5 = pp. 201-228, Chapter 6 = pp. 229-288.

Conventions used in these notes

- "ISLP says" / a page citation = stated in the text I read. Numbers are quoted exactly as printed.
- "(my arithmetic)" = a number I derived from numbers in the text; not printed in the book.
- "Extension (not in ISLP)" = my own application or a standard technique the book does not cover in these chapters. Treat these as design suggestions, not as citations.
- Figures are garbled in the extraction (axes and point clouds only). Everything said about a figure below comes from its caption and the surrounding prose, never from the plotted values. Where an equation was garbled I say so.

Two things the task brief asked about that are NOT in these chapters under that name:

1. There is no section titled "the right and wrong way to cross-validate" in ISLP Chapter 5 (that title belongs to *The Elements of Statistical Learning*, §7.10.2). The same principle is stated in ISLP in §6.1.1 (p. 231), §6.1.3 (p. 239), and most explicitly in Lab §6.5.1 (p. 271). See "Where ISLP states the right-way/wrong-way rule" in the Chapter 5 notes.
2. The phrase "class imbalance" never appears. The topic is treated through the Default data (3.33 % positives): the null classifier, class-specific error rates, threshold moving, ROC (§4.4.2, pp. 152-156) and the Caravan lab (6 % positives; pp. 184-188).

---

# Chapter 4. Classification (pp. 135-199)

## 4.0 Chapter opening (p. 135)

- Qualitative = categorical response. Predicting it = classification. "Often the methods used for classification first predict the probability that the observation belongs to each of the categories of a qualitative variable, as the basis for making the classification. In this sense they also behave like regression methods."
- Classifiers covered: logistic regression, linear discriminant analysis (LDA), quadratic discriminant analysis (QDA), naive Bayes, K-nearest neighbors (KNN); then generalized linear models (GLMs), in particular Poisson regression.

## 4.1 An Overview of Classification (pp. 135-136)

- Setting: training observations (x1, y1), ..., (xn, yn); "We want our classifier to perform well not only on the training data, but also on test observations that were not used to train the classifier."
- Running example: simulated `Default` data, 10,000 individuals, predict `default` (Yes/No) from `balance` (monthly credit card balance) and `income`. "The overall default rate is about 3 %".
- Warning (p. 136): Figure 4.1 shows "a very pronounced relationship between the predictor balance and the response default. In most real applications, the relationship between the predictor and the response will not be nearly so strong."

## 4.2 Why Not Linear Regression? (pp. 136-138)

- Coding a K > 2 class response as 1, 2, 3 "implies an ordering on the outcomes" and equal gaps; a different coding gives "fundamentally different linear models". Only acceptable if the levels have a natural ordering and the gaps are felt to be similar (mild / moderate / severe).
- Binary response, 0/1 dummy coding: least squares "is not completely unreasonable: it can be shown that the X beta-hat obtained using linear regression is in fact an estimate of Pr(drug overdose | X)". But "some of our estimates might be outside the [0, 1] interval". Flipping the coding gives the same final predictions.
- "Curiously, it turns out that the classifications that we get if we use linear regression to predict a binary response will be the same as for the linear discriminant analysis (LDA) procedure".
- Summary (p. 138): two reasons not to use regression for classification: "(a) a regression method cannot accommodate a qualitative response with more than two classes; (b) a regression method will not provide meaningful estimates of Pr(Y|X), even with just two classes."

## 4.3 Logistic Regression (pp. 138-146)

Logistic regression "models the probability that Y belongs to a particular category", e.g. p(balance) = Pr(default = Yes | balance). Classification then needs a threshold: predict Yes when p(balance) > 0.5, or "if a company wishes to be conservative in predicting individuals who are at risk for default, then they may choose to use a lower threshold, such as p(balance) > 0.1" (p. 139). The threshold is a separate decision from the probability model.

### 4.3.1 The Logistic Model (pp. 139-140)

- (4.1) linear probability model: `p(X) = b0 + b1 X`. Problem: negative probabilities for balances near zero, > 1 for very large balances. "Any time a straight line is fit to a binary response that is coded as 0 or 1, in principle we can always predict p(X) < 0 for some values of X and p(X) > 1 for others (unless the range of X is limited)."
- (4.2) logistic function: `p(X) = exp(b0 + b1 X) / (1 + exp(b0 + b1 X))`. Always an S-shaped curve.
- Numeric fact (p. 140): "The average fitted probability in both cases is 0.0333 (averaged over the training data), which is the same as the overall proportion of defaulters in the data set." (Both = linear and logistic fits.)
- (4.3) odds: `p(X) / (1 - p(X)) = exp(b0 + b1 X)`, range 0 to infinity. Examples: p = 0.2 gives odds 1/4 ("1 in 5 people with an odds of 1/4 will default"); p = 0.9 gives odds 9.
- (4.4) log odds / logit: `log( p(X) / (1 - p(X)) ) = b0 + b1 X`. The model "has a logit that is linear in X".
- Interpretation: a one-unit increase in X changes the log odds by b1, i.e. multiplies the odds by exp(b1). "b1 does not correspond to the change in p(X) associated with a one-unit increase in X. The amount that p(X) changes due to a one-unit change in X depends on the current value of X." Only the sign carries over directly.

### 4.3.2 Estimating the Regression Coefficients (pp. 140-141)

- Fitted by maximum likelihood, "preferred, since it has better statistical properties" than non-linear least squares.
- Intuition: choose b0-hat, b1-hat so that p-hat(xi) is "close to one for all individuals who defaulted, and a number close to zero for all individuals who did not".
- (4.5) likelihood function: `L(b0, b1) = prod_{i: yi=1} p(xi) * prod_{i': yi'=0} (1 - p(xi'))`. Estimates maximize it. (The script-ell symbol is printed as `%` in the extraction.)
- "In the linear regression setting, the least squares approach is in fact a special case of maximum likelihood." Mathematical details of the fitting "are beyond the scope of this book".
- Table 4.1 (p. 142), default on balance:

| | Coefficient | Std. error | z-statistic | p-value |
|---|---|---|---|---|
| Intercept | −10.6513 | 0.3612 | −29.5 | <0.0001 |
| balance | 0.0055 | 0.0002 | 24.9 | <0.0001 |

- z-statistic = b1-hat / SE(b1-hat); plays the role of the t-statistic. Null hypothesis H0: b1 = 0 means `p(X) = exp(b0)/(1 + exp(b0))`, i.e. probability does not depend on X.
- Intercept (p. 141): "typically not of interest; its main purpose is to adjust the average fitted probabilities to the proportion of ones in the data (in this case, the overall default rate)". This matters for Qaf-ft: the intercept encodes the training base rate.

### 4.3.3 Making Predictions (pp. 141-142)

- Balance $1,000: `p-hat = exp(−10.6513 + 0.0055×1,000) / (1 + exp(...)) = 0.00576` ("below 1 %"). Balance $2,000: 0.586 (58.6 %).
- Qualitative predictors via dummy variables. Table 4.2 (p. 142), default on student:

| | Coefficient | Std. error | z-statistic | p-value |
|---|---|---|---|---|
| Intercept | −3.5041 | 0.0707 | −49.55 | <0.0001 |
| student[Yes] | 0.4049 | 0.1150 | 3.52 | 0.0004 |

- Pr(default = Yes | student = Yes) = 0.0431; Pr(default = Yes | student = No) = 0.0292.

### 4.3.4 Multiple Logistic Regression and confounding (pp. 142-144)

- (4.6) `log( p(X)/(1 − p(X)) ) = b0 + b1 X1 + ... + bp Xp`.
- (4.7) `p(X) = exp(b0 + b1 X1 + ... + bp Xp) / (1 + exp(b0 + b1 X1 + ... + bp Xp))`. Fitted by maximum likelihood.
- Table 4.3 (p. 143), income in thousands of dollars:

| | Coefficient | Std. error | z-statistic | p-value |
|---|---|---|---|---|
| Intercept | −10.8690 | 0.4923 | −22.08 | <0.0001 |
| balance | 0.0057 | 0.0002 | 24.74 | <0.0001 |
| income | 0.0030 | 0.0082 | 0.37 | 0.7115 |
| student[Yes] | −0.6468 | 0.2362 | −2.74 | 0.0062 |

- The paradox: the student coefficient is positive alone (Table 4.2: +0.4049) and negative with balance and income in the model (Table 4.3: −0.6468).
- Explanation (Figure 4.3, p. 144): `student` and `balance` are correlated. "Students tend to hold higher levels of debt, which is in turn associated with higher probability of default." For a fixed balance and income a student is less likely to default than a non-student, but the overall student default rate is higher because students carry higher balances.
- Stated consequence: "A student is riskier than a non-student if no information about the student's credit card balance is available. However, that student is less risky than a non-student with the same credit card balance!"
- Definition and warning (p. 144): "the results obtained using one predictor may be quite different from those obtained using multiple predictors, especially when there is correlation among the predictors. In general, the phenomenon seen in Figure 4.3 is known as confounding."
- Predictions: student, balance $1,500, income $40,000: (4.8) p-hat = 0.058. Non-student, same balance and income: (4.9) p-hat = 0.105. (Income enters as 40, not 40,000, because the model was fit in units of $1,000.)

### 4.3.5 Multinomial Logistic Regression (pp. 144-146)

- K > 2 classes. Baseline coding with class K as baseline:
  - (4.10) `Pr(Y = k | X = x) = exp(bk0 + bk1 x1 + ... + bkp xp) / (1 + sum_{l=1}^{K−1} exp(bl0 + bl1 x1 + ... + blp xp))` for k = 1..K−1.
  - (4.11) `Pr(Y = K | X = x) = 1 / (1 + sum_{l=1}^{K−1} exp(bl0 + ... + blp xp))`.
  - (4.12) `log( Pr(Y = k | X = x) / Pr(Y = K | X = x) ) = bk0 + bk1 x1 + ... + bkp xp`: "the log odds between any pair of classes is linear in the features".
- Choice of baseline changes the coefficients but not "the fitted values (predictions), the log odds between any pair of classes, and the other key model outputs". Interpretation of coefficients "must be done with care, since it is tied to the choice of baseline".
- Softmax coding (used "extensively in some areas of the machine learning literature", reappears in Chapter 10): treat all K classes symmetrically,
  - (4.13) `Pr(Y = k | X = x) = exp(bk0 + bk1 x1 + ... + bkp xp) / sum_{l=1}^{K} exp(bl0 + bl1 x1 + ... + blp xp)`.
  - (4.14) `log( Pr(Y = k | X = x) / Pr(Y = k' | X = x) ) = (bk0 − bk'0) + (bk1 − bk'1) x1 + ... + (bkp − bk'p) xp`.
  - Coefficients are estimated for all K classes rather than K − 1; only differences matter for the log odds.

## 4.4 Generative Models for Classification (pp. 146-161)

Idea: model the distribution of X separately in each class, then flip with Bayes' theorem to get Pr(Y = k | X = x).

Why another method besides logistic regression (p. 146, quoted):

- "When there is substantial separation between the two classes, the parameter estimates for the logistic regression model are surprisingly unstable. The methods that we consider in this section do not suffer from this problem."
- "If the distribution of the predictors X is approximately normal in each of the classes and the sample size is small, then the approaches in this section may be more accurate than logistic regression."
- They extend naturally to more than two classes.

Definitions: pi_k = prior probability of class k; f_k(X) = Pr(X | Y = k) density of X in class k; p_k(x) = Pr(Y = k | X = x) posterior.

- (4.15) Bayes' theorem: `Pr(Y = k | X = x) = pi_k f_k(x) / sum_{l=1}^{K} pi_l f_l(x)`.
- Estimating pi_k is easy "if we have a random sample from the population": the fraction of training observations in class k. Estimating f_k(x) is hard and needs simplifying assumptions.
- The Bayes classifier (assign to the class with largest p_k(x)) "has the lowest possible error rate out of all classifiers. (Of course, this is only true if all of the terms in (4.15) are correctly specified.)"

### 4.4.1 LDA for p = 1 (pp. 147-149)

- Assume f_k normal: (4.16) `f_k(x) = (1 / (sqrt(2 pi) sigma_k)) exp( −(x − mu_k)^2 / (2 sigma_k^2) )`, with a shared variance sigma^2 across classes.
- (4.17) posterior with these densities plugged into (4.15).
- (4.18) discriminant: assign to the class maximizing `delta_k(x) = x * mu_k / sigma^2 − mu_k^2 / (2 sigma^2) + log(pi_k)`.
- (4.19) K = 2, pi_1 = pi_2: Bayes decision boundary at `x = (mu_1 + mu_2) / 2`.
- Estimates: (4.20) `mu_k-hat = (1/n_k) sum_{i: yi=k} xi`; `sigma^2-hat = (1/(n − K)) sum_k sum_{i: yi=k} (xi − mu_k-hat)^2`. (4.21) `pi_k-hat = n_k / n`. (4.22) plug-in discriminant delta_k-hat(x). "Linear" because delta_k-hat is linear in x.
- Priors can be supplied (p. 149): "Sometimes we have knowledge of the class membership probabilities pi_1, ..., pi_K, which can be used directly."
- Example (Figure 4.4): mu_1 = −1.25, mu_2 = 1.25, sigma^2 = 1, 20 observations per class. Bayes error rate 10.6 %, LDA test error 11.1 %: "only 0.5 % above the smallest possible error rate".

### 4.4.2 LDA for p > 1; the Default threshold discussion; ROC (pp. 150-156)

- Assumption: X ~ N(mu_k, Sigma) in class k, class-specific mean vector, common covariance matrix.
- (4.23) multivariate Gaussian density: `f(x) = (2 pi)^(−p/2) |Sigma|^(−1/2) exp( −(1/2) (x − mu)^T Sigma^(−1) (x − mu) )`.
- (4.24) `delta_k(x) = x^T Sigma^(−1) mu_k − (1/2) mu_k^T Sigma^(−1) mu_k + log pi_k`. (4.25) boundary between classes k and l where delta_k = delta_l.
- Figure 4.6 (three classes, 20 observations each): test error Bayes 0.0746, LDA 0.0770.

LDA on Default (predictors balance and student), 10,000 training observations. Training error rate 2.75 %. Two caveats (p. 152):

1. "training error rates will usually be lower than test error rates, which are the real quantity of interest ... The higher the ratio of parameters p to number of samples n, the more we expect this overfitting to play a role." (Not a concern here: p = 2, n = 10,000.)
2. "since only 3.33 % of the individuals in the training sample defaulted, a simple but useless classifier that always predicts that an individual will not default ... will result in an error rate of 3.33 %. In other words, the trivial null classifier will achieve an error rate that is only a bit higher than the LDA training set error rate."

Footnote 4 (p. 152): student status is qualitative, so LDA's normality assumption "is clearly violated in this example! However, LDA is often remarkably robust to model violations".

**Table 4.4** (p. 152). LDA, threshold 0.5, 10,000 training observations:

| | True default status: No | True default status: Yes | Total |
|---|---|---|---|
| Predicted default status: No | 9644 | 252 | 9896 |
| Predicted default status: Yes | 23 | 81 | 104 |
| Total | 9667 | 333 | 10000 |

Caption: "LDA made incorrect predictions for 23 individuals who did not default and for 252 individuals who did default."

Reading of Table 4.4 (p. 153):

- LDA predicted 104 defaults; 81 actually defaulted, 23 did not. "only 23 out of 9,667 of the individuals who did not default were incorrectly labeled".
- "of the 333 individuals who defaulted, 252 (or 75.7 %) were missed by LDA. So while the overall error rate is low, the error rate among individuals who defaulted is very high."
- Sensitivity = "the percentage of true defaulters that are identified; it equals 24.3 %". Specificity = "the percentage of non-defaulters that are correctly identified; it equals (1 − 23/9667) = 99.8 %".
- (my arithmetic) False positive rate 23/9667 = 0.24 %; precision 81/104 = 77.9 %.

Why sensitivity is so low (p. 153): LDA approximates the Bayes classifier, "which has the lowest total error rate out of all classifiers ... regardless of the class from which the errors stem". The user's costs may be asymmetric. The fix is to move the threshold.

- (4.26) Bayes/LDA default rule: assign to default if `Pr(default = Yes | X = x) > 0.5`.
- (4.27) modified rule: assign to default if `Pr(default = Yes | X = x) > 0.2`.

**Table 4.5** (p. 154). LDA, threshold 0.2:

| | True default status: No | True default status: Yes | Total |
|---|---|---|---|
| Predicted default status: No | 9432 | 138 | 9570 |
| Predicted default status: Yes | 235 | 195 | 430 |
| Total | 9667 | 333 | 10000 |

Reading of Table 4.5 (pp. 153-154):

- LDA now predicts 430 defaults. "Of the 333 individuals who default, LDA correctly predicts all but 138, or 41.4 %" (down from 75.7 %).
- Cost: "now 235 individuals who do not default are incorrectly classified. As a result, the overall error rate has increased slightly to 3.73 %."
- (my arithmetic) Sensitivity 195/333 = 58.6 %; false positive rate 235/9667 = 2.43 %; precision 195/430 = 45.3 %. Moving the threshold from 0.5 to 0.2 multiplied the false positive count by about 10 (23 to 235) while true positives went from 81 to 195.

Figure 4.7 (p. 154; values not readable, caption and prose only): error rates versus threshold. "Using a threshold of 0.5 ... minimizes the overall error rate ... But when a threshold of 0.5 is used, the error rate among the individuals who default is quite high ... As the threshold is reduced, the error rate among individuals who default decreases steadily, but the error rate among the individuals who do not default increases."

Key sentence on choosing the threshold (p. 154): "How can we decide which threshold value is best? Such a decision must be based on domain knowledge, such as detailed information about the costs associated with default."

ROC curve and AUC (pp. 154-155):

- ROC ("receiver operating characteristics") "is a popular graphic for simultaneously displaying the two types of errors for all possible thresholds".
- Figure 4.8 axes: true positive rate (sensitivity) versus false positive rate (1 − specificity), traced as the threshold varies. "The actual thresholds are not shown." "The ideal ROC curve hugs the top left corner". The dotted diagonal is the "no information" classifier.
- AUC = area under the ROC curve = "overall performance of a classifier, summarized over all possible thresholds". For LDA on Default the AUC is 0.95, "close to the maximum of 1.0". "We expect a classifier that performs no better than chance to have an AUC of 0.5 (when evaluated on an independent test set not used in model training)."
- Note that Figure 4.8 is drawn "on the training data".
- "ROC curves are useful for comparing different classifiers, since they take into account all possible thresholds." The ROC curve of the logistic regression of §4.3.4 on these data "is virtually indistinguishable from this one for the LDA model".

Error-type terminology. "+" = the "disease" / non-null / alternative hypothesis; "−" = non-disease / null hypothesis. For Default, "+" = defaults.

**Table 4.6** (p. 156). "Possible results when applying a classifier or diagnostic test to a population."

| | True class: − or Null | True class: + or Non-null | Total |
|---|---|---|---|
| Predicted class: − or Null | True Neg. (TN) | False Neg. (FN) | N∗ |
| Predicted class: + or Non-null | False Pos. (FP) | True Pos. (TP) | P∗ |
| Total | N | P | |

**Table 4.7** (p. 156). "Important measures for classification and diagnostic testing, derived from quantities in Table 4.6."

| Name | Definition | Synonyms |
|---|---|---|
| False Pos. rate | FP/N | Type I error, 1−Specificity |
| True Pos. rate | TP/P | 1−Type II error, power, sensitivity, recall |
| Pos. Pred. value | TP/P∗ | Precision, 1−false discovery proportion |
| Neg. Pred. value | TN/N∗ | (no synonym listed) |

Stated distinction (p. 156): "The denominators for the false positive and true positive rates are the actual population counts in each class. In contrast, the denominators for the positive predictive value and the negative predictive value are the total predicted counts for each class."

### 4.4.3 Quadratic Discriminant Analysis (pp. 156-157)

- Assumption: X ~ N(mu_k, Sigma_k), a covariance matrix per class.
- (4.28) `delta_k(x) = −(1/2)(x − mu_k)^T Sigma_k^(−1) (x − mu_k) − (1/2) log|Sigma_k| + log pi_k`, which expands to `−(1/2) x^T Sigma_k^(−1) x + x^T Sigma_k^(−1) mu_k − (1/2) mu_k^T Sigma_k^(−1) mu_k − (1/2) log|Sigma_k| + log pi_k`: quadratic in x.
- LDA versus QDA is a bias-variance trade-off. A covariance matrix has p(p+1)/2 parameters; QDA estimates K p(p+1)/2. "With 50 predictors this is some multiple of 1,275". LDA has Kp linear coefficients, so it "is a much less flexible classifier than QDA, and so has substantially lower variance", but "if LDA's assumption that the K classes share a common covariance matrix is badly off, then LDA can suffer from high bias".
- Rule of thumb (p. 157): "LDA tends to be a better bet than QDA if there are relatively few training observations and so reducing variance is crucial. In contrast, QDA is recommended if the training set is very large ... or if the assumption of a common covariance matrix for the K classes is clearly untenable."
- Figure 4.9: equal class correlations (0.7) give a linear Bayes boundary and LDA wins; correlations 0.7 versus −0.7 give a quadratic Bayes boundary and QDA wins.

### 4.4.4 Naive Bayes (pp. 158-161)

- Single assumption: "Within the kth class, the p predictors are independent."
- (4.29) `f_k(x) = f_k1(x1) × f_k2(x2) × ... × f_kp(xp)`.
- (4.30) `Pr(Y = k | X = x) = pi_k × f_k1(x1) × ... × f_kp(xp) / sum_{l=1}^{K} pi_l × f_l1(x1) × ... × f_lp(xp)`.
- Why it helps: no need to estimate the joint distribution (the association between predictors). "Do we really believe the naive Bayes assumption ...? In most settings, we do not." But it "often leads to pretty decent results, especially in settings where n is not large enough relative to p for us to effectively estimate the joint distribution of the predictors within each class". "the naive Bayes assumption introduces some bias, but reduces variance".
- Options for the one-dimensional f_kj: (i) Gaussian N(mu_jk, sigma_jk^2), which "amounts to QDA with an additional assumption that the class-specific covariance matrix is diagonal"; (ii) non-parametric: histogram bin fractions, or a kernel density estimator; (iii) for qualitative Xj, class-conditional proportions (example: 100 observations, counts 32, 55, 13 give 0.32, 0.55, 0.13).
- Toy example (Figure 4.10): p = 3, K = 2, equal priors, x∗ = (0.4, 1.5, 1). f11-hat(0.4) = 0.368, f12-hat(1.5) = 0.484, f13-hat(1) = 0.226; f21-hat(0.4) = 0.030, f22-hat(1.5) = 0.130, f23-hat(1) = 0.616. Posteriors 0.944 and 0.056.
- Table 4.8 (p. 160), naive Bayes on Default, threshold 0.5: predicted No = 9621 (true No), 244 (true Yes), total 9865; predicted Yes = 46 (true No), 89 (true Yes), total 135; column totals 9667, 333, 10000.
- Table 4.9 (p. 161), naive Bayes, threshold 0.2: predicted No = 9339, 130, total 9469; predicted Yes = 328, 203, total 531; column totals 9667, 333, 10000.
- ISLP's reading: at 0.5 "LDA has a slightly lower overall error rate, naive Bayes correctly predicts a higher fraction of the true defaulters". At 0.2 "Naive Bayes has a higher error rate, but correctly predicts almost two-thirds of the true defaults."
- (my arithmetic) Table 4.8: error 290/10000 = 2.90 %, sensitivity 89/333 = 26.7 %, false positive rate 46/9667 = 0.48 %. Table 4.9: error 458/10000 = 4.58 %, sensitivity 203/333 = 61.0 %, false positive rate 328/9667 = 3.39 %. At the same nominal threshold naive Bayes produces more false positives than LDA (46 versus 23; 328 versus 235).
- When naive Bayes pays off (p. 161): with n = 10,000 and p = 2 "the reduction in variance resulting from the naive Bayes assumption is not necessarily worthwhile. We expect to see a greater pay-off to using naive Bayes relative to LDA or QDA in instances where p is larger or n is smaller".

## 4.5 A Comparison of Classification Methods (pp. 161-167)

### 4.5.1 An Analytical Comparison (pp. 161-164)

All methods are compared through the log odds against baseline class K, (4.31) `log( Pr(Y = k | X = x) / Pr(Y = K | X = x) )`.

- LDA, (4.32): `= a_k + sum_{j=1}^{p} b_kj x_j`, with `a_k = log(pi_k/pi_K) − (1/2)(mu_k + mu_K)^T Sigma^(−1) (mu_k − mu_K)` and b_kj the jth component of `Sigma^(−1)(mu_k − mu_K)`. Linear in x, like logistic regression.
- QDA, (4.33): `= a_k + sum_j b_kj x_j + sum_j sum_l c_kjl x_j x_l`. Quadratic in x.
- Naive Bayes, (4.34): `= a_k + sum_j g_kj(x_j)`, with `a_k = log(pi_k/pi_K)` and `g_kj(x_j) = log( f_kj(x_j) / f_Kj(x_j) )` (the fraction is garbled in the extraction; this form follows from the derivation lines directly above it). A generalized additive model.

Observations (p. 163):

- LDA is a special case of QDA (c_kjl = 0).
- "Any classifier with a linear decision boundary is a special case of naive Bayes with g_kj(x_j) = b_kj x_j. In particular, this means that LDA is a special case of naive Bayes!"
- Naive Bayes with Gaussian f_kj and shared per-feature variance sigma_j^2 gives `b_kj = (mu_kj − mu_Kj)/sigma_j^2`: "a special case of LDA with Sigma restricted to be a diagonal matrix".
- "Neither QDA nor naive Bayes is a special case of the other." Naive Bayes is flexible per feature but purely additive; QDA has multiplicative terms c_kjl x_j x_l, so it "has the potential to be more accurate in settings where interactions among the predictors are important in discriminating between classes".
- "None of these methods uniformly dominates the others".
- Logistic regression has the same linear form as LDA (4.32). The difference is the fitting: LDA's coefficients come from Gaussian-assumption estimates of pi_k, mu_k, Sigma; logistic regression's "are chosen to maximize the likelihood function (4.5). Thus, we expect LDA to outperform logistic regression when the normality assumption (approximately) holds, and we expect logistic regression to perform better when it does not."

KNN (p. 164): completely non-parametric.

- Expected "to dominate LDA and logistic regression when the decision boundary is highly non-linear, provided that n is very large and p is small".
- "KNN requires a lot of observations relative to the number of predictors—that is, n much larger than p", because it "tends to reduce the bias while incurring a lot of variance".
- With a non-linear boundary but modest n or not-small p, "QDA may be preferred to KNN".
- "Unlike logistic regression, KNN does not tell us which predictors are important: we don't get a table of coefficients".

### 4.5.2 An Empirical Comparison (pp. 164-167)

Six simulated two-class scenarios, p = 2 quantitative predictors, 100 random training sets each, test error on a large test set (Figures 4.11 and 4.12 are boxplots; values not readable). KNN with K = 1 and with K chosen by cross-validation; naive Bayes with univariate Gaussian densities.

| Scenario | Set-up | Result as stated |
|---|---|---|
| 1 (linear) | 20 training observations per class; uncorrelated normal predictors, different class means | LDA "performed well"; logistic "also performed quite well"; KNN "performed poorly because it paid a price in terms of variance"; QDA worse than LDA; naive Bayes "slightly better than QDA, because the naive Bayes assumption of independent predictors is correct" |
| 2 (linear) | As 1, but within-class correlation −0.5 | Most methods similar to scenario 1; "The notable exception is naive Bayes, which performs very poorly here, since the naive Bayes assumption of independent predictors is violated" |
| 3 (linear) | Negative correlation; X1, X2 from the t-distribution; 50 observations per class | "logistic regression outperformed LDA, though both methods were superior to the other approaches"; "the QDA results deteriorated considerably as a consequence of non-normality"; naive Bayes "very poorly" |
| 4 (non-linear) | Normal; correlation 0.5 in class 1, −0.5 in class 2 | "QDA outperformed all of the other approaches"; naive Bayes poor |
| 5 (non-linear) | Uncorrelated normal predictors; responses sampled from the logistic function of a complicated non-linear function | QDA and naive Bayes "slightly better results than the linear methods"; KNN-CV best; "KNN with K = 1 gave the worst results out of all methods" |
| 6 (non-linear) | Normal with a different diagonal covariance matrix per class; "just n = 6 in each class" | "Naive Bayes performed very well, because its assumptions are met"; LDA and logistic poor (non-linear boundary); QDA "a bit worse than naive Bayes" because of variance at tiny n; KNN suffered |

Conclusions (p. 166): "no one method will dominate the others in every situation. When the true decision boundaries are linear, then the LDA and logistic regression approaches will tend to perform well. When the boundaries are moderately non-linear, QDA or naive Bayes may give better results. Finally, for much more complicated decision boundaries, a non-parametric approach such as KNN can be superior. But the level of smoothness for a non-parametric approach must be chosen carefully."

Flexibility by transformation (pp. 166-167): add X^2, X^3, X^4 to logistic regression; this "may or may not improve logistic regression's performance, depending on whether the increase in variance due to the added flexibility is offset by a sufficiently large reduction in bias". Adding all quadratic terms and cross-products to LDA gives the same model form as QDA with different parameter estimates.

## 4.6 Generalized Linear Models (pp. 167-173)

Motivation: a response that is "neither qualitative nor quantitative", here counts: `bikers`, hourly users in the Bikeshare data. Covariates mnth, hr, workingday, temp, weathersit.

### 4.6.1 Linear regression on Bikeshare (pp. 167-169)

- Table 4.10 (least squares): Intercept 73.60 (SE 5.13, t 14.34, p 0.00); workingday 1.27 (1.78, 0.71, 0.48); temp 157.21 (10.26, 15.32, 0.00); weathersit[cloudy/misty] −12.89 (1.96, −6.56, 0.00); weathersit[light rain/snow] −66.49 (2.97, −22.43, 0.00); weathersit[heavy rain/snow] −109.75 (76.67, −1.43, 0.15).
- Problems: (1) "9.6% of the fitted values in the Bikeshare data set are negative". (2) Heteroscedasticity: 1 AM-4 AM in December-February when raining, 5.05 users on average with standard deviation 3.73; 7 AM-10 AM in April-June with clear skies, 243.59 users with standard deviation 131.7. This "is a major violation of the assumptions of a linear model". (3) The response is integer-valued while the linear model's error term is continuous.
- Log-transforming the response avoids negative predictions and much of the heteroscedasticity, but inference is then about log(Y), and it "cannot be applied in settings where the response can take on a value of 0".

### 4.6.2 Poisson regression (pp. 169-172)

- (4.35) `Pr(Y = k) = exp(−lambda) lambda^k / k!`, k = 0, 1, 2, ...; lambda = E(Y) = Var(Y). Example lambda = 5: Pr(Y=0) = 0.0067, Pr(Y=1) = 0.034, Pr(Y=2) = 0.084.
- (4.36) `log(lambda(X1,...,Xp)) = b0 + b1 X1 + ... + bp Xp`; (4.37) `lambda(X1,...,Xp) = exp(b0 + b1 X1 + ... + bp Xp)`. The log guarantees lambda is nonnegative.
- (4.38) likelihood: `L(b0,...,bp) = prod_{i=1}^{n} exp(−lambda(xi)) lambda(xi)^yi / yi!`. Maximum likelihood, as for logistic regression.
- Table 4.11 (Poisson): Intercept 4.12 (SE 0.01, z 683.96, p 0.00); workingday 0.01 (0.00, 7.5, 0.00); temp 0.79 (0.01, 68.43, 0.00); weathersit[cloudy/misty] −0.08 (0.00, −34.53, 0.00); weathersit[light rain/snow] −0.58 (0.00, −141.91, 0.00); weathersit[heavy rain/snow] −0.93 (0.17, −5.55, 0.00).
- Interpretation: a one-unit increase in Xj multiplies E(Y) by exp(bj). Clear to cloudy: exp(−0.08) = 0.923 (92.3 % as many riders). Cloudy to rain: exp(−0.5) = 0.607.
- Mean-variance: Poisson assumes mean = variance; the linear model assumes constant variance.
- No negative fitted values (versus "almost 10%" for the linear model).
- workingday is significant under Poisson but not under linear regression.
- Overdispersion warning (footnote 5, p. 172): "the variance in the Bikeshare data appears to be much higher than the mean, a situation referred to as overdispersion. This causes the Z-values to be inflated in Table 4.11. A more careful analysis should account for this overdispersion".

### 4.6.3 GLMs in greater generality (pp. 172-173)

Common recipe:

1. Conditional on X, Y belongs to a family of distributions: Gaussian (linear), Bernoulli (logistic), Poisson.
2. The mean of Y is modeled as a function of the predictors: (4.39) `E(Y|X) = b0 + b1 X1 + ... + bp Xp`; (4.40) `E(Y|X) = Pr(Y = 1|X) = exp(b0 + ... + bp Xp)/(1 + exp(b0 + ... + bp Xp))`; (4.41) `E(Y|X) = lambda(X) = exp(b0 + ... + bp Xp)`.
3. (4.42) link function eta: `eta(E(Y|X1,...,Xp)) = b0 + b1 X1 + ... + bp Xp`. Links: eta(mu) = mu (linear), eta(mu) = log(mu/(1 − mu)) (logistic), eta(mu) = log(mu) (Poisson).

Gaussian, Bernoulli, Poisson are members of the exponential family, with the exponential, Gamma and negative binomial distributions. Any regression that models Y from an exponential-family member and transforms the mean to be linear in the predictors is a GLM. Gamma regression and negative binomial regression are named but not covered.

## 4.7 Lab: Logistic Regression, LDA, QDA, and KNN (pp. 173-193)

Smarket data: percentage returns of the S&P 500 over 1,250 days, 2001-2005; Lag1-Lag5, Volume, Today, Direction.

- Logistic regression (`sm.GLM(y, X, family=sm.families.Binomial())`) on Lag1-Lag5 and Volume: smallest p-value is Lag1 at 0.145 (coefficient −0.0731); "no clear evidence of a real association".
- Training confusion table (rows predicted, columns truth): predicted Down: 145 (truth Down), 141 (truth Up); predicted Up: 457, 507. 652 correct of 1,250 = 0.5216. Warning (p. 176): "this result is misleading because we trained and tested the model on the same set of 1,250 observations ... the training error rate is often overly optimistic — it tends to underestimate the test error rate."
- Held-out 2005 (252 observations), model trained on 2001-2004: predicted Down: 77, 97; predicted Up: 34, 44. Test accuracy 0.4802, error 0.5198: "worse than random guessing!".
- Refit with Lag1 and Lag2 only, with the stated reason (p. 178): "using predictors that have no relationship with the response tends to cause a deterioration in the test error rate (since such predictors cause an increase in variance without a corresponding decrease in bias)". Result: predicted Down: 35, 35; predicted Up: 76, 106. Accuracy 0.5595; accuracy on days predicted Up 106/(106+76) = 0.5824. But "a much simpler strategy of predicting that the market will increase every day will also be correct 56% of the time!" and "one would need to investigate more carefully whether this small improvement was real or just due to random chance."
- LDA (sklearn `LinearDiscriminantAnalysis`): priors 0.492 (Down), 0.508 (Up); scalings −0.642, −0.513; test confusion table identical to the logistic one. Non-default threshold (pp. 180-181): requiring posterior probability of Down above 0.9 gives `np.sum(lda_prob[:,0] > 0.9)` = 0: "No days in 2005 meet that threshold! In fact, the greatest posterior probability of decrease in all of 2005 was 52.02%."
- QDA: predicted Down: 30, 20; predicted Up: 81, 121; accuracy 0.599. Caution: "we recommend evaluating this method's performance on a larger test set before betting that this approach will consistently beat the market!"
- Naive Bayes (`GaussianNB`): predicted Down: 29, 20; predicted Up: 82, 121; "accurate predictions over 59% of the time". Variances use the 1/n formula (footnote 6).
- KNN: K = 1 gives 43, 58 / 68, 83 and accuracy 0.5; K = 3 gives 0.532.
- sklearn design note (p. 181): instantiate, `fit()`, `predict()`; this "makes it possible to cleanly copy the classifier so that it can be fit on different data; e.g. different training sets arising in cross-validation".

Caravan data (rare positive class): 85 predictors, 5,822 individuals, Purchase Yes = 348, No = 5474 (348/5822 = 0.0598).

- Standardize before KNN (p. 185): "Any variables that are on a large scale will have a much larger effect on the distance between the observations"; `StandardScaler` gives mean zero, standard deviation one.
- Test set of 1,000. KNN K = 1: error 0.111 versus 0.067 for always predicting No. "since just over 6% of customers purchased insurance, we could get the error rate down to almost 6% by always predicting No regardless of the values of the predictors! This is known as the null rate."
- What matters instead (p. 186): "the overall error rate is not of interest. Instead, the fraction of individuals that are correctly predicted to buy insurance is of interest." KNN-1 confusion: predicted No: 880, 58; predicted Yes: 53, 9. 9/(53+9) = 0.145, "double the rate that one would obtain from random guessing".
- Tuning parameter = hyperparameter (p. 187). K = 1..5 gives 62/9 (14.5 %), 6/1 (16.7 %), 20/3 (15.0 %), 3/0 (0.0 %), 7/1 (14.3 %) predicted/correct: "We see some variability — the numbers for K=4 are very different from the rest."
- Logistic regression in sklearn (pp. 187-188): "by default it fits something like the ridge regression version of logistic regression ... This can be modified by appropriately setting the argument C below. Its default value is 1 but by setting it to a very large number, the algorithm converges to the same solution as the usual (unregularized) logistic regression estimator". Code uses `LogisticRegression(C=1e10, solver='liblinear')`.
- With a 0.5 cut-off "none of the test observations are predicted to purchase insurance" (table: predicted No: 933, 67; predicted Yes: 0, 0). (The printed code line reads `logit_pred[:,1] > 5`; the prose says 0.5. I report it as printed.) With cut-off 0.25: predicted No: 913, 58; predicted Yes: 20, 9; 9/(20+9) = 0.310, "almost five times better than random guessing!". "we are not required to use a cut-off of 0.5."

Bikeshare (pp. 188-193): 8645 rows × 15 columns; `sm.OLS` and `sm.GLM(..., family=sm.families.Poisson())`. Dummy coding choice (baseline versus sum-to-zero) "really does not matter, provided that we interpret the model output correctly": fitted values agree (sum of squared differences 1.53e-20). Other families: `sm.families.Gamma()`.

## 4.8 Exercises (pp. 193-199, skimmed)

- Conceptual: (1) equivalence of (4.2) and (4.3); (2), (3) derive LDA and QDA discriminants for p = 1; (4) curse of dimensionality for KNN and other local methods (fraction of data in a 10 % neighbourhood for p = 1, 2, 100); (5) LDA versus QDA on training versus test error; (6) logistic prediction with b0-hat = −6, b1-hat = 0.05, b2-hat = 1; (7) Bayes' theorem with a normal density (means 10 and 0, variance 36, prior 80 %); (8) logistic 20 % train / 30 % test versus 1-NN averaging 18 % over train and test: the trap is that 1-NN training error is zero; (9) odds; (10), (11) forms of (4.32) and (4.33); (12) baseline versus softmax coefficients: only coefficient differences are identified.
- Applied: (13) Weekly data, all classifiers, held-out 2009-2010; (14) Auto mpg01; (15) writing functions; (16) Boston crime rate above/below median.

## Application to Qaf-ft (Chapter 4)

Mapping of terms. Use Table 4.6 with "+" = the word/letter was recited wrongly (non-null), "−" = recited correctly (null).

| Qaf-ft quantity | Table 4.7 name | Formula | Depends on mistake prevalence? |
|---|---|---|---|
| False-flag rate (worst failure) | False positive rate, Type I error, 1 − specificity | FP/N, N = truly correct words | No |
| Mistake detection rate | True positive rate, sensitivity, recall, power | TP/P, P = truly wrong words | No |
| "When we flag, are we right?" | Positive predictive value, precision | TP/P∗, P∗ = flagged words | Yes |
| "When we pass, is it right?" | Negative predictive value | TN/N∗ | Yes |

1. Accuracy alone is misleading when mistakes are rare (§4.4.2 p. 152; lab p. 186). On Default a classifier that never predicts default has error 3.33 %, against 2.75 % for LDA, while LDA misses 75.7 % of defaulters. On Caravan, KNN's 11.1 % error is worse than the 6.7 % null rate. For Qaf-ft: a verifier that never flags anything will have accuracy equal to one minus the mistake prevalence. Always report (a) the null rate, (b) the false-flag rate FP/N, (c) the detection rate TP/P at the chosen threshold, (d) the full confusion matrix in the layout of Table 4.4. Never report a single accuracy number without these.

2. The 0.5 threshold is the wrong default. The Bayes threshold minimizes total error "regardless of the class from which the errors stem" (p. 153). The book's bank lowers the threshold to 0.2 because missing defaulters is costly. Qaf-ft's cost asymmetry points the other way (a false flag is the worst outcome), so the threshold on Pr(mistake | features) should be moved up, not down. The Smarket lab's "predict a decrease only if posterior at least 90 %" (pp. 180-181) is the same pattern, and its outcome (zero days flagged) is the warning: after raising the threshold, check that the detection rate is not zero.

3. Choosing the mistake-flag threshold from the ROC curve at a target false-positive rate.
   - ISLP: the ROC curve shows TPR against FPR for all thresholds, but "The actual thresholds are not shown" (Figure 4.8), and the choice "must be based on domain knowledge" of costs (p. 154).
   - Procedure for Qaf-ft (Extension, consistent with §4.4.2): fix the target false-flag rate alpha (for example 1 % or 0.5 % of correct words). Collect verifier scores for truly correct words that the verifier did not train on (out-of-fold, speaker-disjoint; see Chapter 5). Set the threshold t to the (1 − alpha) quantile of those scores, so that FP/N is about alpha. Report the TPR at t as the operating point. Keep the table of (threshold, FPR, TPR) because the plotted ROC hides the thresholds.
   - FPR and TPR have true-class denominators (p. 156), so an operating point fixed by FPR does not move when the proportion of mistakes changes between synthetic-heavy development data and real learners. Precision does move. Illustration (my arithmetic): at FPR 1 % and TPR 60 %, a 5 % mistake prevalence gives per 10,000 words 300 TP and 95 FP, precision 75.9 %; a 2 % prevalence gives 120 TP and 98 FP, precision 55.0 %.
   - Per-ayah consequence (Extension, my arithmetic, assuming independent words): with a per-word false-flag rate of 1 %, a correctly recited 20-word ayah gets at least one false flag with probability 1 − 0.99^20 = 18.2 %; at 0.1 % per word it is 2.0 %. Decide whether the product requirement is per word or per ayah before fixing alpha.
   - Report AUC for comparing verifier variants ("ROC curves are useful for comparing different classifiers"), but Qaf-ft operates in the far-left corner of the ROC, so also report TPR at fixed FPR values (Extension). Two verifiers with equal AUC can differ a lot at FPR 1 %.
   - The book's Figure 4.8 and AUC 0.95 are on training data. Qaf-ft's ROC must come from held-out speakers; the book's own qualifier is "when evaluated on an independent test set not used in model training".

4. Which classifier for the verifier, and why.
   - Default choice: (regularized) logistic regression. It models Pr(mistake | features) directly, gives a probability that can be thresholded, has a linear logit (low variance when real mistake examples are few), and yields a coefficient table with z-statistics so each feature's contribution can be inspected (KNN "does not tell us which predictors are important"). The verifier's features (alignment log-likelihood ratio, posterior margin, entropy, duration) are skewed and heavy-tailed rather than Gaussian, which is the situation of Scenario 3 where "logistic regression outperformed LDA" and "the QDA results deteriorated considerably as a consequence of non-normality".
   - Keep LDA as the comparison model: it has the same linear log-odds form (4.32), it "is often remarkably robust to model violations" (footnote 4), and it does not suffer from the instability of logistic regression "When there is substantial separation between the two classes" (p. 146). Synthetic errors can be much easier than real ones and may be almost perfectly separable on the log-likelihood-ratio feature; that is the situation in which unpenalized logistic coefficients blow up. Remedies: a ridge penalty on the logistic regression (sklearn's default behaviour, lab p. 187; Chapter 6), or LDA.
   - Avoid naive Bayes for this feature set. Log-likelihood ratio, posterior margin and entropy are all functions of the same frame posteriors, so they are strongly correlated within class. Scenarios 2 and 3: naive Bayes "performs very poorly here, since the naive Bayes assumption of independent predictors is violated". Tables 4.8 and 4.9 also show naive Bayes giving more false positives than LDA at the same nominal threshold. It becomes attractive only "where p is larger or n is smaller".
   - QDA only if there is a lot of data and clear evidence that the two classes have different covariance structure (K p(p+1)/2 covariance parameters). An intermediate step is logistic regression with a few chosen squared or interaction terms (pp. 166-167), accepted only if cross-validation shows the variance cost is repaid.
   - KNN is not suitable: needs "n much larger than p", gives no coefficients, and is scale-dependent.
   - The CTC output layer is a softmax (4.13). The log odds between the expected symbol and a competitor is a difference of logits (4.14), so a posterior-margin or log-likelihood-ratio feature is already on the log-odds scale that logistic regression is linear in. (Extension: enter such features untransformed; put durations on a log scale.)

5. Base rate and the intercept. The logistic intercept's "main purpose is to adjust the average fitted probabilities to the proportion of ones in the data" (p. 141), and the average fitted probability equals the training proportion of positives (0.0333 in the book). If the verifier is trained on data where synthetic errors make up, say, 30 % of words and deployed where 3 % of words are wrong, its probabilities will be too high across the board. Either train on a realistic mix, or correct the prior: LDA lets known class probabilities be "used directly" (p. 149); for logistic regression the corresponding correction is an intercept shift by the change in prior log odds (Extension, not in ISLP). The fixed-FPR threshold in point 3 is unaffected, which is another reason to define the operating point by FPR and not by a probability value.

6. Confounding (§4.3.4). Verifier features are correlated with each other and with nuisance variables, so single-feature analyses can show the wrong sign, exactly as student status did. Likely confounders in Qaf-ft: word length in phonemes and recitation tempo (confound duration), speaker proficiency and recording quality (confound entropy and margin), tajweed context such as madd (confounds duration), and above all the data source. If synthetic errors come from different speakers, microphones or processing than real correct recitations, a feature can separate "synthetic versus real" instead of "wrong versus right". Actions: include the nuisance variables as features (as balance was included alongside student) rather than judging features one at a time; interpret each coefficient as "holding the other features fixed"; report verifier performance separately on real learner mistakes and on synthetic ones; never rank candidate features by univariate tests alone.

7. Coefficient standard errors. The z-statistics of Tables 4.1-4.3 assume independent observations. Words within a clip and clips within a speaker are not independent, so software-reported standard errors for the verifier will be too small (Extension; the same direction as the overdispersion remark on p. 172). Use the speaker-level bootstrap of Chapter 5 for coefficient uncertainty.

8. GLMs for other Qaf-ft outputs. Number of mistakes per ayah or per clip is a count: use Poisson regression with a log link (4.36), not least squares, which would give negative fitted counts and assumes constant variance. Expect overdispersion (mistakes cluster within weak reciters); footnote 5 warns that it inflates z-values, and the negative binomial family is named on p. 173 as another GLM. A per-rule tajweed pass/fail is a Bernoulli response (logistic regression). A graded tajweed score should not be coded 1, 2, 3 for least squares unless the gaps are believed equal (p. 137). Segment durations are positive and skewed; the Gamma family (lab p. 193) or a log transform is the GLM-consistent treatment.

9. Lab habits to copy: standardize features with a scaler fitted on training data; drop features with no relationship to the response because they add variance without reducing bias (p. 178); compare every result with the always-negative baseline; distrust small improvements on a small test set ("whether this small improvement was real or just due to random chance", p. 178).

---

<!-- NOTES-CONTINUE -->
