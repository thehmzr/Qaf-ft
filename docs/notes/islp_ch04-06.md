# ISLP study notes: Chapters 4, 5, 6 (for Qaf-ft)

Source: `refs/islp_text/islp.txt`, lines 8735-18740 (pdftotext extraction of *An Introduction to Statistical Learning with Applications in Python*, 2023). Page numbers are the book's own (running headers). Chapter 4 = pp. 135-199, Chapter 5 = pp. 201-228, Chapter 6 = pp. 229-288.

Conventions used in these notes

- "ISLP says" / a page citation = stated in the text that was read. Numbers are quoted exactly as printed.
- "(derived arithmetic)" = a number derived from numbers in the text; not printed in the book.
- "Extension (not in ISLP)" = an application to Qaf-ft or a standard technique the book does not cover in these chapters. Treat these as design suggestions, not as citations.
- Figures are garbled in the extraction (axes and point clouds only). Everything said about a figure below comes from its caption and the surrounding prose, never from the plotted values. Where an equation was garbled, that is stated.

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
- (derived arithmetic) False positive rate 23/9667 = 0.24 %; precision 81/104 = 77.9 %.

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
- (derived arithmetic) Sensitivity 195/333 = 58.6 %; false positive rate 235/9667 = 2.43 %; precision 195/430 = 45.3 %. Moving the threshold from 0.5 to 0.2 multiplied the false positive count by about 10 (23 to 235) while true positives went from 81 to 195.

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
- (derived arithmetic) Table 4.8: error 290/10000 = 2.90 %, sensitivity 89/333 = 26.7 %, false positive rate 46/9667 = 0.48 %. Table 4.9: error 458/10000 = 4.58 %, sensitivity 203/333 = 61.0 %, false positive rate 328/9667 = 3.39 %. At the same nominal threshold naive Bayes produces more false positives than LDA (46 versus 23; 328 versus 235).
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
- With a 0.5 cut-off "none of the test observations are predicted to purchase insurance" (table: predicted No: 933, 67; predicted Yes: 0, 0). (The printed code line reads `logit_pred[:,1] > 5`; the prose says 0.5. Reported here as printed.) With cut-off 0.25: predicted No: 913, 58; predicted Yes: 20, 9; 9/(20+9) = 0.310, "almost five times better than random guessing!". "we are not required to use a cut-off of 0.5."

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
   - FPR and TPR have true-class denominators (p. 156), so an operating point fixed by FPR does not move when the proportion of mistakes changes between synthetic-heavy development data and real learners. Precision does move. Illustration (derived arithmetic): at FPR 1 % and TPR 60 %, a 5 % mistake prevalence gives per 10,000 words 300 TP and 95 FP, precision 75.9 %; a 2 % prevalence gives 120 TP and 98 FP, precision 55.0 %.
   - Per-ayah consequence (Extension, derived arithmetic, assuming independent words): with a per-word false-flag rate of 1 %, a correctly recited 20-word ayah gets at least one false flag with probability 1 − 0.99^20 = 18.2 %; at 0.1 % per word it is 2.0 %. Decide whether the product requirement is per word or per ayah before fixing alpha.
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

# Chapter 5. Resampling Methods (pp. 201-228)

## 5.0 Chapter opening (p. 201)

- Resampling methods "involve repeatedly drawing samples from a training set and refitting a model of interest on each sample in order to obtain additional information about the fitted model".
- Two methods: cross-validation and the bootstrap.
- Definitions: "The process of evaluating a model's performance is known as model assessment, whereas the process of selecting the proper level of flexibility for a model is known as model selection."
- The bootstrap is used "most commonly to provide a measure of accuracy of a parameter estimate or of a given statistical learning method".

## 5.1 Cross-Validation (pp. 202-211)

- Test error = "the average error that results from using a statistical learning method to predict the response on a new observation—that is, a measurement that was not used in training the method".
- Warning (p. 202): "the training error rate often is quite different from the test error rate, and in particular the former can dramatically underestimate the latter."
- Two families of test-error estimates: mathematical adjustment of the training error (Chapter 6: Cp, AIC, BIC, adjusted R2), or holding out a subset of the training observations (this chapter).

### 5.1.1 The Validation Set Approach (pp. 202-204)

- Randomly divide the observations into a training set and a validation (hold-out) set; fit on the first, predict the second; the validation error (MSE for a quantitative response) estimates the test error.
- Auto data example: 392 observations split 196 / 196; mpg on polynomial functions of horsepower. One split (Figure 5.2, left): quadratic much better than linear, cubic slightly worse than quadratic. Ten different random splits (Figure 5.2, right): all ten agree that quadratic beats linear, but "each of the ten curves results in a different test MSE estimate for each of the ten regression models considered. And there is no consensus among the curves as to which model results in the smallest validation set MSE."
- Two drawbacks (pp. 203-204):
  1. "the validation estimate of the test error rate can be highly variable, depending on precisely which observations are included in the training set and which observations are included in the validation set."
  2. Only a subset of the observations is used to fit. "Since statistical methods tend to perform worse when trained on fewer observations, this suggests that the validation set error rate may tend to overestimate the test error rate for the model fit on the entire data set."

### 5.1.2 Leave-One-Out Cross-Validation (pp. 204-206)

- Hold out one observation (x1, y1), fit on the other n − 1, compute MSE1 = (y1 − y1-hat)^2. It is "approximately unbiased" for the test error but "a poor estimate because it is highly variable, since it is based upon a single observation". Repeat for every observation.
- (5.1) `CV(n) = (1/n) sum_{i=1}^{n} MSE_i`.
- Advantages over the validation set approach (p. 205): "far less bias" (training sets of n − 1), so it "tends not to overestimate the test error rate as much"; and no randomness: "performing LOOCV multiple times will always yield the same results".
- Cost: n model fits. Shortcut for least squares linear or polynomial regression only:
  - (5.2) `CV(n) = (1/n) sum_{i=1}^{n} ( (y_i − y_i-hat) / (1 − h_i) )^2`, where h_i is the leverage (eq. 3.37, p. 105), which "lies between 1/n and 1, and reflects the amount that an observation influences its own fit".
  - "The magic formula (5.2) does not hold in general, in which case the model has to be refit n times." LOOCV itself is general and can be used with logistic regression or LDA.

### 5.1.3 k-Fold Cross-Validation (pp. 206-208)

- Randomly divide the observations "into k groups, or folds, of approximately equal size"; each fold in turn is the validation set while the method is fit on the other k − 1 folds.
- (5.3) `CV(k) = (1/k) sum_{i=1}^{k} MSE_i`.
- LOOCV is the special case k = n. "In practice, one typically performs k-fold CV using k = 5 or k = 10."
- Variability (Figure 5.4, right): nine different 10-fold CV runs on Auto give "some variability in the CV estimates as a result of the variability in how the observations are divided into ten folds. But this variability is typically much lower than the variability in the test error estimates that results from the validation set approach".
- Simulated-data check (Figure 5.6): LOOCV and 10-fold CV estimates "are very similar". Compared with the true test MSE they are almost identical in one panel, overestimate at high flexibility in another, and "have the correct general shape, but they underestimate the true test MSE" in the third.
- Two different uses of CV (p. 208): sometimes "the actual estimate of the test MSE is of interest"; at other times "we are interested only in the location of the minimum point in the estimated test MSE curve" (choosing among methods or flexibility levels). "despite the fact that they sometimes underestimate the true test MSE, all of the CV curves come close to identifying the correct level of flexibility".

### 5.1.4 Bias-Variance Trade-Off for k-Fold Cross-Validation (pp. 208-209)

- Bias: the validation set approach overestimates test error (trained on about half the data). LOOCV is "approximately unbiased" (n − 1 observations). k-fold with k = 5 or 10 has "an intermediate level of bias, since each training set contains approximately (k − 1)n/k observations". "from the perspective of bias reduction, it is clear that LOOCV is to be preferred to k-fold CV."
- Variance: "LOOCV has higher variance than does k-fold CV with k < n." Reason: LOOCV averages "the outputs of n fitted models, each of which is trained on an almost identical set of observations; therefore, these outputs are highly (positively) correlated with each other". k-fold models overlap less. "Since the mean of many highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV tends to have higher variance".
- Recommendation (p. 209): "one performs k-fold cross-validation using k = 5 or k = 10, as these values have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance."

Summary table (assembled from §5.1.1-5.1.4):

| Approach | Training size per fit | Bias of test-error estimate | Variance | Cost | Random? |
|---|---|---|---|---|---|
| Validation set | about n/2 | overestimates (highest bias) | high: depends on the split | 1 fit | yes |
| LOOCV | n − 1 | approximately unbiased | higher than k-fold (correlated fits) | n fits, or 1 with (5.2) for least squares | no |
| k-fold, k = 5 or 10 | (k − 1)n/k | intermediate | lower than LOOCV | k fits | yes, mildly |

### 5.1.5 Cross-Validation on Classification Problems (pp. 209-211)

- Same procedure, but the error is the number of misclassified observations.
- (5.4) `CV(n) = (1/n) sum_{i=1}^{n} Err_i`, with `Err_i = I(y_i != y_i-hat)`. k-fold and validation-set error rates are defined analogously.
- Example (Figure 5.7; simulated two-dimensional data of Figure 2.13): logistic regression with polynomial terms of degree 1 to 4 has true test error rates 0.201, 0.197, 0.160, 0.162; the Bayes error rate is 0.133.
- (5.5) quadratic logistic regression: `log( p/(1 − p) ) = b0 + b1 X1 + b2 X1^2 + b3 X2 + b4 X2^2`.
- Figure 5.8 (left): training error "tends to decrease as the flexibility of the fit increases"; test error "displays a characteristic U-shape"; the 10-fold CV error "provides a pretty good approximation to the test error rate. While it somewhat underestimates the error rate, it reaches a minimum when fourth-order polynomials are used, which is very close to the minimum of the test curve, which occurs when third-order polynomials are used." The true test error is about the same for third to sixth order.
- Figure 5.8 (right), KNN: "the training error rate cannot be used to select the optimal value for K"; the CV curve "slightly underestimates the test error rate" but has its minimum "very close to the best value for K".

## 5.2 The Bootstrap (pp. 212-215)

- Purpose: "a widely applicable and extremely powerful statistical tool that can be used to quantify the uncertainty associated with a given estimator or statistical learning method". Its power is that it "can be easily applied to a wide range of statistical learning methods, including some for which a measure of variability is otherwise difficult to obtain and is not automatically output by statistical software".
- Toy problem: invest a fraction alpha in asset X and 1 − alpha in Y to minimize Var(alpha X + (1 − alpha) Y).
  - (5.6) `alpha = (sigma_Y^2 − sigma_XY) / (sigma_X^2 + sigma_Y^2 − 2 sigma_XY)`.
  - (5.7) the same with estimated variances and covariance gives alpha-hat.
- Simulation with known truth: sigma_X^2 = 1, sigma_Y^2 = 1.25, sigma_XY = 0.5, so alpha = 0.6. Four simulated sets of 100 pairs give alpha-hat = 0.576, 0.532, 0.657, 0.651 (Figure 5.9). Repeating 1,000 times: mean of alpha-hat = 0.5996, standard deviation 0.083, so SE(alpha-hat) is about 0.083: "for a random sample from the population, we would expect alpha-hat to differ from alpha by approximately 0.08, on average."
- The problem with real data: "we cannot generate new samples from the original population". The bootstrap answer: "Rather than repeatedly obtaining independent data sets from the population, we instead obtain distinct data sets by repeatedly sampling observations from the original data set."
- Mechanics (Figure 5.11, n = 3): draw n observations from the data set Z with replacement to form a bootstrap data set Z∗1 ("the same observation can occur more than once"). "Note that if an observation is contained in Z∗1, then both its X and Y values are included." Compute alpha-hat∗1. Repeat B times.
- (5.8) bootstrap standard error: `SE_B(alpha-hat) = sqrt( (1/(B − 1)) sum_{r=1}^{B} ( alpha-hat∗r − (1/B) sum_{r'=1}^{B} alpha-hat∗r' )^2 )`.
- Result: with B = 1,000 bootstrap samples from a single data set the bootstrap SE is 0.087, "very close to the estimate of 0.083 obtained using 1,000 simulated data sets"; the bootstrap histogram resembles the idealized one (Figure 5.10).

## 5.3 Lab: Cross-Validation and the Bootstrap (pp. 215-224)

- Validation set (pp. 216-217): `train_test_split(Auto, test_size=196, random_state=0)`. "It is generally a good idea to set a random seed". Validation MSE for degree 1, 2, 3: 23.62, 18.76, 18.80. With `random_state=3`: 20.76, 16.95, 16.97. Same conclusion, different numbers.
- Cross-validation (pp. 217-220): `cross_validate(model, X, Y, cv=...)`; an integer cv gives K-fold, cv = n gives LOOCV. LOOCV MSE for degrees 1-5: 24.2315, 19.2482, 19.3350, 19.4244, 19.0332. 10-fold (`KFold(n_splits=10, shuffle=True, random_state=0)`, with the code comment "use same splits for each degree"): 24.2077, 19.1853, 19.2763, 19.4785, 19.1372.
- `cross_validate()` "is flexible and can take different splitting mechanisms as an argument" (p. 219), for example `ShuffleSplit`.
- Warning on repeated random splits (p. 220): ten `ShuffleSplit` repetitions give mean 23.8022 and standard deviation 1.4218, but "this standard deviation is not a valid estimate of the sampling variability of the mean test score or the individual scores, since the randomly-selected training samples overlap and hence introduce correlations. But it does give an idea of the Monte Carlo variation incurred by picking different random folds."
- Bootstrap (pp. 220-224): "it can be applied in almost all situations. No complicated mathematical calculations are required." Pattern: a function `func(D, idx)` computing the statistic on rows idx; `boot_SE(func, D, n=None, B=1000, seed=0)` draws `rng.choice(D.index, n, replace=True)` B times and returns the standard deviation of the B values.
  - Portfolio: alpha-hat = 0.5758 on all 100 observations; one bootstrap replicate 0.6074; bootstrap SE with B = 1,000 is 0.0912.
  - Regression coefficients (mpg on horsepower): bootstrap SE 0.8488 (intercept) and 0.0074 (slope) versus formula SE 0.717 and 0.006. The book's reading (p. 223): this "suggests the opposite" of a bootstrap problem. The formulas rely on an estimate of the noise variance that assumes the linear model is correct (it is not: the relationship is non-linear) and on the x_i being fixed. "The bootstrap approach does not rely on any of these assumptions, and so it is likely giving a more accurate estimate of the standard errors".
  - Quadratic model (which fits well): bootstrap 2.067840, 0.033019, 0.000120 versus formula 1.800, 0.031, 0.000: closer agreement.
  - The bootstrap function uses `clone()` so that derived features "will be re-fit on the resampled data frame".

## 5.4 Exercises (pp. 224-228, skimmed)

- (2) probability that observation j is not in a bootstrap sample is (1 − 1/n)^n. (Derived arithmetic: the inclusion probability 1 − (1 − 1/n)^n is 0.672 for n = 5, 0.634 for n = 100, and tends to 1 − 1/e = 0.632; the book asks for these, it does not print them.)
- (3) advantages and disadvantages of k-fold versus validation set and LOOCV. (4) estimating the standard deviation of a prediction.
- (5) validation-set estimate for logistic regression on Default with three different splits. (6) bootstrap versus formula standard errors of logistic regression coefficients. (7) LOOCV by hand for logistic regression. (8) LOOCV on simulated data with different seeds (LOOCV is deterministic). (9) bootstrap SE of the mean, the median and the tenth percentile of medv; hint for a 95 % confidence interval: `[mu-hat − 2 SE(mu-hat), mu-hat + 2 SE(mu-hat)]`; "there is no simple formula for computing the standard error of the median".

## Where ISLP states the right-way / wrong-way rule for cross-validation

Chapter 5 has no section on this. The rule is stated in Chapter 6:

- §6.1.1 (p. 231): "If cross-validation is used to select the best model, then Step 2 is repeated on each training fold, and the validation errors are averaged to select the best value of k. Then the model M_k fit on the full training set is delivered for the chosen k."
- §6.1.3 (p. 239): "when cross-validation is used, the sequence of models M_k in Algorithms 6.1–6.3 is determined separately for each training fold ... This means, for example with best-subset regression, that M_k, the best subset of size k, can differ across the folds. Once the best size k is chosen, we find the best model of that size on the full data set."
- Lab §6.5.1 (p. 271), the clearest statement: "We must use only the training observations to perform all aspects of model-fitting — including variable selection. Therefore, the determination of which model of a given size is best must be made using only the training observations in each training fold. This point is subtle but important. If the full data set is used to select the best subset at each step, then the validation set errors and cross-validation errors that we obtain will not be accurate estimates of the test error."
- Lab §6.5.2 (p. 277): standardization inside a `Pipeline` under `GridSearchCV` "is carried out on each fold"; with `ElasticNetCV` inside a pipeline "it is carried out only once", so results differ slightly.
- Lab §6.5.2 (p. 278): after choosing lambda by CV, "cross-validation will have touched all of its data in choosing lambda, hence we have no further data to estimate test error". Remedy: split off a test set first and run CV on the training part only, "cross-validation nested within the validation set approach" (75 % / 25 % in the lab; test MSE 132393.84).

Wrong way: use all the data (including what will later be held out) to choose features, standardization constants, hyperparameters or thresholds, then cross-validate only the final fit. Right way: every step that looks at the response is repeated inside each training fold; the held-out fold is used only to score.

(From *The Elements of Statistical Learning* §7.10.2, recalled from memory and not part of the text read here, so verify before citing: with 50 samples, 5,000 pure-noise predictors and a true error rate of 50 %, screening the 100 predictors most correlated with the labels on all the data and then cross-validating a 1-nearest-neighbour classifier gave an average CV error of about 3 %.)

## Application to Qaf-ft (Chapter 5)

1. Hold out by SPEAKER, not by word or clip (Extension; ISLP's folds are a random division of independent observations and the book does not discuss grouped data). The book's argument for why a held-out error is honest is that the held-out observation "was not used in the fitting process" (p. 204). Words in one clip and clips from one reciter share voice, microphone, room, tempo and error habits. If the same reciter is in both the training and validation folds, the validation words are not new in the sense that matters, and the CV error is optimistic in the same direction as the training error. Rules:
   - Folds are formed over speaker IDs (sklearn `GroupKFold` / `StratifiedGroupKFold` passed as `cv=`; the lab notes that `cross_validate()` "can take different splitting mechanisms").
   - A synthetic-error clip derived from a real clip goes in the same fold as its source clip and that source's speaker. Otherwise near-duplicate audio sits on both sides of the split.
   - The acoustic model is part of "all aspects of model-fitting". If the CTC model was fine-tuned on a speaker, verifier features for that speaker come from a model that has seen that voice, and they will look cleaner than for a new user. Verifier training and evaluation features should come from speakers the acoustic model did not train on, or from cross-fitted acoustic models.
   - If speaker IDs are unreliable or missing for some clips, group by the most conservative unit available and state it.

2. Everything that touches labels goes inside the loop (pp. 231, 239, 271, 277). For the verifier that means: feature selection, feature standardization constants, the ridge/lasso lambda, any calibration map, the prior correction, and the decision threshold. Selecting features or the threshold on the pooled data and then reporting the cross-validated false-flag rate is the wrong way: the choice was tuned to the labels of the words that are then used to score it, so the reported rate is too low. This matters most for exactly the quantity Qaf-ft cares about, because a threshold tuned to hit a 1 % false-flag rate on a sample will show 1 % on that sample by construction.

3. A workable evaluation design with about 900 labelled learner clips (Extension, following p. 278's nested scheme):
   - Set aside a final test set of speakers first; touch it once.
   - On the remaining speakers run speaker-grouped 5- or 10-fold CV (§5.1.4 recommendation). Inside each training fold: fit scaler, select features, choose lambda with an inner grouped CV, fit the verifier, pick the threshold for the target false-flag rate from inner out-of-fold scores of correct words. On the outer held-out fold: record FP, N, TP, P at that threshold.
   - Pool the outer-fold counts to estimate FP/N and TP/P. Then refit everything on all development data for the delivered model ("the model is re-fit using all of the available observations", p. 252).
   - Stratify folds so that each one contains real mistakes; with rare positives an unstratified fold can contain almost none (Extension).

4. Choice of k. Use k = 5 or 10 over speakers. Leave-one-speaker-out is affordable for a small logistic verifier but there is no shortcut formula (5.2 holds only for least squares) and the estimate has higher variance (§5.1.4). A single train/validation split is the weakest option for a data set this small: Figure 5.2 shows ten splits giving ten different answers, and it overestimates error because it trains on a fraction of the data.

5. Use CV for selection, a clean test set for the headline number. CV curves locate the best flexibility well even when their level is off (p. 208, Figure 5.6), so trust CV to choose among verifier variants, but quote the final false-flag rate from data that played no part in any choice.

6. Cross-validate the metric that matters. (5.4) uses misclassification rate, which is the accuracy number that §4.4.2 shows to be uninformative for rare positives. The Chapter 6 lab notes "One can cross-validate different metrics to choose a parameter" (p. 277). For Qaf-ft cross-validate the detection rate at the fixed false-flag rate, or the binomial deviance, not accuracy.

7. Repeat the CV with different fold assignments to see the Monte Carlo variation (Figure 5.4), but do not turn the spread across repeats or folds into a confidence interval: the lab says such a standard deviation "is not a valid estimate of the sampling variability" because training sets overlap (p. 220; also footnote 9, p. 271, "crude"). Use the bootstrap for intervals.

8. Bootstrap confidence intervals for word error rate, resampling by speaker (Extension built on §5.2). WER is a ratio (substitutions + deletions + insertions over reference words) summed over utterances of unequal length; like the median in exercise 9 it has no simple standard-error formula, and its terms are correlated within speaker. Procedure:
   - Unit of resampling = speaker. Draw S speakers with replacement from the S test speakers; a speaker drawn twice contributes all of that speaker's clips twice. The book's rule that a resampled observation brings "both its X and Y values" generalizes to: a resampled speaker brings every clip, reference, hypothesis and label.
   - Recompute the pooled WER (total errors over total reference words) on each bootstrap sample. B = 1,000 as in the book.
   - SE from (5.8). Interval: the book's own form is estimate ± 2 SE (exercise 9d). A percentile interval (2.5th and 97.5th percentiles of the B values) is the usual alternative for a bounded, skewed statistic (Extension).
   - Resampling words or clips instead of speakers treats correlated items as independent and gives intervals that are too narrow.
   - The same procedure applies to diacritic error rate and phoneme error rate.

9. Bootstrap confidence interval for the false-flag rate (Extension). Same speaker-level resampling; statistic = FP/N at the frozen threshold. Notes:
   - To include the uncertainty from choosing the threshold, redo the threshold choice inside each replicate and score on speakers not drawn in that replicate; with the threshold frozen the interval covers only test-sampling uncertainty. Say which one is reported.
   - Rough size of the problem (derived arithmetic, binomial, assuming independent words, which is optimistic): the standard error of an estimated rate alpha from N correct words is sqrt(alpha(1 − alpha)/N). For alpha = 1 % and N = 5,000 it is 0.14 percentage points. For alpha = 0.1 % and N = 5,000 only about 5 false flags are expected and the standard error is 0.045 points, a relative error of about 45 %. A very low false-flag target cannot be verified on a small test set; correlated words within speakers make it harder still.
   - With zero false flags observed the bootstrap interval collapses to zero width, which is not evidence of a zero rate. Report an exact binomial upper bound instead (roughly 3/N at 95 % under independence; Extension, not in ISLP).
   - For comparing two systems, resample the same speakers for both and bootstrap the difference (paired bootstrap, Extension).

10. What the bootstrap does not fix. It measures sampling variability of the data in hand. If the test clips are unrepresentative (for example, mistakes are mostly synthetic), the interval is a tight interval around the wrong quantity. Report real-learner and synthetic results separately, each with its own interval.

11. Model-free standard errors for the verifier's coefficients. The lab's lesson (p. 223) is that formula standard errors are wrong when their assumptions fail and the bootstrap is then more trustworthy. Logistic-regression z-statistics assume independent words; use the speaker bootstrap of the whole fitting pipeline (the lab's `clone()` pattern) to see whether a feature's coefficient is stable in sign and size.

---

# Chapter 6. Linear Model Selection and Regularization (pp. 229-288)

## 6.0 Chapter opening (pp. 229-230)

- (6.1) `Y = b0 + b1 X1 + ... + bp Xp + e`.
- Two reasons to replace plain least squares:
  - Prediction accuracy. With n much larger than p least squares has low variance. "if n is not much larger than p, then there can be a lot of variability in the least squares fit, resulting in overfitting and consequently poor predictions on future observations". "if p > n, then there is no longer a unique least squares coefficient estimate: there are infinitely many solutions. Each of these least squares solutions gives zero error on the training data, but typically very poor test set performance due to extremely high variance." "By constraining or shrinking the estimated coefficients, we can often substantially reduce the variance at the cost of a negligible increase in bias."
  - Model interpretability. Irrelevant variables add "unnecessary complexity"; "least squares is extremely unlikely to yield any coefficient estimates that are exactly zero". Feature (variable) selection removes them.
- Footnote 1 (p. 230; the comparison symbol is garbled, evidently "p much larger than n"): the least squares solution with the smallest sum of squared coefficients "can sometimes perform quite well. See Section 10.8".
- Three classes of methods: subset selection; shrinkage (regularization); dimension reduction.
- Scope (p. 230): "the same concepts apply to other methods, such as the classification models seen in Chapter 4."

## 6.1 Subset Selection (pp. 231-240)

### 6.1.1 Best Subset Selection (pp. 231-233)

Algorithm 6.1 (best subset selection):

1. M_0 = null model (no predictors; predicts the sample mean).
2. For k = 1, ..., p: fit all (p choose k) models with exactly k predictors; M_k = the one with smallest RSS (largest R2).
3. Select one model among M_0, ..., M_p "using the prediction error on a validation set, Cp (AIC), BIC, or adjusted R2. Or use the cross-validation method."

- Step 3 must not use RSS or R2: "the RSS of these p + 1 models decreases monotonically, and the R2 increases monotonically, as the number of features included in the models increases ... a low RSS or a high R2 indicates a model with a low training error, whereas we wish to choose a model that has a low test error."
- With cross-validation, Step 2 is repeated on each training fold (p. 231).
- For logistic regression use the deviance in place of RSS: "negative two times the maximized log-likelihood; the smaller the deviance, the better the fit" (p. 232).
- Cost: 2^p models. p = 10 gives about 1,000; p = 20 "over one million"; "computationally infeasible for values of p greater than around 40".
- Credit example (Figure 6.1): ten predictors, eleven variables after dummy coding; "from the three-variable model on, there is little improvement in RSS and R2".

### 6.1.2 Stepwise Selection (pp. 233-235)

- Statistical reason to restrict the search (p. 233): "The larger the search space, the higher the chance of finding models that look good on the training data, even though they might not have any predictive power on future data. Thus an enormous search space can lead to overfitting and high variance of the coefficient estimates."

Algorithm 6.2 (forward stepwise): start from M_0; for k = 0, ..., p − 1 consider the p − k models that add one predictor to M_k and keep the best (smallest RSS / highest R2) as M_{k+1}; then select among M_0, ..., M_p by validation error, Cp (AIC), BIC, adjusted R2 or cross-validation.

- Fits 1 + p(p + 1)/2 models. For p = 20: 211 versus 1,048,576 for best subset.
- Not guaranteed to find the best model (nested by construction). Table 6.1 (Credit): one variable: rating (both methods); two: rating, income; three: rating, income, student; four: best subset picks cards, income, student, limit, forward stepwise picks rating, income, student, limit.
- Usable when n < p, but only up to M_{n−1}.

Algorithm 6.3 (backward stepwise): start from the full model M_p; for k = p, ..., 1 drop the least useful predictor to get M_{k−1}; select as above. Also 1 + p(p + 1)/2 models. "Backward selection requires that the number of samples n is larger than the number of variables p". Forward stepwise "is the only viable subset method when p is very large".

Hybrid approaches add variables sequentially and may remove variables that no longer help.

Footnotes 2 and 3: both stepwise methods perform a guided search, so "the effective model space considered contains substantially more than p(p + 1)/2 + 1 models".

### 6.1.3 Choosing the Optimal Model (pp. 235-240)

Two routes: (1) indirectly, adjust the training error for overfitting; (2) directly, validation set or cross-validation.

Adjusted training-error criteria for a least squares model with d predictors, sigma^2-hat an estimate of the error variance ("Typically ... estimated using the full model containing all predictors"):

- (6.2) `Cp = (1/n)(RSS + 2 d sigma^2-hat)`. Penalty 2 d sigma^2-hat. If sigma^2-hat is unbiased, Cp is an unbiased estimate of test MSE. Choose the lowest. Footnote 4: Mallow's version `Cp' = RSS/sigma^2-hat + 2d − n`, with `Cp = (1/n) sigma^2-hat (Cp' + n)`, same minimizer.
- AIC (unnumbered, p. 237): `AIC = (1/n)(RSS + 2 d sigma^2-hat)` for least squares with Gaussian errors, constants omitted; "Cp and AIC are proportional to each other". AIC "is defined for a large class of models fit by maximum likelihood". Footnote 5: a second AIC formula with log(RSS) applies when sigma^2 is not estimated explicitly.
- (6.3) `BIC = (1/n)(RSS + log(n) d sigma^2-hat)`. "Since log n > 2 for any n > 7, the BIC statistic generally places a heavier penalty on models with many variables, and hence results in the selection of smaller models than Cp."
- (6.4) `Adjusted R2 = 1 − (RSS/(n − d − 1)) / (TSS/(n − 1))`. Larger is better; maximizing it is equivalent to minimizing RSS/(n − d − 1). Adding noise variables increases d with little RSS decrease, so adjusted R2 falls.
- Credit results: Cp selects six variables (income, limit, rating, cards, age, student); BIC selects four (income, limit, cards, student); adjusted R2 selects seven (adds own). The curves are flat after four variables.
- Standing of the criteria (p. 238): Cp, AIC, BIC "have rigorous theoretical justifications" that "rely on asymptotic arguments (scenarios where the sample size n is very large)". Adjusted R2 "is not as well motivated in statistical theory". AIC and BIC "can also be defined for more general types of models".

Validation and cross-validation (pp. 238-240):

- Advantage over the criteria: "it provides a direct estimate of the test error, and makes fewer assumptions about the true underlying model. It can also be used in a wider range of model selection tasks, even in cases where it is hard to pinpoint the model degrees of freedom ... or hard to estimate the error variance".
- "nowadays with fast computers, the computations required to perform cross-validation are hardly ever an issue. Thus, cross-validation is a very attractive approach".
- Credit (Figure 6.3): validation set (three-quarters training) and 10-fold CV both pick a six-variable model, but "all three approaches suggest that the four-, five-, and six-variable models are roughly equivalent"; the estimated test errors of the 3- to 11-variable models "are quite similar"; a different split "would surely change" the exact minimizer.

One-standard-error rule (p. 240): "We first calculate the standard error of the estimated test MSE for each model size, and then select the smallest model for which the estimated test error is within one standard error of the lowest point on the curve. The rationale here is that if a set of models appear to be more or less equally good, then we might as well choose the simplest model—that is, the model with the smallest number of predictors." On Credit it selects the three-variable model.

## 6.2 Shrinkage Methods (pp. 240-253)

Fit all p predictors but constrain (regularize) the coefficients; "shrinking the coefficient estimates can significantly reduce their variance".

### 6.2.1 Ridge Regression (pp. 240-244)

- Least squares minimizes `RSS = sum_{i=1}^{n} ( y_i − b0 − sum_{j=1}^{p} b_j x_ij )^2`.
- (6.5) ridge minimizes `RSS + lambda sum_{j=1}^{p} b_j^2`, with tuning parameter lambda >= 0. The second term is the shrinkage penalty.
- lambda = 0 gives least squares; lambda to infinity sends the coefficients to zero (the null model). One coefficient vector per lambda; "Selecting a good value for lambda is critical".
- The intercept is not penalized (p. 241): "We want to shrink the estimated association of each variable with the response; however, we do not want to shrink the intercept". With centered predictors, b0-hat = y-bar.
- l2 norm: `||b||_2 = sqrt( sum_j b_j^2 )`. The ratio `||b-hat^R_lambda||_2 / ||b-hat||_2` runs from 1 (lambda = 0) to 0 (lambda infinite) and measures the amount of shrinkage.
- Individual coefficients "may occasionally increase as lambda increases" although the norm always decreases.
- Scale: least squares is scale equivariant; ridge is not. The ridge fit for X_j depends on lambda, on the scaling of X_j, and "may even depend on the scaling of the other predictors". "Therefore, it is best to apply ridge regression after standardizing the predictors":
  - (6.6) `x~_ij = x_ij / sqrt( (1/n) sum_{i=1}^{n} (x_ij − x-bar_j)^2 )`.
- Why it helps (pp. 242-244): bias-variance. "As lambda increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias." Simulation with p = 45, n = 50 (Figure 6.5): for lambda up to about 10 "the variance decreases rapidly, with very little increase in bias"; minimum test MSE at approximately lambda = 30; the least squares MSE (lambda = 0) "is almost as high as that of the null model".
- When it works best: "in situations where the least squares estimates have high variance", i.e. p close to n; for p > n least squares has no unique solution but ridge "can still perform well".
- Computation: one fit per lambda, and the whole path costs about one least squares fit, versus 2^p fits for best subset.

### 6.2.2 The Lasso (pp. 244-251)

- Ridge's disadvantage: it "will include all p predictors in the final model"; no coefficient is set exactly to zero (unless lambda is infinite). Fine for accuracy, a problem for interpretation.
- (6.7) lasso minimizes `RSS + lambda sum_{j=1}^{p} |b_j|`. l1 norm `||b||_1 = sum |b_j|`.
- The l1 penalty forces "some of the coefficient estimates to be exactly equal to zero when the tuning parameter lambda is sufficiently large". The lasso "performs variable selection" and "yields sparse models".
- Credit path (Figure 6.6): rating enters first, then student and limit almost together, then income.

Constrained formulations (pp. 245-246):

- (6.8) lasso: minimize RSS subject to `sum_j |b_j| <= s`.
- (6.9) ridge: minimize RSS subject to `sum_j b_j^2 <= s`.
- (6.10) best subset: minimize RSS subject to `sum_j I(b_j != 0) <= s`. Computationally infeasible for large p; ridge and lasso are "computationally feasible alternatives to best subset selection that replace the intractable form of the budget". "the lasso is much more closely related to best subset selection".
- For every lambda there is an s giving the same solution.

Geometry (Figure 6.7, pp. 246-247): for p = 2 the lasso constraint region is a diamond `|b1| + |b2| <= s`, the ridge region a circle `b1^2 + b2^2 <= s`. RSS contours are ellipses centered at the least squares estimate. The solution is "the first point at which an ellipse contacts the constraint region". The circle has no corners, so contact is generally off the axes and all ridge coefficients are non-zero. "the lasso constraint has corners at each of the axes, and so the ellipse will often intersect the constraint region at an axis. When this occurs, one of the coefficients will equal zero." In higher dimensions (polytope versus hypersphere) many coefficients can be zero at once.

Lasso versus ridge on prediction (pp. 247-249):

- Figure 6.8: all 45 predictors related to the response. Biases almost identical, ridge variance slightly lower, so "the minimum MSE of ridge regression is slightly smaller than that of the lasso".
- Figure 6.9: response depends on only 2 of 45 predictors. "Now the lasso tends to outperform ridge regression in terms of bias, variance, and MSE."
- Rule (p. 248): "neither ridge regression nor the lasso will universally dominate the other. In general, one might expect the lasso to perform better in a setting where a relatively small number of predictors have substantial coefficients, and the remaining predictors have coefficients that are very small or that equal zero. Ridge regression will perform better when the response is a function of many predictors, all with coefficients of roughly equal size. However, the number of predictors that is related to the response is never known a priori for real data sets. A technique such as cross-validation can be used in order to determine which approach is better on a particular data set."
- Models with different regularization can be compared by indexing them on training R2 (Figure 6.8, right).

Special case n = p, X the identity, no intercept (pp. 249-250):

- (6.11) least squares minimizes `sum_j (y_j − b_j)^2`, solution b_j-hat = y_j.
- (6.12) ridge objective `sum_j (y_j − b_j)^2 + lambda sum_j b_j^2`; (6.13) lasso objective `sum_j (y_j − b_j)^2 + lambda sum_j |b_j|`.
- (6.14) ridge: `b_j-hat^R = y_j / (1 + lambda)`: every coefficient shrunk by the same proportion.
- (6.15) lasso: `b_j-hat^L = y_j − lambda/2 if y_j > lambda/2; y_j + lambda/2 if y_j < −lambda/2; 0 if |y_j| <= lambda/2`: soft-thresholding, a constant shift toward zero, and small coefficients become exactly zero.
- General case: "ridge regression more or less shrinks every dimension of the data by the same proportion, whereas the lasso more or less shrinks all coefficients toward zero by a similar amount, and sufficiently small coefficients are shrunken all the way to zero."

Bayesian interpretation (pp. 250-251): posterior `p(b | X, Y) proportional to f(Y | X, b) p(b)`, with `p(b) = prod_j g(b_j)`.

- g Gaussian with mean zero: the posterior mode is the ridge solution (and it is also the posterior mean).
- g double-exponential (Laplace) with mean zero: the posterior mode is the lasso solution (not the posterior mean, which is not sparse).
- "the lasso prior is steeply peaked at zero ... the lasso expects a priori that many of the coefficients are (exactly) zero, while ridge assumes the coefficients are randomly distributed about zero."

Exercise 5 (p. 285) adds: "ridge regression tends to give similar coefficient values to correlated variables, whereas the lasso may give quite different coefficient values to correlated variables"; with two identical predictors the lasso solution is not unique.

### 6.2.3 Selecting the Tuning Parameter (pp. 252-253)

- Procedure: "We choose a grid of lambda values, and compute the cross-validation error for each value of lambda ... We then select the tuning parameter value for which the cross-validation error is smallest. Finally, the model is re-fit using all of the available observations and the selected value of the tuning parameter."
- Figure 6.12 (ridge on Credit, LOOCV): the selected lambda is small and "the dip is not very pronounced, so there is rather a wide range of values that would give a very similar error. In a case like this we might simply use the least squares solution."
- Figure 6.13 (lasso, ten-fold CV, sparse simulation with p = 45, n = 50): the CV minimum corresponds to a fit in which only the two signal variables are non-zero, whereas least squares "assigns a large coefficient estimate to only one of the two signal variables". Signal versus noise variables.
- Note: the text applies the one-standard-error rule to subset size (§6.1.3); in §6.2.3 it picks the lambda with the smallest CV error. Applying the one-standard-error rule to lambda is the natural analogue but is an Extension here.

## 6.3 Dimension Reduction Methods (pp. 253-262)

- (6.16) `Z_m = sum_{j=1}^{p} phi_jm X_j`, M < p linear combinations.
- (6.17) `y_i = theta_0 + sum_{m=1}^{M} theta_m z_im + e_i`, fit by least squares.
- (6.18) implied constraint `b_j = sum_{m=1}^{M} theta_m phi_jm`. The constraint can bias the coefficients, but when p is large relative to n, choosing M much smaller than p "can significantly reduce the variance of the fitted coefficients". M = p with independent Z_m is just least squares.

### 6.3.1 Principal Components Regression (pp. 254-260)

- First principal component direction = that "along which the observations vary the most"; equivalently the line "as close as possible to the data". Example (pop, ad; 100 cities): (6.19) `Z1 = 0.839 × (pop − pop-bar) + 0.544 × (ad − ad-bar)`, loadings constrained by phi_11^2 + phi_21^2 = 1; (6.20) scores z_i1 (examples −26.1 and 18.7). Second component `Z2 = 0.544 × (pop − pop-bar) − 0.839 × (ad − ad-bar)`: uncorrelated with Z1, i.e. orthogonal direction, largest variance subject to that.
- PCR: regress Y on the first M components. Assumption: "the directions in which X1, ..., Xp show the most variation are the directions that are associated with Y. While this assumption is not guaranteed to be true, it often turns out to be a reasonable enough approximation".
- Simulations (Figures 6.18, 6.19; n = 50, p = 45): bias falls and variance rises with M (U-shaped MSE); PCR with a good M beats least squares, but on the data of Figures 6.8/6.9 it "does not perform as well as the two shrinkage methods". When the response depends only on the first five components, MSE has a clear minimum at M = 5 and "PCR and ridge regression slightly outperform the lasso".
- Not feature selection (p. 259): each component "is a linear combination of all p of the original features". "PCR is more closely related to ridge regression than to the lasso ... One can even think of ridge regression as a continuous version of PCR!" (footnote 8 points to *The Elements of Statistical Learning*, Section 3.5).
- M is "typically chosen by cross-validation". Credit: lowest CV error at M = 10, "almost no dimension reduction at all" (M = 11 is least squares).
- Standardize each predictor with (6.6) before computing components; otherwise "the high-variance variables will tend to play a larger role". Exception: variables all in the same units.

### 6.3.2 Partial Least Squares (pp. 260-262)

- PCR's directions are found "in an unsupervised way"; "there is no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response."
- PLS is supervised. First direction: after standardizing, set each phi_j1 to the coefficient of the simple regression of Y on X_j (proportional to their correlation), so PLS "places the highest weight on the variables that are most strongly related to the response". Later directions: regress each variable on Z1, take residuals, repeat. Then least squares on Z1, ..., ZM.
- M chosen by cross-validation; "We generally standardize the predictors and response before performing PLS."
- Verdict (p. 262): "In practice it often performs no better than ridge regression or PCR. While the supervised dimension reduction of PLS can reduce bias, it also has the potential to increase variance, so that the overall benefit of PLS relative to PCR is a wash."

## 6.4 Considerations in High Dimensions (pp. 262-267)

### 6.4.1 High-Dimensional Data (pp. 262-263)

- Low-dimensional: n much greater than p. High-dimensional: more features than observations (examples: n about 200 with p about 500,000 SNPs; bag-of-words with n about 1,000).
- "the considerations that we will now discuss certainly also apply if p is slightly smaller than n, and are best always kept in mind when performing supervised learning."

### 6.4.2 What Goes Wrong in High Dimensions? (pp. 263-265)

- "The same concepts apply to logistic regression, linear discriminant analysis, and other classical statistical approaches."
- When p >= n: "regardless of whether or not there truly is a relationship between the features and the response, least squares will yield a set of coefficient estimates that result in a perfect fit to the data, such that the residuals are zero." (Figure 6.22: two points, one line.) "when p > n or p ≈ n, a simple least squares regression line is too flexible and hence overfits the data."
- Figure 6.23 (n = 20, 1 to 20 features "completely unrelated to the response"): R2 rises to 1, training MSE falls to 0, test MSE "becomes extremely large". "someone who carelessly examines only the R2 or the training set MSE might erroneously conclude that the model with the greatest number of variables is best."
- "the Cp, AIC, and BIC approaches are not appropriate in the high-dimensional setting, because estimating sigma^2-hat is problematic"; adjusted R2 can easily reach 1.

### 6.4.3 Regression in High Dimensions (pp. 265-266)

- Forward stepwise, ridge, lasso and PCR "are particularly useful" because they are less flexible than least squares.
- Figure 6.24 (lasso, n = 100, p = 20, 50, 2,000, of which 20 are truly associated): three points: "(1) regularization or shrinkage plays a key role in high-dimensional problems, (2) appropriate tuning parameter selection is crucial for good predictive performance, and (3) the test error tends to increase as the dimensionality of the problem (i.e. the number of features or predictors) increases, unless the additional features are truly associated with the response."
- Curse of dimensionality: "the test set MSE almost doubles as p increases from 20 to 2,000." Noise features "increase the dimensionality of the problem, exacerbating the risk of overfitting (since noise features may be assigned nonzero coefficients due to chance associations with the response on the training set) without any potential upside". Even relevant features can hurt: "the variance incurred in fitting their coefficients may outweigh the reduction in bias that they bring."

### 6.4.4 Interpreting Results in High Dimensions (pp. 266-267)

- Multicollinearity is extreme: "any variable in the model can be written as a linear combination of all of the other variables in the model ... we can never know exactly which variables (if any) truly are predictive of the outcome".
- A selected model is "simply one of many possible models", and "must be further validated on independent data sets". Another data set would likely give "a different, and perhaps even non-overlapping, set" of features.
- Reporting rule: "one should never use sum of squared errors, p-values, R2 statistics, or other traditional measures of model fit on the training data as evidence of a good model fit in the high-dimensional setting." "It is important to instead report results on an independent test set, or cross-validation errors."

## 6.5 Lab: Linear Models and Regularization Methods (pp. 267-282)

Hitters data: Salary missing for 59 players; after `dropna()` the shape is (263, 20); 19 predictors.

- Forward selection with Cp (pp. 268-270): custom negative-Cp scorer, sigma^2 estimated from the full model. Scoring by MSE selects all 19 variables; scoring by Cp selects 10 (Assists, AtBat, CAtBat, CRBI, CRuns, CWalks, Division, Hits, PutOuts, Walks).
- Forward selection with validation and CV (pp. 270-272): full path of 20 models including the null model. "the in-sample MSE improves each step we take, indicating we must use either the validation or cross-validation approach to select the number of steps." 5-fold CV with the selection redone in each fold (the p. 271 rule quoted above). Standard error of the CV mean: `cv_mse.std(1) / np.sqrt(K)`, plotted as error bars; footnote 9: "The estimate is crude because the five error estimates are based on overlapping training sets, and hence are not independent." "the validation set approach does not provide standard errors."
- Best subset with `l0bnb` (pp. 272-273): a path with the subset size as a penalty rather than a constraint.
- Ridge (pp. 273-279): `skl.ElasticNet` with `l1_ratio=0`; "It is good practice to standardize the columns of X in these applications, if the variables are measured in different units." lambda is called `alpha` in sklearn. Grid `10**np.linspace(8, -2, 100) / Y.std()`, "essentially covering the full range of scenarios from the null model containing only the intercept, to the least squares fit". At lambda = 25.535 the l2 norm of the standardized coefficients is 24.17; at lambda = 0.2437 it is 160.4237. `Pipeline([('scaler', ...), ('ridge', ...)])` separates normalization from fitting.
  - "Choosing an a priori value of lambda for ridge regression is difficult if not impossible." Validation-set MSE: 134214.0 at lambda = 0.01; 231788.32 at lambda = 1e10 (the intercept-only model).
  - `GridSearchCV` with a validation split selects alpha = 0.005899; with 5-fold CV via `ElasticNetCV` the minimum is at 1.19e-02 with CV MSE 115526.71. (The text then says this is "a further improvement over the test MSE that we got using lambda = 4"; no lambda = 4 fit appears earlier in this lab, so that sentence looks like a leftover.) All 19 ridge coefficients are non-zero.
  - Nested estimate (p. 278): outer 75/25 split, inner 5-fold CV: test MSE 132393.84, noticeably higher than the 115526.71 CV minimum that was used for selection.
- Lasso (pp. 279-280): `ElasticNetCV(n_alphas=100, l1_ratio=1, cv=kfold)`; selected alpha 3.147; minimum CV MSE 114690.73; "6 of the 19 coefficient estimates are exactly zero", leaving 13 variables.
- PCR (pp. 280-282): pipeline scaler, PCA, linear regression; components chosen by `GridSearchCV`. Smallest CV error at 17 components, but "roughly the same when only one component is included". Null model CV MSE 204139.31. Variance explained: 38.31 % (M = 1), an additional 21.84 % (M = 2), total 60.15 %; 88.63 % by M = 6.
- PLS (p. 282): `PLSRegression(n_components=2, scale=True)`; CV error minimized at 12, "though there is little noticable difference between this point and a much lower number like 2 or 3 components".

## 6.6 Exercises (pp. 283-288, skimmed)

- (1) training versus test RSS for best subset, forward and backward stepwise; nesting properties. (2) lasso and ridge are less flexible than least squares and help "when its increase in bias is less than its decrease in variance". (3), (4) behaviour of training RSS, test RSS, variance, squared bias and irreducible error as s or lambda increases. (5) ridge versus lasso with identical predictors. (6) verify (6.14) and (6.15). (7) Bayesian derivation of lasso (double-exponential prior) and ridge (normal prior).
- Applied: (8) forward/backward stepwise and lasso on simulated polynomial data; (9) College data, least squares versus ridge, lasso, PCR, PLS on a test set; (10) training versus test error as features are added (p = 20, n = 1,000, 100 for training); (11) Boston crime rate, with the instruction to evaluate "using validation set error, cross-validation, or some other reasonable alternative, as opposed to using training error".

## Application to Qaf-ft (Chapter 6)

1. Count the effective sample size before adding verifier features. The relevant n is not the number of words. It is closer to the number of independent speakers, and for the mistake class the number of real mistake examples. Once neighbouring-word context, per-phoneme statistics, interactions or embedding dimensions are added, p can approach that n, which is the regime of §6.4.2 ("if p is slightly smaller than n" the same cautions apply). Figure 6.23's lesson applies directly: features with no relation to mistakes will still raise training-set detection and still lower real performance.

2. L2 (ridge) regularization of the verifier. Use ridge-penalized logistic regression as the default fit.
   - It is the right tool when "the least squares estimates have high variance" (few real mistakes, correlated features), and it fixes the separation instability noted on p. 146.
   - The verifier's features are few and most of them carry signal, which is the situation in which "Ridge regression will perform better" (p. 248). They are also strongly correlated; ridge "tends to give similar coefficient values to correlated variables" whereas the lasso may pick one arbitrarily (exercise 5).
   - Standardize every feature first (6.6); otherwise the penalty depends on units (seconds versus milliseconds for duration, nats versus bits for entropy).
   - Do not penalize the intercept (p. 241). In the verifier the intercept carries the base rate of mistakes; shrinking it would bias every probability.
   - In sklearn's `LogisticRegression`, the default C = 1 already applies a ridge-type penalty and a very large C removes it (lab p. 187). Treat C as a tuning parameter chosen by speaker-grouped CV, not as a default.

3. L1 (lasso) for feature screening. Use the lasso when a large pool of candidate features is added of which only a few are expected to matter (context windows, per-phoneme features), which is the Figure 6.9 situation. It returns a sparse verifier that is easier to audit and cheaper on-device. Cautions: with correlated features the selected set is unstable ("simply one of many possible models", p. 267), so do not read meaning into which of two correlated features survived; and the selection must be redone inside every training fold. Decide between ridge and lasso by cross-validation, as the book prescribes (p. 248). `ElasticNet` with an intermediate `l1_ratio` is the hybrid the lab mentions.

4. Link to weight decay in the neural acoustic model. The ridge penalty lambda × sum of squared coefficients (6.5) is the same L2 penalty that is added to a neural network's objective; the book does this in Chapter 10 (eq. 10.31, §10.7.2, p. 430, outside the range read for these notes, checked only for this cross-reference), where the weight matrices are penalized and the output-layer biases are not, mirroring the unpenalized intercept. "Weight decay" is the usual deep-learning name for this penalty (the term is not used in ISLP). What carries over from Chapter 6:
   - It trades a little bias for a large variance reduction, most valuable when parameters far outnumber observations, which is the case when a large pretrained CTC model is fine-tuned on a small corpus.
   - Bayesian reading (p. 251): a zero-mean Gaussian prior on the weights. For fine-tuning, a prior centered on the pretrained weights rather than on zero is arguably the better match (Extension, not in ISLP).
   - lambda has no a priori value ("difficult if not impossible", p. 276). Tune it on speaker-disjoint validation data, never on training loss, which always prefers lambda = 0.
   - Ridge depends on feature scale; in a network the analogous point is that the effective strength of one weight-decay value differs across layers with different scales or normalization (Extension).

5. Choosing lambda (§6.2.3 and lab). Log-spaced grid from "null model" to "unpenalized fit"; speaker-grouped k-fold CV with scaler and model in one pipeline so that standardization "is carried out on each fold"; plot the CV curve with standard-error bars; refit on all development data at the chosen lambda. If the curve is flat (as in Figure 6.12), the exact value does not matter and the choice should lean toward more regularization. Remember the lab's nested result: the CV score at the selected lambda (115526.71) was better than the honest outer test score (132393.84); the same optimism will apply to a verifier's CV score at its selected lambda and threshold.

6. One-standard-error rule for picking the simpler model (p. 240; lab p. 272). For each candidate verifier (number of features, lambda, with or without context features) compute the per-fold CV error, its mean and SE = standard deviation across folds / sqrt(K). Take the best mean, add one SE, and choose the simplest candidate under that line: fewest features, or the largest lambda (the latter is the Extension noted in §6.2.3). Rationale in the book: when models "appear to be more or less equally good, then we might as well choose the simplest". For Qaf-ft simpler also means fewer ways to produce a surprising false flag. Caveat from footnote 9: the fold-based SE is "crude" because folds share training data; with grouped folds it is still the practical yardstick.

7. Subset selection for the verifier. With around 10 candidate features best subset is feasible (about 1,000 models) but the search itself overfits ("The larger the search space, the higher the chance of finding models that look good on the training data", p. 233). Use the deviance, not accuracy or RSS, to rank logistic models of the same size (p. 232), and cross-validation to choose the size. Prefer CV over AIC/BIC here: the criteria rest on large-n asymptotics and an independent-observations n, and with correlated words it is unclear what n to put in log(n) (Extension). Never choose by training-set fit, which always picks the largest model.

8. PCR and PLS: when they would matter. They are relevant only if high-dimensional inputs are fed to the verifier, for example pooled hidden-state embeddings of the acoustic model with several hundred dimensions against a few hundred real mistakes. Then PCA followed by logistic regression (standardize first; M chosen by grouped CV) or ridge on the raw embedding are the book's options. Cautions: PCA directions are unsupervised and need not be the directions that separate mistakes from correct recitation (p. 260); PCR does not select features (p. 259); PLS "often performs no better than ridge regression or PCR" (p. 262). For the small hand-built feature set, none of this is needed.

9. High-dimensional reporting discipline (§6.4.4) applies to the whole system. The acoustic model has vastly more parameters than there are labelled clips, so training-set loss, training WER and training-set verifier accuracy say nothing about quality ("this provides absolutely no evidence of a compelling model"). Only speaker-disjoint held-out or cross-validated numbers are reportable. Feature-importance stories should be worded as "one of many possible models" and rechecked on independent data.

10. Adding features is not free (§6.4.3). Each proposed feature must earn its place in grouped CV at the operating point (detection rate at the fixed false-flag rate). "Even if they are relevant, the variance incurred in fitting their coefficients may outweigh the reduction in bias that they bring."

---

# Cross-chapter checklist for Qaf-ft

1. Define positives as mistakes; report FP/N (false-flag rate), TP/P (detection rate), precision, the confusion matrix and the null rate. Never accuracy alone (§4.4.2; Tables 4.4-4.7).
2. Fix the false-flag rate first, derive the threshold from held-out scores of correct words, then read off the detection rate (§4.4.2, Figure 4.8).
3. Verifier = ridge-penalized logistic regression on standardized features, LDA as a check, no naive Bayes on correlated features (§4.4, §4.5, §6.2).
4. Watch the intercept/base rate when synthetic errors change the class mix (pp. 141, 149).
5. Check for confounding by data source, tempo, word length and speaker before interpreting any coefficient (§4.3.4).
6. Split by speaker everywhere; keep derived synthetic clips with their source (Chapter 5, Extension).
7. Put scaling, selection, lambda, calibration and threshold inside the CV loop; keep one untouched speaker-disjoint test set (pp. 231, 239, 271, 278).
8. k = 5 or 10 grouped folds; repeat to see Monte Carlo spread, but take confidence intervals from a speaker-level bootstrap with B = 1,000 (§5.1.4, §5.2, p. 220).
9. Choose model size and lambda by CV with the one-standard-error rule; prefer the simpler model when curves are flat (p. 240, p. 252).
10. Report only held-out numbers; training-set fit is not evidence (§6.4.4).
