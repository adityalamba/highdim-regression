# High-Dimensional Regression from Scratch: Lasso, Ridge, and Sparse Recovery

**Author:** Aditya Kumar Lamba · M.Sc. Mathematics, JIIT · [github.com/adityalamba](https://github.com/adityalamba)

## Motivation

This project implements two of the most widely used regularized regression
estimators — **Lasso** and **Ridge** — entirely from first principles (no
`sklearn.linear_model` shortcuts), and uses them to empirically study two
classical questions in high-dimensional statistics:

1. **When can Lasso exactly recover a sparse signal?** As the number of
   features `p` grows relative to the sample size `n`, model-selection
   consistency depends on how the regularization parameter scales with
   `p` and `n` (Zhao & Yu, 2006; Wainwright, 2009). This repo simulates
   that boundary directly rather than just citing the theorem.
2. **What does the bias-variance tradeoff actually look like for Ridge?**
   A Monte Carlo decomposition of expected prediction error at a fixed
   test point, verifying the textbook identity
   `E[(y_hat - f(x))^2] = Bias^2 + Variance + irreducible noise`
   empirically.

This project grew out of coursework in Mathematical Statistics, Regression
Models for Data Inference and Prediction, and Pattern Recognition Models for
Learning from Data during my M.Sc., and is meant to sit alongside my
dissertation work (comparative study of Logistic Regression, Random Forest,
and Naïve Bayes on the UCI Adult Income dataset) as evidence of hands-on
work with statistical learning theory, not just applied model-fitting.

## What's implemented

| Module | What it does |
|---|---|
| `src/lasso_coordinate_descent.py` | Lasso via cyclic coordinate descent with soft-thresholding, derived and implemented manually. Includes a `lasso_path` utility for fitting across a lambda grid. |
| `src/ridge_closed_form.py` | Ridge via the closed-form normal equations, plus a from-scratch K-fold cross-validation routine to select lambda. |
| `experiments/high_dim_simulation.py` | Simulates exact support recovery rate as `p` grows from `p < n` to `p >> n`, with lambda scaled at the theoretical rate `~ sqrt(log(p)/n)`. |
| `experiments/bias_variance_tradeoff.py` | Monte Carlo bias-variance decomposition for Ridge across a lambda grid. |

## Results

### Sparse recovery degrades as p/n grows

![Sparse recovery](figures/sparse_recovery.png)

With `n = 60` fixed and a fixed sparsity level `k = 5`, Lasso's exact support
recovery rate falls off as `p` grows past `n`, consistent with the theory
that consistent sparse recovery requires `n` to grow at least like
`k log(p)`. (Recovery rate here is based on 25 trials per `p`, so
point-to-point noise — e.g. the small uptick at `p = 300` — is expected
Monte Carlo variability, not a real reversal of the trend; more trials
would smooth this out.)

### The bias-variance tradeoff for Ridge, empirically

![Bias-variance tradeoff](figures/bias_variance_tradeoff.png)

As lambda increases, variance stays roughly flat (even decreases slightly)
while squared bias grows — reproducing the classical tradeoff from first
principles via repeated resampling, rather than citing it.

## Running it

```bash
pip install -r requirements.txt

# Sanity-check the Lasso implementation against a known sparse signal
python -m src.lasso_coordinate_descent

# Sanity-check Ridge + cross-validation
python -m src.ridge_closed_form

# Reproduce the two experiments and figures
python -m experiments.high_dim_simulation
python -m experiments.bias_variance_tradeoff
```

## Why this exists (for anyone reviewing it, e.g. for PhD admissions)

Most undergraduate/master's-level ML portfolios call `sklearn.Lasso()` and
stop there. The point of this repo is the opposite: derive the coordinate
descent update from the KKT conditions of the Lasso objective, implement it
without high-level library shortcuts, verify it against known ground truth,
and then use it to *empirically probe* a piece of high-dimensional statistics
theory (sparse recovery thresholds) rather than just fit a model to a dataset.
It's meant as a demonstration of comfort moving between the mathematical
statistics layer and its computational implementation — the kind of thing a
statistics/ML PhD involves daily.

## Possible extensions

- Add the LARS algorithm and compare its solution path to coordinate descent's.
- Replace the Gaussian design with correlated designs and study how the
  *irrepresentable condition* failing affects recovery.
- Add elastic net (Lasso + Ridge combined penalty) and compare grouped
  variable selection behavior.
- Bootstrap confidence intervals for Lasso-selected coefficients
  (post-selection inference is an active, nontrivial research area).

## References

- Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso.* JRSS-B.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). *Regularization paths
  for generalized linear models via coordinate descent.* JSS.
- Zhao, P., & Yu, B. (2006). *On model selection consistency of Lasso.* JMLR.
- Wainwright, M. J. (2009). *Sharp thresholds for high-dimensional and
  noisy sparsity recovery using L1-constrained quadratic programming (Lasso).*
  IEEE Trans. Information Theory.
- Hastie, T., Tibshirani, R., & Friedman, J. *The Elements of Statistical
  Learning*, 2nd ed., Ch. 2–3.

## License

MIT — see `LICENSE`.
