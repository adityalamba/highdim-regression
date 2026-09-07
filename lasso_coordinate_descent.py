"""
Lasso Regression via Coordinate Descent — implemented from first principles.

Solves:  min_beta  (1/2n) ||y - X beta||_2^2 + lambda ||beta||_1

Coordinate descent update (standard result, e.g. Friedman, Hastie, Tibshirani 2010):
For each feature j, holding all other coefficients fixed, the partial residual is

    r_j = y - X_{-j} beta_{-j}

and the optimal beta_j (assuming standardized columns, ||X_j||^2 = n) is the
soft-threshold of the correlation between X_j and r_j:

    beta_j = S( (1/n) X_j^T r_j , lambda )

where S(z, g) = sign(z) * max(|z| - g, 0) is the soft-thresholding operator.
"""

import numpy as np


def soft_threshold(z, gamma):
    """Soft-thresholding operator S(z, gamma) = sign(z) * max(|z| - gamma, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def standardize(X):
    """Standardize columns of X to zero mean, unit variance. Returns X_std, mean, std."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0  # guard against constant columns
    return (X - mu) / sigma, mu, sigma


class LassoCD:
    """
    Lasso regression solved by cyclic coordinate descent.

    Parameters
    ----------
    lam : float
        Regularization strength (lambda).
    max_iter : int
        Maximum number of full passes over all coordinates.
    tol : float
        Convergence tolerance on the max coordinate-wise change.
    fit_intercept : bool
        Whether to center y and fit an intercept.
    """

    def __init__(self, lam=1.0, max_iter=1000, tol=1e-6, fit_intercept=True):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        X_std, self.mu_, self.sigma_ = standardize(X)

        if self.fit_intercept:
            y_mean = y.mean()
            y_centered = y - y_mean
        else:
            y_mean = 0.0
            y_centered = y

        beta = np.zeros(p)
        residual = y_centered.copy()  # residual = y - X_std @ beta, beta starts at 0

        for it in range(self.max_iter):
            max_delta = 0.0
            for j in range(p):
                # Add back feature j's current contribution to residual
                residual += X_std[:, j] * beta[j]
                rho = X_std[:, j] @ residual / n
                beta_j_new = soft_threshold(rho, self.lam)
                delta = abs(beta_j_new - beta[j])
                if delta > max_delta:
                    max_delta = delta
                beta[j] = beta_j_new
                # Remove feature j's new contribution from residual
                residual -= X_std[:, j] * beta[j]

            self.n_iter_ = it + 1
            if max_delta < self.tol:
                break

        # Coefficients are in standardized-X space; convert back to original scale
        self.coef_ = beta / self.sigma_
        self.intercept_ = y_mean - np.dot(self.mu_, self.coef_) if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def lasso_path(X, y, lambdas, **kwargs):
    """Fit Lasso across a grid of lambda values, warm-starting isn't used here
    for simplicity/clarity — each fit is independent. Returns array of shape
    (len(lambdas), n_features)."""
    coefs = []
    for lam in lambdas:
        model = LassoCD(lam=lam, **kwargs).fit(X, y)
        coefs.append(model.coef_.copy())
    return np.array(coefs)


if __name__ == "__main__":
    # Quick sanity check against a known sparse signal
    rng = np.random.default_rng(0)
    n, p, k = 100, 20, 5
    X = rng.normal(size=(n, p))
    true_beta = np.zeros(p)
    true_beta[:k] = rng.uniform(2, 5, size=k)
    y = X @ true_beta + rng.normal(scale=1.0, size=n)

    model = LassoCD(lam=0.3).fit(X, y)
    print("True support:", np.nonzero(true_beta)[0])
    print("Recovered support:", np.nonzero(np.abs(model.coef_) > 1e-4)[0])
    print("Converged in", model.n_iter_, "iterations")
