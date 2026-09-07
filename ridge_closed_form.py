"""
Ridge Regression — closed-form solution and K-fold cross-validation, from scratch.

Solves:  min_beta  ||y - X beta||_2^2 + lambda ||beta||_2^2

Closed-form solution (standard normal-equations derivation):
    beta_hat = (X^T X + lambda I)^{-1} X^T y

This module also demonstrates the classical bias-variance identity for the
ridge estimator, used later in experiments/bias_variance_tradeoff.py.
"""

import numpy as np
from src.lasso_coordinate_descent import standardize


class RidgeCF:
    """Ridge regression via the closed-form normal equations."""

    def __init__(self, lam=1.0, fit_intercept=True):
        self.lam = lam
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

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

        A = X_std.T @ X_std + self.lam * np.eye(p)
        b = X_std.T @ y_centered
        beta = np.linalg.solve(A, b)  # solve linear system directly (more stable than inverting)

        self.coef_ = beta / self.sigma_
        self.intercept_ = y_mean - np.dot(self.mu_, self.coef_) if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def k_fold_cv_ridge(X, y, lambdas, k=5, seed=0):
    """
    K-fold cross-validation for Ridge regression.

    Returns
    -------
    mean_mse : np.ndarray, shape (len(lambdas),)
        Mean held-out MSE for each lambda in the grid.
    best_lambda : float
        Lambda achieving lowest mean CV MSE.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    mse_grid = np.zeros((len(lambdas), k))
    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(idx, test_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        for li, lam in enumerate(lambdas):
            model = RidgeCF(lam=lam).fit(X_train, y_train)
            preds = model.predict(X_test)
            mse_grid[li, fi] = np.mean((y_test - preds) ** 2)

    mean_mse = mse_grid.mean(axis=1)
    best_lambda = lambdas[np.argmin(mean_mse)]
    return mean_mse, best_lambda


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n, p = 200, 15
    X = rng.normal(size=(n, p))
    true_beta = rng.normal(size=p)
    y = X @ true_beta + rng.normal(scale=2.0, size=n)

    lambdas = np.logspace(-2, 3, 30)
    mean_mse, best_lambda = k_fold_cv_ridge(X, y, lambdas, k=5)
    print(f"Best lambda by 5-fold CV: {best_lambda:.4f}")

    model = RidgeCF(lam=best_lambda).fit(X, y)
    pred_err = np.mean((y - model.predict(X)) ** 2)
    print(f"Training MSE at best lambda: {pred_err:.4f}")
