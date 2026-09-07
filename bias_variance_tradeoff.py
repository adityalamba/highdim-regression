"""
Experiment: Empirical Bias-Variance Decomposition for Ridge Regression

For a fixed true beta and design distribution, we repeatedly resample
(X, y), fit Ridge at each lambda, and empirically decompose the expected
prediction error at a fixed test point x0 into:

    E[(y0_hat - f(x0))^2] = Bias(y0_hat)^2 + Var(y0_hat) + irreducible noise

This is the textbook bias-variance identity (Hastie, Tibshirani & Friedman,
ESL Ch. 2 & 3), verified empirically via Monte Carlo rather than derived
analytically.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.ridge_closed_form import RidgeCF


def run_experiment(n=50, p=30, noise_sd=1.0, lambdas=None, n_sims=300, seed=0):
    if lambdas is None:
        lambdas = np.logspace(-2, 2.5, 20)

    rng = np.random.default_rng(seed)
    true_beta = rng.normal(scale=1.0, size=p)
    x0 = rng.normal(size=p)          # fixed test point
    f_x0 = x0 @ true_beta            # true noiseless target at x0

    bias_sq = np.zeros(len(lambdas))
    variance = np.zeros(len(lambdas))
    total_err = np.zeros(len(lambdas))

    for li, lam in enumerate(lambdas):
        preds = np.zeros(n_sims)
        for s in range(n_sims):
            X = rng.normal(size=(n, p))
            y = X @ true_beta + rng.normal(scale=noise_sd, size=n)
            model = RidgeCF(lam=lam).fit(X, y)
            preds[s] = model.predict(x0.reshape(1, -1))[0]

        mean_pred = preds.mean()
        bias_sq[li] = (mean_pred - f_x0) ** 2
        variance[li] = preds.var()
        total_err[li] = bias_sq[li] + variance[li] + noise_sd ** 2

    return lambdas, bias_sq, variance, total_err


def plot_results(lambdas, bias_sq, variance, total_err, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas, bias_sq, label="Bias$^2$", color="#c05621", linewidth=2)
    ax.plot(lambdas, variance, label="Variance", color="#2b6cb0", linewidth=2)
    ax.plot(lambdas, total_err, label="Total expected error", color="#2d3748",
            linewidth=2, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"Regularization strength $\lambda$ (log scale)")
    ax.set_ylabel("Error")
    ax.set_title("Bias-Variance Tradeoff for Ridge Regression (Monte Carlo)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    lambdas, bias_sq, variance, total_err = run_experiment()
    best_idx = np.argmin(total_err)
    print(f"Lambda minimizing total expected error: {lambdas[best_idx]:.3f}")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_results(lambdas, bias_sq, variance, total_err,
                 os.path.join(out_dir, "bias_variance_tradeoff.png"))
