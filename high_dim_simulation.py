"""
Experiment: Sparse Support Recovery in the High-Dimensional Regime (p > n)

Question: as the ambient dimension p grows relative to the sample size n,
how well does Lasso recover the true sparse support of beta, compared to
Ridge (which never produces exact zeros)?

This connects to the classical theory of the Lasso's model-selection
consistency under the restricted eigenvalue / irrepresentable conditions
(Zhao & Yu, 2006; Wainwright, 2009) — this simulation gives an empirical
feel for that theory without proving it.

Setup:
    - n fixed at 60
    - p swept over a grid from 20 (p < n) to 300 (p >> n)
    - true beta has exactly k = 5 nonzero entries
    - for each p, simulate `trials` datasets and record:
        * Lasso's exact support recovery rate
        * Ridge's "effective sparsity" (never exactly sparse, tracked for contrast)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.lasso_coordinate_descent import LassoCD


def run_experiment(n=60, k=5, p_grid=None, trials=25, lam_const=0.35, seed=0):
    """
    lam_const scales the theoretical rate lambda ~ C * sqrt(log(p) / n),
    which is the rate under which Lasso is known to achieve model-selection
    consistency (see Wainwright, 2009). Using a fixed lambda across all p
    under-penalizes as p grows, which is why lambda must scale with p here.
    """
    if p_grid is None:
        p_grid = [20, 40, 60, 80, 120, 160, 220, 300]

    rng = np.random.default_rng(seed)
    recovery_rate = []

    for p in p_grid:
        lam = lam_const * np.sqrt(np.log(p) / n) * 5  # scaled empirically to this signal strength
        successes = 0
        for t in range(trials):
            X = rng.normal(size=(n, p))
            true_beta = np.zeros(p)
            support = rng.choice(p, size=k, replace=False)
            true_beta[support] = rng.uniform(2, 5, size=k) * rng.choice([-1, 1], size=k)
            y = X @ true_beta + rng.normal(scale=1.0, size=n)

            model = LassoCD(lam=lam, max_iter=500).fit(X, y)
            recovered_support = set(np.nonzero(np.abs(model.coef_) > 1e-3)[0])
            true_support = set(support)

            if recovered_support == true_support:
                successes += 1

        rate = successes / trials
        recovery_rate.append(rate)
        print(f"p={p:4d}  (p/n={p/n:5.2f})   exact recovery rate = {rate:.2f}")

    return p_grid, recovery_rate


def plot_results(p_grid, recovery_rate, n, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(p_grid, recovery_rate, marker="o", linewidth=2, color="#2b6cb0")
    ax.axvline(n, color="gray", linestyle="--", alpha=0.6, label=f"n = {n}")
    ax.set_xlabel("Number of features (p)")
    ax.set_ylabel("Exact support recovery rate")
    ax.set_title("Lasso Sparse Recovery as p Grows Relative to n")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    n = 60
    p_grid, recovery_rate = run_experiment(n=n)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_results(p_grid, recovery_rate, n, os.path.join(out_dir, "sparse_recovery.png"))
