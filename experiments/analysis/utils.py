import numpy as np
from matplotlib import pyplot as plt


def plot_dolan_more(df, ax=None, palette=None, linestyle="-", lw=2):
    """
    Plots the Dolan-More performance profile for a DataFrame of metrics.
    Assumes 'Lower is Better' (e.g., NLL, Error).
    """
    if ax is None:
        ax = plt.gca()

    # 1. Handle potential non-positive values (Standard Dolan-More requires > 0)
    # If values are <= 0, we shift all data by |min| + epsilon to make them positive
    min_val = df.min().min()
    if min_val <= 0:
        shift = abs(min_val) + 1e-8
        df_shifted = df + shift
    else:
        df_shifted = df.copy()

    # 2. Calculate Ratios: cost(model) / min_cost_across_models
    # shape: (n_samples, n_models)
    min_per_row = df_shifted.min(axis=1)
    ratios = df_shifted.div(min_per_row, axis=0)

    # 3. Plot Step Functions for each model
    # The y-axis is the fraction of problems solved within factor tau (x-axis)
    n_problems = len(df)

    for model in df.columns:
        # Sort ratios to calculate CDF
        sorted_ratios = np.sort(ratios[model])

        # Y-axis: Cumulative probability (1/N, 2/N, ..., 1.0)
        yvals = np.arange(1, n_problems + 1) / n_problems

        # Add the starting point (1.0, 0.0) to make the step plot start from y=0
        # Dolan-More profiles technically start at x=1 (best model)
        xs = np.concatenate(([1.0], sorted_ratios))
        ys = np.concatenate(([0.0], yvals))

        color = palette[model] if palette else None

        # Step plot
        ax.step(xs, ys, where="post", label=model,
                color=color, linestyle=linestyle, linewidth=lw)

    # Formatting
    ax.set_xlabel(r"Performance Ratio $\tau$ (Best Possible = 1.0)")
    ax.set_ylabel(r"Proportion of problems solved ($P(r_{p,s} \leq \tau)$)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Optional: Log scale is often better if some models perform very poorly
    # ax.set_xscale('log')

    return ax