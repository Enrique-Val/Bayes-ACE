import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_dolan_more(df, ax=None, palette=None, linestyle="-", lw=2, metric=None, title=None, x_lim = 10):
    """
    Plots the Dolan-More performance profile for a DataFrame of metrics
    with scalar x-ticks and specific quantile markers.
    """
    if ax is None:
        ax = plt.gca()

    # 1. Handle potential non-positive values
    min_val = df.min().min()
    if min_val <= 0:
        shift = abs(min_val) + 1e-8
        df_shifted = df + shift
    else:
        df_shifted = df.copy()

    # 2. Calculate Ratios
    min_per_row = df_shifted.min(axis=1)
    ratios = df_shifted.div(min_per_row, axis=0)

    # 3. Plot Step Functions
    n_problems = len(df)

    # Define the specific Y-levels for markers
    marker_levels = [0.2, 0.4, 0.6, 0.8]

    for model in df.columns:
        sorted_ratios = np.sort(ratios[model])

        # Y-axis: Cumulative probability
        yvals = np.arange(1, n_problems + 1) / n_problems

        # Construct coordinates for step plot
        xs = np.concatenate(([1.0], sorted_ratios))
        ys = np.concatenate(([0.0], yvals))

        color = palette[model] if palette else None

        # Main Step plot
        line, = ax.step(xs, ys, where="post", label=model,
                        color=color, linestyle=linestyle, linewidth=lw)

        # --- Add Dot Markers at 0.2, 0.4, 0.6, 0.8 ---
        # We find the X-value where the curve crosses these Y-thresholds.
        # Since it's a step function (post), the 'jump' happens at specific xs.
        marker_xs = []
        marker_ys_found = []

        for level in marker_levels:
            # Find first index where we reach or exceed the level
            if np.max(ys) >= level:
                idx = np.argmax(ys >= level)
                marker_xs.append(xs[idx])
                marker_ys_found.append(level)

        if marker_xs:
            # Use the same color as the line, but markers only (no line connection)
            # zorder=3 ensures markers sit on top of the grid and lines
            ax.plot(marker_xs, marker_ys_found, 'o', color=line.get_color(),
                    markersize=6, zorder=3)

    # Formatting
    if title is not None:
        ax.set_title(title, fontsize=12)

    if metric is None:
        ax.set_xlabel(r"Performance ratio")
    else:
        ax.set_xlabel(r"Performance ratio (" + metric + ")")

    ax.set_ylabel(r"Proportion of problems")
    ax.set_ylim(0, 1)
    # Adjust xlim as needed, 1 to 10 is standard for tight profiles
    ax.set_xlim(1, x_lim)

    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Log scale
    ax.set_xscale('log')

    # --- 1. Force Non-Scientific Notation on X-Axis ---
    # 1. Define a simple scalar formatter: just returns the number as a string (e.g. "2", "10")
    # '{:g}' automatically handles integers (2.0 -> "2") and floats (2.5 -> "2.5") cleanly.
    scalar_fmt = ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x))

    # 2. Apply to Major Ticks (1, 10, 100...)
    ax.xaxis.set_major_formatter(scalar_fmt)

    # 3. Apply to Minor Ticks (2, 3, 4, 5...)
    # This converts them to scalar strings. It relies on the default LogLocator
    # to decide *which* ticks exist, so it won't force ticks where there isn't space.
    ax.xaxis.set_minor_formatter(scalar_fmt)
    #ax.xaxis.set_minor_formatter(formatter)

    # Optional: If you want standard ticks (1, 2, 5, 10) instead of log spacing
    # ax.set_xticks([1, 2, 5, 10])
    # ax.set_xticklabels(["1", "2", "5", "10"])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax