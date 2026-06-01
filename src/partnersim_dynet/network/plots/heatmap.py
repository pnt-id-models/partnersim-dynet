"""Degree heatmap evolution plot.

A grid plot showing how mean degree per (AgeGroup, Sex) varies across
demographic orientations and across time. Rows are orientations (3),
columns are user-chosen snapshot timesteps. Each cell is a small
heatmap of age groups × sex with cell color = mean degree.

Data backbone: takes the output of ``degree_by_demographic_over_time``
filtered to ``snapshot_times``. No graph building happens here — that's
the metrics module's job.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from partnersim_dynet.config import AGE_GROUPS
from partnersim_dynet.network.plots.style import (
    OutputFormats,
    publication_style,
    save_figure,
)

# Display ordering for the heatmap axes
SEX_DISPLAY_ORDER: tuple[str, ...] = ("Male", "Female")
ORIENTATION_DISPLAY_ORDER: tuple[str, ...] = (
    "Opposite-sex",
    "Same-sex",
    "Bisexual",
)


def plot_degree_heatmap_evolution(
    degree_by_demographic: pd.DataFrame,
    snapshot_times: list[int],
    output_dir: str,
    filename_stem: str = "degree_heatmap_evolution",
    vmax_percentile: float = 95.0,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Plot mean degree per (AgeGroup × Sex) across snapshot timesteps.

    For each snapshot timestep and each orientation, draw a small
    heatmap of AgeGroup (rows) × Sex (columns) with cell color = mean
    degree. The full figure is a grid: 3 orientation rows × N snapshot
    columns.

    Parameters
    ----------
    degree_by_demographic : DataFrame
        Output of ``degree_by_demographic_over_time``. Must contain
        columns: t, AgentSex, AgentOrientation, AgentAgeGroup, MeanDegree.
    snapshot_times : list of int
        Timesteps to render columns for. Order is preserved.
    output_dir : str
        Where to write the figure.
    filename_stem : str
        Output filename without extension.
    vmax_percentile : float
        Percentile (0-100) of the MeanDegree distribution to use as the
        upper bound of the color scale. 95 by default — caps the
        extreme tail so the typical range stays distinguishable.
    formats : OutputFormats
        Which file formats to write.

    Returns
    -------
    list of str
        Paths of files actually written.

    Raises
    ------
    ValueError
        If ``snapshot_times`` is empty or any t is missing from the
        DataFrame.
    KeyError
        If the DataFrame is missing required columns.
    """
    required = {"t", "AgentSex", "AgentOrientation", "AgentAgeGroup", "MeanDegree"}
    missing = required - set(degree_by_demographic.columns)
    if missing:
        raise KeyError(f"degree_by_demographic missing columns: {sorted(missing)}")

    if not snapshot_times:
        raise ValueError("snapshot_times must be a non-empty list")

    available_t = set(degree_by_demographic["t"].unique())
    missing_t = [t for t in snapshot_times if t not in available_t]
    if missing_t:
        raise ValueError(
            f"snapshot_times not in DataFrame: {missing_t} "
            f"(available range: {min(available_t)}..{max(available_t)})"
        )

    # Color-scale upper bound from the overall distribution
    vmax = float(np.percentile(degree_by_demographic["MeanDegree"], vmax_percentile))
    if vmax <= 0:
        vmax = 1.0  # avoid degenerate all-zero color scale

    n_rows = len(ORIENTATION_DISPLAY_ORDER)
    n_cols = len(snapshot_times)

    with publication_style():
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.2 * n_cols, 6),
            constrained_layout=True,
        )

        # Normalise axes indexing: always treat as 2D
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for row_idx, ori in enumerate(ORIENTATION_DISPLAY_ORDER):
            for col_idx, t in enumerate(snapshot_times):
                ax = axes[row_idx, col_idx]

                # Pivot the long-form data for this (orientation, t) cell
                cell_data = degree_by_demographic[
                    (degree_by_demographic["AgentOrientation"] == ori)
                    & (degree_by_demographic["t"] == t)
                ]

                if cell_data.empty:
                    pivot = pd.DataFrame(
                        index=list(AGE_GROUPS),
                        columns=list(SEX_DISPLAY_ORDER),
                    ).astype(float)
                else:
                    pivot = cell_data.pivot_table(
                        index="AgentAgeGroup",
                        columns="AgentSex",
                        values="MeanDegree",
                        aggfunc="mean",
                    ).reindex(
                        index=list(AGE_GROUPS),
                        columns=list(SEX_DISPLAY_ORDER),
                    )

                # Last column on each row gets the colorbar
                show_cbar = col_idx == n_cols - 1

                sns.heatmap(
                    pivot,
                    ax=ax,
                    cmap=sns.color_palette("mako_r", as_cmap=True),
                    vmin=0,
                    vmax=vmax,
                    cbar=show_cbar,
                    xticklabels=(row_idx == n_rows - 1),  # only bottom row
                    yticklabels=(col_idx == 0),  # only leftmost col
                )

                if row_idx == 0:
                    ax.set_title(f"t = {t}", fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(ori, fontsize=10, fontweight="bold")
                else:
                    ax.set_ylabel("")

                ax.set_xlabel("Sex" if row_idx == n_rows - 1 else "")

        fig.suptitle(
            "Evolution of mean degree by age group × sex × orientation",
            fontsize=13,
            fontweight="bold",
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written
