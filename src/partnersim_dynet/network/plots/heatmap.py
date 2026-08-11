"""Degree heatmap evolution plot."""

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

SEX_DISPLAY_ORDER: tuple[str, ...] = ("Male", "Female")
ORIENTATION_DISPLAY_ORDER: tuple[str, ...] = (
    "Opposite-sex",
    "Same-sex",
    "Bisexual",
)

BURN_IN = 50
CENSORING = 50
T_MAX = 1875
N_BLOCKS = 5

analysis_start = 1 + BURN_IN
analysis_end = T_MAX - CENSORING
block_width = (analysis_end - analysis_start + 1) // N_BLOCKS

heatmap_windows = [
    (
        analysis_start + i * block_width,
        analysis_start + (i + 1) * block_width - 1,
    )
    for i in range(N_BLOCKS)
]

# Age groups to display — exclude "Unknown"
_DISPLAY_AGE_GROUPS = [g for g in AGE_GROUPS if g != "Unknown"]


def plot_degree_heatmap_evolution(
    degree_by_demographic: pd.DataFrame,
    windows: list[tuple[int, int]],
    output_dir: str,
    filename_stem: str = "degree_heatmap_evolution",
    vmax_percentile: float = 95.0,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Plot mean degree per (AgeGroup × Sex) across time windows.

    Each column is a time window; each row is an orientation. Cells show
    mean degree as a heatmap with the mean agent count (N) annotated.

    Parameters
    ----------
    degree_by_demographic : DataFrame
        Output of ``degree_by_demographic_over_time``. Must contain:
        t, AgentSex, AgentOrientation, AgentAgeGroup, MeanDegree, N.
    windows : list of (int, int)
        Each tuple is (window_start, window_end) inclusive.
    output_dir : str
    filename_stem : str
    vmax_percentile : float
        Upper colour-scale percentile, computed per orientation row so
        sparse orientations are not washed out by the opposite-sex range.
    formats : OutputFormats
    """
    required = {"t", "AgentSex", "AgentOrientation", "AgentAgeGroup", "MeanDegree", "N"}
    missing = required - set(degree_by_demographic.columns)
    if missing:
        raise KeyError(f"degree_by_demographic missing columns: {sorted(missing)}")
    if not windows:
        raise ValueError("windows must be a non-empty list")

    # Restrict vmax calculation to steady-state window data only
    all_t = {t for ws, we in windows for t in range(ws, we + 1)}
    analysis_data = degree_by_demographic[degree_by_demographic["t"].isin(all_t)]

    ori_vmax: dict[str, float] = {}
    for ori in ORIENTATION_DISPLAY_ORDER:
        vals = analysis_data.loc[analysis_data["AgentOrientation"] == ori, "MeanDegree"].dropna()
        if vals.empty or vals.max() == 0:
            ori_vmax[ori] = 1.0
        else:
            q = float(np.percentile(vals, vmax_percentile))
            ori_vmax[ori] = q if q > 0 else float(vals.max())

    n_rows = len(ORIENTATION_DISPLAY_ORDER)
    n_cols = len(windows)
    cmap = "Blues"

    with publication_style():
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.4 * n_cols, 2.2 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )

        for row_idx, ori in enumerate(ORIENTATION_DISPLAY_ORDER):
            vmax = ori_vmax[ori]
            ori_data = degree_by_demographic[degree_by_demographic["AgentOrientation"] == ori]

            for col_idx, (w_start, w_end) in enumerate(windows):
                ax = axes[row_idx, col_idx]

                cell_data = ori_data[(ori_data["t"] >= w_start) & (ori_data["t"] <= w_end)]

                if cell_data.empty:
                    mean_pivot = pd.DataFrame(
                        np.nan,
                        index=_DISPLAY_AGE_GROUPS,
                        columns=list(SEX_DISPLAY_ORDER),
                        dtype=float,
                    )
                    n_pivot = mean_pivot.copy()
                else:
                    # Mean degree averaged across all timesteps in the window
                    mean_pivot = cell_data.pivot_table(
                        index="AgentAgeGroup",
                        columns="AgentSex",
                        values="MeanDegree",
                        aggfunc="mean",
                    ).reindex(
                        index=_DISPLAY_AGE_GROUPS,
                        columns=list(SEX_DISPLAY_ORDER),
                    )

                    # Mean N per timestep: sum N across all rows in the window,
                    # then divide by the number of distinct timesteps.
                    # Each row is one (t, sex, orientation, age_group) record,
                    # so summing then dividing gives the mean per-timestep count.
                    n_timesteps = max(cell_data["t"].nunique(), 1)
                    n_pivot = (
                        cell_data.pivot_table(
                            index="AgentAgeGroup",
                            columns="AgentSex",
                            values="N",
                            aggfunc="sum",
                        )
                        / n_timesteps
                    ).reindex(
                        index=_DISPLAY_AGE_GROUPS,
                        columns=list(SEX_DISPLAY_ORDER),
                    )

                show_cbar = col_idx == n_cols - 1
                sns.heatmap(
                    mean_pivot,
                    ax=ax,
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    cbar=show_cbar,
                    cbar_kws=({"shrink": 0.8, "label": "Mean degree"} if show_cbar else {}),
                    xticklabels=(row_idx == n_rows - 1),
                    yticklabels=(col_idx == 0),
                    linewidths=0.3,
                    linecolor="white",
                    annot=False,  # we annotate manually below for full control
                )

                # N annotations — show mean agent count per cell
                for r_i, age in enumerate(_DISPLAY_AGE_GROUPS):
                    for c_i, sex in enumerate(SEX_DISPLAY_ORDER):
                        try:
                            n_val = n_pivot.at[age, sex]
                            mean_val = mean_pivot.at[age, sex]
                        except KeyError:
                            continue
                        if pd.isna(n_val):
                            continue
                        norm = float(mean_val) / vmax if (pd.notna(mean_val) and vmax > 0) else 0.0
                        ink = "white" if norm > 0.45 else "#1a1a2e"
                        ax.text(
                            c_i + 0.5,
                            r_i + 0.5,
                            f"N={int(round(n_val))}",
                            ha="center",
                            va="center",
                            fontsize=5.5,
                            color=ink,
                        )

                if row_idx == 0:
                    ax.set_title(f"[{w_start}–{w_end}]", fontsize=8, fontweight="bold")
                ax.set_ylabel(ori if col_idx == 0 else "", fontsize=9, fontweight="bold")
                ax.set_xlabel("Sex" if row_idx == n_rows - 1 else "", fontsize=8)

        fig.suptitle(
            "Evolution of mean degree by age group × sex × orientation",
            fontsize=12,
            fontweight="bold",
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


ORIENTATION_DISPLAY_ORDER: tuple[str, ...] = (
    "Opposite-sex",
    "Same-sex",
    "Bisexual",
)
SEX_DISPLAY_ORDER: tuple[str, ...] = ("Male", "Female")

_DISPLAY_AGE_GROUPS = [g for g in AGE_GROUPS if g != "Unknown"]

BURN_IN = 50
CENSORING = 50
T_MAX = 1875
N_BLOCKS = 5  # kept for heatmap_windows compatibility

analysis_start = 1 + BURN_IN
analysis_end = T_MAX - CENSORING
block_width = (analysis_end - analysis_start + 1) // N_BLOCKS

temporal_windows = [
    (51, 500),
    (501, 1000),
    (1001, 1500),
    (1501, 1825),
]


def _window_label(w_start: int, w_end: int) -> str:
    return f"{w_start}–{w_end}"


def plot_degree_temporal_heatmaps(
    degree_by_demographic: pd.DataFrame,
    output_dir: str,
    windows: list[tuple[int, int]] | None = None,
    filename_stem: str = "degree_temporal_heatmap",
    vmax_percentile: float = 95.0,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Three figures (one per orientation) showing how mean degree evolves.

    Each figure:
      - Left panel:  Male
      - Right panel: Female
      - Rows:        time windows (top = early, bottom = late)
      - Columns:     age groups
      - Cell colour: mean degree in that window

    Parameters
    ----------
    degree_by_demographic : DataFrame
        Output of ``degree_by_demographic_over_time``.
        Required columns: t, AgentSex, AgentOrientation, AgentAgeGroup,
        MeanDegree, N.
    output_dir : str
    windows : list of (int, int) or None
        Time windows. Defaults to ``temporal_windows`` (10 evenly-spaced
        blocks across the steady-state period).
    filename_stem : str
    vmax_percentile : float
        Upper colour-scale percentile, computed per orientation.
    formats : OutputFormats
    """
    required = {"t", "AgentSex", "AgentOrientation", "AgentAgeGroup", "MeanDegree", "N"}
    missing = required - set(degree_by_demographic.columns)
    if missing:
        raise KeyError(f"degree_by_demographic missing columns: {sorted(missing)}")

    if windows is None:
        windows = temporal_windows

    window_labels = [_window_label(ws, we) for ws, we in windows]
    n_windows = len(windows)
    n_ages = len(_DISPLAY_AGE_GROUPS)

    written: list[str] = []

    for ori in ORIENTATION_DISPLAY_ORDER:
        ori_data = degree_by_demographic[degree_by_demographic["AgentOrientation"] == ori]

        # vmax: 95th percentile of MeanDegree within the window range
        # Use t between analysis_start and analysis_end rather than
        # enumerating every individual timestep (much faster).
        ss_vals = ori_data.loc[
            (ori_data["t"] >= analysis_start) & (ori_data["t"] <= analysis_end),
            "MeanDegree",
        ].dropna()
        if ss_vals.empty or ss_vals.max() == 0:
            vmax = 1.0
        else:
            vmax = float(np.percentile(ss_vals, vmax_percentile))
            vmax = vmax if vmax > 0 else float(ss_vals.max())

        # Build one (n_windows × n_ages) matrix per sex.
        # Each row in degree_by_demographic is already one (t, sex, ori, age)
        # record with MeanDegree = mean degree at that timestep and N = agent
        # count at that timestep. Within a window we average MeanDegree across
        # timesteps (mean of means = correct since N is ~stable), and take the
        # mean N as the typical agent count in that window.
        matrices: dict[str, np.ndarray] = {}
        n_matrices: dict[str, np.ndarray] = {}

        for sex in SEX_DISPLAY_ORDER:
            mat = np.full((n_windows, n_ages), np.nan)
            n_mat = np.full((n_windows, n_ages), np.nan)

            sex_data = ori_data[ori_data["AgentSex"] == sex]

            for row_i, (ws, we) in enumerate(windows):
                window_data = sex_data[(sex_data["t"] >= ws) & (sex_data["t"] <= we)]
                if window_data.empty:
                    continue
                for col_i, age in enumerate(_DISPLAY_AGE_GROUPS):
                    age_rows = window_data[window_data["AgentAgeGroup"] == age]
                    if age_rows.empty:
                        continue
                    # mean of per-timestep means = mean degree in window
                    mat[row_i, col_i] = age_rows["MeanDegree"].mean()
                    # mean of per-timestep N = typical agent count in window
                    n_mat[row_i, col_i] = age_rows["N"].mean()

            matrices[sex] = mat
            n_matrices[sex] = n_mat

        # ── Figure ────────────────────────────────────────────────────────
        cmap = "Blues"

        with publication_style():
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(max(8.0, 1.1 * n_ages), max(4.0, 0.55 * n_windows + 1.5)),
                sharey=True,
            )
            fig.patch.set_facecolor("white")

            im = None
            for ax, sex in zip(axes, SEX_DISPLAY_ORDER, strict=False):
                mat = matrices[sex]
                n_mat = n_matrices[sex]

                im = ax.imshow(
                    mat,
                    aspect="auto",
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    interpolation="nearest",
                )

                # Cell grid lines
                ax.set_xticks(np.arange(n_ages + 1) - 0.5, minor=True)
                ax.set_yticks(np.arange(n_windows + 1) - 0.5, minor=True)
                ax.grid(which="minor", color="white", linewidth=0.8)
                ax.tick_params(which="minor", length=0)

                ax.set_xticks(np.arange(n_ages))
                ax.set_xticklabels(
                    _DISPLAY_AGE_GROUPS,
                    rotation=35,
                    ha="right",
                    fontsize=8,
                )
                ax.set_xlabel("Age group", fontsize=9)
                ax.set_title(sex, fontsize=10, fontweight="bold", pad=6)

                if ax is axes[0]:
                    ax.set_yticks(np.arange(n_windows))
                    ax.set_yticklabels(window_labels, fontsize=7.5)
                    ax.set_ylabel("Simulation window", fontsize=9)
                else:
                    ax.tick_params(axis="y", left=False)

                # Cell annotations: mean degree + mean N
                for row_i in range(n_windows):
                    for col_i in range(n_ages):
                        val = mat[row_i, col_i]
                        n = n_mat[row_i, col_i]
                        if np.isnan(val):
                            continue
                        norm = val / vmax if vmax > 0 else 0
                        ink = "white" if norm > 0.55 else "#1a1a1a"
                        ax.text(
                            col_i,
                            row_i,
                            f"{val:.2f}\nN={int(round(n))}" if not np.isnan(n) else f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=ink,
                        )

            # Shared colourbar on the right
            if im is not None:
                cbar = fig.colorbar(
                    im,
                    ax=axes,
                    orientation="vertical",
                    fraction=0.03,
                    pad=0.02,
                    shrink=0.85,
                )
                cbar.set_label("Mean degree", fontsize=9)
                cbar.ax.tick_params(labelsize=8)

            fig.suptitle(
                f"{ori}  —  mean degree by age group over time",
                fontsize=11,
                fontweight="bold",
                y=1.01,
            )
            fig.subplots_adjust(
                left=0.15,
                right=0.88,
                top=0.92,
                bottom=0.18,
                wspace=0.06,
            )

            safe_stem = ori.lower().replace("-", "_").replace(" ", "_")
            output_base = os.path.join(output_dir, f"{filename_stem}_{safe_stem}")
            written += save_figure(fig, output_base, formats)
            plt.close(fig)

    return written
