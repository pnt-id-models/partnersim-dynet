"""Network metric plots for partnership simulation output.

Structural snapshots (distribution plots)

- ``plot_degree_summary`` : mean, median, and max degree time series (stacked panels)


Calculations and definitions
-----------------------------
Degree k:
    Number of active partnerships an agent holds simultaneously.
    Computed per-timestep on the instantaneous partnership graph.

Average degree (time series):
    mean(k_i) over all agents i with at least one active partnership at t.
    Equivalently: 2 * |edges| / |nodes with degree >= 1|.

Maximum degree (time series):
    max(k_i) over all agents at t. Tracks the most-concurrent agent.

Degree distribution P(k) [structural]:
    Probability density: P(k) = (nodes with degree k) / (total nodes * bin_width).
    Plotted on log-log axes. A straight line suggests power-law scaling.

"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from partnersim_dynet.network.plots.style import (
    PALETTE,
    OutputFormats,
    publication_style,
    save_figure,
)

# Constants for burn-in and censoring steps. These are used to define the steady-state window for time-series plots.

BURN_IN_STEPS = 50
CENSORING_STEPS = 50


@dataclass(frozen=True)
class TimeseriesSpec:
    metric_column: str
    ylabel: str
    color: str
    integer_y: bool = False
    filename_stem: str = ""

    def __post_init__(self) -> None:
        if not self.filename_stem:
            object.__setattr__(self, "filename_stem", f"{self.metric_column}_over_time")


# Shared helpers for time-series plots. Steady-state window is defined as the range of timesteps after discarding
# the first BURN_IN_STEPS and last CENSORING_STEPS. If the simulation is too short to have a steady-state window, raises ValueError.


def _steady_state_mask(
    metrics: pd.DataFrame,
) -> tuple[pd.Series, int, int, int, int]:
    t_min = int(metrics["t"].min())
    t_max = int(metrics["t"].max())
    t_start = t_min + BURN_IN_STEPS
    t_end = t_max - CENSORING_STEPS
    if t_start >= t_end:
        raise ValueError(
            f"Simulation too short ({t_max - t_min + 1} steps) to have a "
            f"steady-state window after discarding {BURN_IN_STEPS} burn-in "
            f"and {CENSORING_STEPS} censoring steps."
        )
    mask = (metrics["t"] >= t_start) & (metrics["t"] <= t_end)
    return mask, t_min, t_max, t_start, t_end


# Generate shaded burn-in and censoring zones on a time-series plot, with vertical dashed lines and labels.
def _shade_zones(ax, t_min, t_start, t_end, t_max) -> None:
    ax.axvspan(t_min, t_start, color="#6F92FC", alpha=0.08, zorder=0)
    ax.axvline(t_start, color="#2B4598", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        t_min + BURN_IN_STEPS / 2,
        1.02,
        "burn-in",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#2D4250",
        fontstyle="italic",
    )
    ax.axvspan(t_end, t_max, color="#CC4125", alpha=0.08, zorder=0)
    ax.axvline(t_end, color="#CC4125", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(
        t_end + CENSORING_STEPS / 2,
        1.02,
        "censoring",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#A32D18",
        fontstyle="italic",
    )


# Statistics box for steady-state window, placed in the top-right corner of the plot. Displays mean and max values
def _stats_box(ax, window: pd.Series, extra: str = "") -> None:
    txt = f"Steady-state\nmean={window.mean():.3g}   max={window.max():.3g}"
    if extra:
        txt += f"\n{extra}"
    ax.text(
        0.99,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=PALETTE.annotation,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
    )


# Time-series plotting core functions for a single metric. Returns list of written file paths.


def _plot_timeseries_core(
    metrics: pd.DataFrame,
    col: str,
    ylabel: str,
    color: str,
    output_dir: str,
    filename_stem: str,
    formats: OutputFormats,
    integer_y: bool = False,
    ylim_bottom: float | None = None,
    low_value_note: str | None = None,
) -> list[str]:
    """Core time-series renderer.

    Parameters
    ----------
    ylim_bottom : float or None
        If set, overrides the y-axis lower bound (e.g. 5 for max_degree
        to remove wasted whitespace at the bottom).
    trend_line : bool
        If True, overlay a linear trend line on the steady-state window (not used).
    low_value_note : str or None
        If set, adds a footnote below the stats box (e.g. for transitivity
        explaining that near-zero values are expected in partnership networks).
    """
    if col not in metrics.columns:
        raise KeyError(f"{col!r} not in metrics DataFrame")

    # Use mask to select the steady-state window, and also define burn-in and censoring masks for faded plotting.
    mask, t_min, t_max, t_start, t_end = _steady_state_mask(metrics)
    burn_mask = metrics["t"] < t_start
    censor_mask = metrics["t"] > t_end

    with publication_style():
        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor("white")

        t_ss = metrics.loc[mask, "t"].values
        v_ss = metrics.loc[mask, col].values

        # Trusted window — full opacity
        ax.plot(t_ss, v_ss, color=color, lw=1.4)
        ax.fill_between(t_ss, 0, v_ss, color=color, alpha=0.12)

        # Faded burn-in / censoring zones
        for m in (burn_mask, censor_mask):
            ax.plot(metrics.loc[m, "t"], metrics.loc[m, col], color=color, lw=1.4, alpha=0.45)

        _shade_zones(ax, t_min, t_start, t_end, t_max)

        ax.set_xlim(t_min, t_max)
        ax.set_xlabel("Timesteps (days)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, color="grey")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        if integer_y:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)

        window = metrics.loc[mask, col]
        if len(window) > 0:
            _stats_box(ax, window, extra=low_value_note or "")

        fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)
        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# Plotting functions for specific metrics, using the core class above. Returns list of written file paths.
def plot_timeseries(
    metrics: pd.DataFrame,
    spec: TimeseriesSpec,
    output_dir: str,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    return _plot_timeseries_core(
        metrics,
        spec.metric_column,
        spec.ylabel,
        spec.color,
        output_dir,
        spec.filename_stem,
        formats,
        integer_y=spec.integer_y,
    )


# Plot degree summary (mean, median, max) time series for a single scenario. Returns list of written file paths.
# We check for the presence of concurrent-only and monogamous-only columns, and overlay them if present.
# The steady-state window is shaded, and the mean/max values are annotated in the top-right corner of each panel.
def plot_degree_summary(
    metrics: pd.DataFrame,
    output_dir: str,
    scenario_label: str = "",
    filename_stem: str = "degree_summary",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Mean, median, and max degree time series for a single scenario.

    Three stacked panels sharing a timestep x-axis. If the metrics
    DataFrame includes concurrency-only columns (mean_degree_concurrent,
    median_degree_concurrent) and monogamous-only columns
    (mean_degree_monogamous, median_degree_monogamous) with at least one
    non-null value. These are overlaid on the mean/median panels — red
    dashed for concurrent agents (deg >= 2) and blue dashed for
    monogamous agents (deg == 1). Scenarios without one of those groups
    simply omit that overlay.
    """
    cols = ["avg_degree", "median_degree", "max_degree"]
    missing = [c for c in cols if c not in metrics.columns]
    if missing:
        raise KeyError(f"metrics missing columns: {missing}")
    # has_concurrent = (
    #     "mean_degree_concurrent" in metrics.columns
    #     and metrics["mean_degree_concurrent"].notna().any()
    # )
    has_monogamous = (
        "mean_degree_monogamous" in metrics.columns
        and metrics["mean_degree_monogamous"].notna().any()
    )

    mask, t_min, t_max, t_start, t_end = _steady_state_mask(metrics)
    burn_mask = metrics["t"] < t_start
    censor_mask = metrics["t"] > t_end

    titles = ["Mean degree", "Median degree", "Max degree"]
    color = PALETTE.timeseries_line
    concurrent_color = "#C0392B"
    monogamous_color = "#1053A5"

    with publication_style():
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        fig.patch.set_facecolor("white")
        if scenario_label:
            fig.suptitle(scenario_label, fontsize=11, fontweight="bold", y=0.995)

        for ax, col, ylabel in zip(axes, cols, titles, strict=False):
            t_ss = metrics.loc[mask, "t"].values
            v_ss = metrics.loc[mask, col].values
            show_max = col == "max_degree"

            ax.plot(t_ss, v_ss, color=color, lw=1.4, label="All agents")
            ax.fill_between(t_ss, 0, v_ss, color=color, alpha=0.12)

            for m in (burn_mask, censor_mask):
                ax.plot(metrics.loc[m, "t"], metrics.loc[m, col], color=color, lw=1.4, alpha=0.45)

            concurrent_col = None
            monogamous_col = None
            if col == "avg_degree":
                concurrent_col = "mean_degree_concurrent"
                monogamous_col = "mean_degree_monogamous"
            elif col == "median_degree":
                concurrent_col = "median_degree_concurrent"
                monogamous_col = "median_degree_monogamous"

            if concurrent_col:
                v_c = metrics.loc[mask, concurrent_col].values
                ax.plot(
                    t_ss,
                    v_c,
                    color=concurrent_color,
                    lw=1.2,
                    linestyle="--",
                    label="Concurrent agents only (deg ≥ 2)",
                )

            if monogamous_col and has_monogamous:
                v_m = metrics.loc[mask, monogamous_col].values
                ax.plot(
                    t_ss,
                    v_m,
                    color=monogamous_color,
                    lw=1.2,
                    linestyle="--",
                    label="Monogamous agents only (deg = 1)",
                )

            ax.axvspan(t_min, t_start, color="#9DC3E6", alpha=0.18, zorder=0)
            ax.axvline(t_start, color="#2E5EFF", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvspan(t_end, t_max, color="#0B3D91", alpha=0.12, zorder=0)
            ax.axvline(t_end, color="#0B3D91", linestyle="--", linewidth=0.8, alpha=0.6)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, color="grey")
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

            window = metrics.loc[mask, col]
            if len(window) > 0:
                summary = f"All agents: mean={window.mean():.3g}"
                if show_max:
                    summary += f"  max={window.max():.3g}"
                ax.text(
                    0.99,
                    0.95,
                    summary,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"
                    ),
                )

            if concurrent_col:
                window_c = metrics.loc[mask, concurrent_col].dropna()
                if len(window_c) > 0:
                    ax.text(
                        0.99,
                        0.83,
                        "Concurrent only",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                        color=concurrent_color,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            alpha=0.85,
                            edgecolor="none",
                        ),
                    )

            if monogamous_col and has_monogamous:
                window_m = metrics.loc[mask, monogamous_col].dropna()
                if len(window_m) > 0:
                    ax.text(
                        0.99,
                        0.71,
                        "Monogamous only",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                        color=monogamous_color,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            alpha=0.85,
                            edgecolor="none",
                        ),
                    )
        axes[0].text(
            t_min + BURN_IN_STEPS / 2,
            1.04,
            "burn-in",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1B3A6B",
            fontstyle="italic",
        )
        axes[0].text(
            t_end + CENSORING_STEPS / 2,
            1.04,
            "censoring",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#0B3D91",
            fontstyle="italic",
        )

        axes[-1].set_xlabel("Timestep (days)", fontsize=9)
        axes[-1].set_xlim(t_min, t_max)

        fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.07, hspace=0.20)
        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written
