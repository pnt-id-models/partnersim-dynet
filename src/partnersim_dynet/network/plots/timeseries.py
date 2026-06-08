"""Single-metric time series plots.

Four metrics, four plotting functions, all sharing the same layout:
- Line plot with faint fill below
- Summary stats box in the upper right corner
- Configurable x-axis window for warm-up cropping

Each function takes the metrics DataFrame from
``compute_temporal_metrics``, plus an output directory. Files are
written with the metric name as the filename.
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

# ─────────────────────────────────────────────────────────────────────────────
# Spec for a single timeseries plot
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimeseriesSpec:
    """Configuration for one timeseries plot.

    Fixing the spec for each metric in one place avoids parameter
    duplication across the four plot functions.
    """

    metric_column: str
    ylabel: str
    title: str
    color: str
    integer_y: bool = False
    filename_stem: str = ""

    def __post_init__(self) -> None:
        # If filename_stem unset, derive it from the metric_column
        if not self.filename_stem:
            object.__setattr__(self, "filename_stem", f"{self.metric_column}_over_time")


# Built-in specs for the four standard plots
SPEC_AVG_DEGREE = TimeseriesSpec(
    metric_column="avg_degree",
    ylabel="Average degree",
    title="Average network degree (contacts per timestep)",
    color=PALETTE.avg_degree,
)

SPEC_MAX_DEGREE = TimeseriesSpec(
    metric_column="max_degree",
    ylabel="Maximum degree (peak concurrent partners)",
    title="Peak concurrent partners over time",
    color=PALETTE.max_degree,
    integer_y=True,
)

SPEC_TRANSITIVITY = TimeseriesSpec(
    metric_column="transitivity",
    ylabel="Global clustering",
    title="Global clustering over time",
    color=PALETTE.transitivity,
)

SPEC_AVG_PATH_LENGTH = TimeseriesSpec(
    metric_column="avg_path_length",
    ylabel="Average path length (largest component)",
    title="Average path length over time",
    color=PALETTE.avg_path_length,
)

# Core plotting function


def plot_timeseries(
    metrics: pd.DataFrame,
    spec: TimeseriesSpec,
    output_dir: str,
    t_start: int = 1,
    t_end: int | None = None,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Plot one metric over time and write to disk.

    Parameters
    ----------
    metrics : DataFrame
        Output of ``compute_temporal_metrics``. Must contain a ``t``
        column and the column named by ``spec.metric_column``.
    spec : TimeseriesSpec
        Defines the metric column, axis labels, color, etc.
    output_dir : str
        Directory to write the plot files to. Created if missing.
    t_start, t_end : int
        Inclusive x-axis window. If ``t_end`` is None, defaults to the
        last timestep in ``metrics``. Useful for cropping out a warm-up
        period.
    formats : OutputFormats
        Which file formats to write.

    Returns
    -------
    list of str
        Paths of files actually written.

    Raises
    ------
    KeyError
        If ``spec.metric_column`` is not in ``metrics``.
    """
    if spec.metric_column not in metrics.columns:
        raise KeyError(
            f"metric column {spec.metric_column!r} not in metrics DataFrame; "
            f"available columns: {list(metrics.columns)}"
        )

    if t_end is None:
        t_end = int(metrics["t"].max())

    with publication_style():
        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor("white")

        ax.plot(metrics["t"], metrics[spec.metric_column], color=spec.color, lw=1.4)
        ax.fill_between(metrics["t"], 0, metrics[spec.metric_column], color=spec.color, alpha=0.12)

        ax.set_xlim(t_start, t_end)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(spec.ylabel)
        ax.set_title(spec.title)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, color="grey")
        ax.set_axisbelow(True)

        if spec.integer_y:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Window-summary annotation
        window = metrics[(metrics["t"] >= t_start) & (metrics["t"] <= t_end)][spec.metric_column]
        if len(window) > 0:
            stats = f"window mean={window.mean():.3g}   max={window.max():.3g}"
            ax.text(
                0.99,
                0.97,
                stats,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=PALETTE.annotation,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                ),
            )

        fig.tight_layout()

        output_base = os.path.join(output_dir, spec.filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# Wrappers for the four standard plots, which just call the core function


def plot_avg_degree(
    metrics: pd.DataFrame,
    output_dir: str,
    t_start: int = 1,
    t_end: int | None = None,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Plot average degree over time."""
    return plot_timeseries(metrics, SPEC_AVG_DEGREE, output_dir, t_start, t_end, formats)


def plot_max_degree(
    metrics: pd.DataFrame,
    output_dir: str,
    t_start: int = 1,
    t_end: int | None = None,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Plot maximum degree (peak concurrent partners) over time."""
    return plot_timeseries(metrics, SPEC_MAX_DEGREE, output_dir, t_start, t_end, formats)


def plot_transitivity(
    metrics: pd.DataFrame,
    output_dir: str,
    t_start: int = 1,
    t_end: int | None = None,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Plot global clustering (transitivity) over time."""
    return plot_timeseries(metrics, SPEC_TRANSITIVITY, output_dir, t_start, t_end, formats)


def plot_avg_path_length(
    metrics: pd.DataFrame,
    output_dir: str,
    t_start: int = 1,
    t_end: int | None = None,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Plot average path length in the largest connected component."""
    return plot_timeseries(metrics, SPEC_AVG_PATH_LENGTH, output_dir, t_start, t_end, formats)
