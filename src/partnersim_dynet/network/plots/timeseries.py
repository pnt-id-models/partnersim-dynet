"""Network metric plots for partnership simulation output.

Two categories of plots:

Time series


- ``plot_transitivity``        : global clustering coefficient per timestep
- ``plot_largest_component_size`` : LCC size per timestep


Structural snapshots (distribution plots)
-----------------------------------------
Computed on the aggregate graph over the steady-state window.

- ``plot_shortest_path_distribution`` : shortest path length histogram (sampled)
- ``plot_hub_distribution`` : degree distribution histogram (log-log)
- ``plot_degree_summary`` : mean, median, and max degree time series (stacked panels)
- ``plot_degree_temporal_heatmaps`` : heatmaps of degree by demographic over time

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

Global clustering / transitivity (time series):
    T = 3 * triangles / triads
    where triads = paths of length 2. Measures triangle closure.
    Values near 0 are normal in partnership networks — most partnerships
    are independent (the ego's partners rarely know each other).

Largest connected component (time series):
    |LCC| = number of nodes in the largest connected subgraph.
    High LCC means a single chain of partnerships links many agents,
    relevant to epidemic spread.

Degree distribution P(k) [structural]:
    Probability density: P(k) = (nodes with degree k) / (total nodes * bin_width).
    Plotted on log-log axes. A straight line suggests power-law scaling.

Shortest path length [structural]:
    Sampled: 2000 random source nodes, all reachable targets.
    Mean path length = average geodesic distance in the LCC.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd

from partnersim_dynet.network.plots.style import (
    PALETTE,
    OutputFormats,
    publication_style,
    save_figure,
)

# Constants

BURN_IN_STEPS = 50
CENSORING_STEPS = 50


# Structural plot colours
_C_PATH = "#154E7F"  # blue
_C_BETWEEN = "#154E7F"  # blue


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


# SPEC_AVG_DEGREE = TimeseriesSpec(
#     metric_column="avg_degree", ylabel="Average degree",
#     color=PALETTE.timeseries_line,
# )
# SPEC_MAX_DEGREE = TimeseriesSpec(
#     metric_column="max_degree",
#     ylabel="Maximum degree (peak concurrent partners)",
#     color=PALETTE.timeseries_line, integer_y=True,
# )
SPEC_TRANSITIVITY = TimeseriesSpec(
    metric_column="transitivity",
    ylabel="Global clustering coefficient",
    color=PALETTE.timeseries_line,
)
SPEC_AVG_PATH_LENGTH = TimeseriesSpec(
    metric_column="avg_path_length",
    ylabel="Average path length (largest component)",
    color=PALETTE.timeseries_line,
)
SPEC_LARGEST_COMPONENT_SIZE = TimeseriesSpec(
    metric_column="largest_component_size",
    ylabel="Largest connected component (agents)",
    color=PALETTE.timeseries_line,
    integer_y=True,
)

# Shared helpers


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


# Time-series plots


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
        If True, overlay a linear OLS trend line on the steady-state window.
        Useful for LCC where a long-run upward trend is the key finding.
    low_value_note : str or None
        If set, adds a footnote below the stats box (e.g. for transitivity
        explaining that near-zero values are expected in partnership networks).
    """
    if col not in metrics.columns:
        raise KeyError(f"{col!r} not in metrics DataFrame")

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


def plot_density(metrics, output_dir, formats=OutputFormats()):
    """Network density over time.

    Calculation: 2|E| / (n(n-1)) at each timestep t.
    """
    return _plot_timeseries_core(
        metrics,
        "density",
        "Network density",
        PALETTE.timeseries_line,
        output_dir,
        "density_over_time",
        formats,
        low_value_note="Near-zero expected: most partnerships are independent",
    )


def plot_transitivity(metrics, output_dir, formats=OutputFormats()):
    """Global clustering coefficient (transitivity) over time.

    Calculation: T = 3 × triangles / open_triads
    where open_triads = paths of length 2 (i.e. any node u with two
    neighbours v, w where v–w is not necessarily an edge).

    Values near 0 are expected in partnership networks because an agent's
    partners rarely know each other independently — most edges are 'bridge'
    edges connecting otherwise separate neighbourhoods.
    """
    return _plot_timeseries_core(
        metrics,
        "transitivity",
        "Global clustering coefficient",
        PALETTE.timeseries_line,
        output_dir,
        "transitivity_over_time",
        formats,
        low_value_note="Near-zero expected: partners rarely share contacts",
    )


def plot_hub_distribution(
    g_agg: nx.Graph,
    output_dir: str,
    top_n: int = 30,
    xlim_max: float = 100,
    filename_stem: str = "hub_distribution",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Top-N agents by lifetime degree in the aggregate steady-state graph.

    Horizontal bar chart only (no rank-degree curve). Fixes the x-axis to
    ``xlim_max`` so figures from different scenarios (e.g. 0% vs 15%
    concurrency) share the same scale and are directly visually comparable
    — use the same xlim_max for every scenario you intend to compare.

    Parameters
    ----------
    xlim_max : float
        Upper bound for the x-axis (lifetime degree). Pick a value that
        comfortably covers the highest top-agent degree across all
        scenarios being compared (e.g. 100 or 200), not just this one,
        so bars aren't clipped in the higher-concurrency condition.
    """
    degs = sorted((d for _, d in g_agg.degree()), reverse=True)
    degs = np.array(degs)
    total_endpoints = degs.sum()
    top_share = degs[:top_n].sum() / total_endpoints if total_endpoints else 0.0

    top_nodes = [n for n, _ in sorted(g_agg.degree(), key=lambda x: -x[1])[:top_n]]
    top_degs = [g_agg.degree(n) for n in top_nodes]

    n_clipped = sum(1 for d in top_degs if d > xlim_max)

    with publication_style():
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("white")

        ax.barh(range(top_n), top_degs[::-1], color=_C_BETWEEN, alpha=0.85)
        ax.set_yticks([])
        ax.set_xlabel("Lifetime degree", fontsize=10)
        ax.set_ylabel(f"Top {top_n} agents (rank)", fontsize=9)
        ax.set_xlim(0, xlim_max)
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4, color="grey")
        ax.set_axisbelow(True)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(range(top_n, 0, -1))
        ax.set_ylabel("Agent rank", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        note = f"Top {top_n} agents hold {top_share:.1%}\nof all partnership endpoints"
        if n_clipped:
            note += f"\n({n_clipped} bar{'s' if n_clipped > 1 else ''} exceed x-axis limit)"

        ax.text(
            0.97,
            0.03,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=PALETTE.annotation,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
        )

        fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.12)
        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)
    return written


def plot_largest_component_size(metrics, output_dir, formats=OutputFormats()):
    """Size of the largest connected component over time.

    Calculation: |LCC| = node count of the largest connected subgraph
    at each timestep. A linear trend line is overlaid on the steady-state
    window; a positive slope means partnership chains are growing over the
    simulation — relevant to epidemic spread potential.
    """
    return _plot_timeseries_core(
        metrics,
        "largest_component_size",
        "Largest connected component (agents)",
        PALETTE.timeseries_line,
        output_dir,
        "largest_component_over_time",
        formats,
        integer_y=True,
    )


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


# 1. Shortest path distribution  (per-agent time-averaged mean path length)
def plot_avg_path_length(metrics, output_dir, formats=OutputFormats()):
    """Weighted average shortest path length over time.

    Calculation: size-weighted mean shortest path length across all
    components (see weighted_avg_path_length), computed every
    PATH_LENGTH_STRIDE timesteps. Interpolated for display since the
    underlying column has NaN gaps between stride samples.
    """
    metrics = metrics.copy()
    metrics["avg_path_length_weighted"] = metrics["avg_path_length_weighted"].interpolate(
        limit_direction="both"
    )
    return _plot_timeseries_core(
        metrics,
        "avg_path_length_weighted",
        "Average shortest path length (weighted)",
        PALETTE.timeseries_line,
        output_dir,
        "avg_path_length_over_time",
        formats,
    )


def plot_shortest_path_distribution(
    g: nx.Graph,
    output_dir: str,
    sample_pairs: int = 2000,
    filename_stem: str = "shortest_path_distribution",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Shortest path length distribution in the LCC."""
    lcc_nodes = max(nx.connected_components(g), key=len)
    lcc = g.subgraph(lcc_nodes).copy()

    n = lcc.number_of_nodes()
    sources = list(lcc.nodes())
    rng = np.random.default_rng(42)
    if n > sample_pairs:
        sources = rng.choice(sources, size=sample_pairs, replace=False).tolist()

    lengths = []
    for src in sources:
        spl = nx.single_source_shortest_path_length(lcc, src)
        lengths.extend(v for v in spl.values() if v > 0)

    lengths = np.array(lengths)
    if len(lengths) == 0:
        return []

    max_len = int(lengths.max())
    mean_len = lengths.mean()
    median_len = float(np.median(lengths))

    with publication_style():
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")

        bins = np.arange(0.5, max_len + 1.5, 1)
        ax.hist(lengths, bins=bins, color=_C_PATH, alpha=1.0, edgecolor="white", linewidth=0.5)

        ax.axvline(
            mean_len, color="#CC4125", lw=1.4, linestyle="--", label=f"Mean = {mean_len:.2f}"
        )
        ax.axvline(
            median_len, color="#238B45", lw=1.4, linestyle=":", label=f"Median = {median_len:.1f}"
        )
        ax.legend(frameon=False, fontsize=9, loc="upper right")

        ax.set_xlabel("Shortest path length", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{int(x):,}")
        )
        ax.set_xlim(0.5, max_len + 0.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, color="grey")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.12)
        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


def plot_all_structural(
    g_agg: nx.Graph,
    output_dir: str,
    sample_pairs: int = 2000,
    top_n: int = 30,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Run the structural snapshot plots on the aggregate graph.

    ``shortest_path_distribution`` and ``hub_distribution`` use ``g_agg``
    directly, since degree rank and path-length distribution are
    naturally aggregate-graph concepts. Time-series metrics (including
    avg_path_length_weighted) are handled separately by
    ``run_metrics_group`` / ``plot_avg_path_length``, since they need
    the per-timestep metrics DataFrame, not a single aggregate graph.
    """
    written = plot_shortest_path_distribution(
        g_agg,
        output_dir,
        sample_pairs=sample_pairs,
        filename_stem="shortest_path_distribution",
        formats=formats,
    )
    written += plot_hub_distribution(
        g_agg,
        output_dir,
        top_n=top_n,
        formats=formats,
    )
    return written


# def plot_all_structural(
#     g_agg: nx.Graph,
#     output_dir: str,
#     sample_pairs: int = 2000,
#     top_n: int = 30,
#     formats: OutputFormats = OutputFormats(),
# ) -> list[str]:
#     """Run all five structural snapshot plots.

#     ``degree_distribution`` and ``centrality_distribution`` still use
#     ``g_agg`` (aggregate graph) because degree rank and centrality are
#     naturally aggregate concepts.

#     ``shortest_path``, ``clustering``, and ``embeddedness`` now use the
#     new time-averaged per-agent approach.
#     """

#     written = plot_shortest_path_distribution(
#         g_agg, output_dir,
#         sample_pairs=sample_pairs,
#         filename_stem="shortest_path_distribution",
#         formats=formats,
#     )
#     written += plot_hub_distribution(
#         g_agg, output_dir, top_n=top_n, formats=formats,
#     )

#     written += plot_avg_path_length(
#         metrics=g_agg.graph.get("temporal_metrics", pd.DataFrame()),
#         output_dir=output_dir,
#         formats=formats,
#     )
#     return written
