"""Regenerate network plots for a single replicate, without rerunning the simulation.

Reads two parquet files per replicate:
  - partnerships parquet: output of PartnershipGenerator.simulate_partnerships()
  - agent log parquet:    output of PartnershipGenerator.get_agent_log()

and reconstructs the derived objects (PartnershipArrays, ActiveIntervals,
temporal metrics, aggregate graph) needed to call the existing plot
functions directly — skipping the simulation loop entirely.

Usage
-----
    python regenerate_plots.py \\
        --partnerships path/to/run_042_partnerships.parquet \\
        --agent-log path/to/run_042_agents.parquet \\
        --total-timesteps 1875 \\
        --output-dir path/to/plots_out/run_042 \\
        --groups metrics structural ego
 python src/partnersim-dynet/network/plots/regenerate_network_plots.py \\
        --partnerships examples/output/sweep/0pc_concurrency_12thJuly_15kagents/replicate_380863079/partnerships.parquet \\
        --agent-log examples/output/sweep/0pc_concurrency_12thJuly_15kagents/replicate_380863079/agents_log.parquet \\
        --total-timesteps 1875 \\
        --output-dir examples/output/sweep/0pc_concurrency_12thJuly_15kagents/replicate_380863079/updated_network_plots
Groups
------
- metrics:    plot_density, plot_transitivity, plot_largest_component_size,
              plot_degree_summary, plot_avg_path_length (all from metrics.py
              time series — requires compute_temporal_metrics, which is the
              slow step; only run this group if you actually touched
              metrics.py / timeseries.py)
- structural: plot_all_structural (shortest path distribution, hub
              distribution) on the full-simulation aggregate graph
- ego:        plot_ego_network_static_aggregate for the top-N concurrent
              agents over the full simulation window

Add more groups / plot calls as your plotting functions evolve — this is
meant to be edited, not treated as fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from partnersim_dynet.network.active_intervals import ActiveIntervals
from partnersim_dynet.network.graph_builder import PartnershipArrays, prepare_partnerships
from partnersim_dynet.network.metrics import compute_temporal_metrics
from partnersim_dynet.network.plots import (
    plot_all_structural,
    plot_avg_path_length,
    plot_degree_summary,
    plot_density,
    plot_largest_component_size,
    plot_transitivity,
)
from partnersim_dynet.network.plots.ego_network_new import (
    build_node_attr,
    plot_ego_network_static_aggregate,
)
from partnersim_dynet.network.plots.style import OutputFormats

# ---------------------------------------------------------------------------
# Loading + reconstruction
# ---------------------------------------------------------------------------


def load_replicate(
    partnerships_path: Path,
    agent_log_path: Path,
    total_timesteps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, PartnershipArrays, ActiveIntervals]:
    """Load one replicate's parquet files and build the derived structures.

    Returns
    -------
    (partnerships_df, agent_log, partnerships_arrays, active_intervals)
        partnerships_df is the raw DataFrame, kept around because
        identify_top_concurrent_agents() and similar functions take the
        DataFrame form, not the PartnershipArrays form.
    """
    partnerships_df = pd.read_parquet(partnerships_path)
    agent_log = pd.read_parquet(agent_log_path)

    partnerships = prepare_partnerships(partnerships_df, total_timesteps=total_timesteps)
    active = ActiveIntervals.from_agent_log(agent_log, total_timesteps=total_timesteps)

    return partnerships_df, agent_log, partnerships, active


def build_aggregate_graph(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    t_start: int,
    t_end: int,
) -> nx.Graph:
    """Union of every partnership active at any point in [t_start, t_end].

    Public equivalent of the private `_build_aggregate_graph` helper that
    lives inside ego_network_extended.py — duplicated here rather than
    imported since it's underscore-prefixed. If you'd rather have one
    source of truth, promote it to a public function in graph_builder.py
    and import it in both places instead.
    """
    overlap = (partnerships.start <= t_end) & (partnerships.end > t_start)
    a = partnerships.agent[overlap]
    b = partnerships.partner[overlap]

    node_universe: set[int] = set()
    for t in range(t_start, t_end + 1):
        node_universe |= active.active_at(t)

    g = nx.Graph()
    g.add_nodes_from(node_universe)
    for ai, bi in zip(a.tolist(), b.tolist(), strict=False):
        if int(ai) in node_universe and int(bi) in node_universe and ai != bi:
            g.add_edge(int(ai), int(bi))
    return g


# ---------------------------------------------------------------------------
# Plot groups
# ---------------------------------------------------------------------------


def run_metrics_group(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    total_timesteps: int,
    output_dir: Path,
    formats: OutputFormats,
) -> list[str]:
    """Recompute the temporal metrics DataFrame, then run its plots.

    This re-walks the event stream (same cost as during the original sim
    run for the metrics step) — it's the one group here that isn't cheap.
    Only run it if metrics.py / timeseries.py actually changed.
    """
    metrics = compute_temporal_metrics(partnerships, active, total_timesteps=total_timesteps)

    written = []
    written += plot_density(metrics, str(output_dir), formats=formats)
    written += plot_transitivity(metrics, str(output_dir), formats=formats)
    written += plot_largest_component_size(metrics, str(output_dir), formats=formats)
    written += plot_degree_summary(metrics, str(output_dir), formats=formats)
    written += plot_avg_path_length(metrics, str(output_dir), formats=formats)
    return written


def run_structural_group(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    total_timesteps: int,
    output_dir: Path,
    formats: OutputFormats,
) -> list[str]:
    g_agg = build_aggregate_graph(partnerships, active, t_start=1, t_end=total_timesteps)
    return plot_all_structural(g_agg, str(output_dir), formats=formats)


def run_ego_group(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    total_timesteps: int,
    output_dir: Path,
    formats: OutputFormats,
    top_n: int = 6,
) -> list[str]:
    node_attr = build_node_attr(agent_log)
    return plot_ego_network_static_aggregate(
        partnerships_df=partnerships_df,
        partnerships=partnerships,
        active=active,
        output_dir=str(output_dir),
        top_n=top_n,
        t_start=1,
        t_end=total_timesteps,
        node_attr=node_attr,
        formats=formats,
    )


GROUP_RUNNERS = {
    "metrics": run_metrics_group,
    "structural": run_structural_group,
    "ego": run_ego_group,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--partnerships",
        type=Path,
        required=True,
        help="Path to the replicate's partnerships parquet",
    )
    parser.add_argument(
        "--agent-log", type=Path, required=True, help="Path to the replicate's agent log parquet"
    )
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(GROUP_RUNNERS),
        default=["metrics", "structural", "ego"],
        help="Which plot groups to run (default: all)",
    )
    parser.add_argument(
        "--top-n", type=int, default=6, help="Top-N concurrent agents for the ego group"
    )
    parser.add_argument("--pdf", action="store_true", help="Also write PDF versions alongside PNG")
    args = parser.parse_args()

    formats = OutputFormats(png=True, pdf=args.pdf)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    partnerships_df, agent_log, partnerships, active = load_replicate(
        args.partnerships,
        args.agent_log,
        args.total_timesteps,
    )

    written: list[str] = []
    for group in args.groups:
        print(f"[regenerate_plots] running group: {group}")
        if group == "ego":
            written += run_ego_group(
                partnerships_df,
                partnerships,
                active,
                agent_log,
                args.total_timesteps,
                args.output_dir,
                formats,
                top_n=args.top_n,
            )
        else:
            written += GROUP_RUNNERS[group](
                partnerships,
                active,
                args.total_timesteps,
                args.output_dir,
                formats,
            )

    print(f"[regenerate_plots] wrote {len(written)} file(s):")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
