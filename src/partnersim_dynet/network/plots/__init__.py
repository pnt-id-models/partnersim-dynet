"""Publication-quality plots for partnership network analysis.

All plot functions:

- Take pre-computed metric DataFrames (no internal graph building)
- Write files to disk via ``save_figure`` (configurable formats)
- Wrap their drawing in ``publication_style`` (no global rcParams pollution)
"""

from partnersim_dynet.network.plots.ego_network_new import (
    EgoLayout,
    build_node_attr,
    build_shared_ego_layouts,
    identify_agents_by_spec,
    identify_top_concurrent_agents,
    plot_ego_3hop_aggregate_per_agent,
    plot_ego_3hop_snapshots,
    plot_ego_network_static_aggregate,
)
from partnersim_dynet.network.plots.heatmap import plot_degree_temporal_heatmaps
from partnersim_dynet.network.plots.style import (
    PALETTE,
    NetworkPalette,
    OutputFormats,
    publication_style,
    save_figure,
)
from partnersim_dynet.network.plots.timeseries import (
    SPEC_LARGEST_COMPONENT_SIZE,
    SPEC_TRANSITIVITY,
    TimeseriesSpec,
    plot_all_structural,
    plot_avg_path_length,
    plot_degree_summary,
    plot_density,
    plot_hub_distribution,
    plot_largest_component_size,
    plot_shortest_path_distribution,
    plot_timeseries,
    plot_transitivity,
)

__all__ = [
    # style
    "PALETTE",
    "NetworkPalette",
    "OutputFormats",
    "publication_style",
    "save_figure",
    # timeseries — specs
    "SPEC_TRANSITIVITY",
    "SPEC_LARGEST_COMPONENT_SIZE",
    "TimeseriesSpec",
    # timeseries — time series plots
    "plot_transitivity",
    "plot_largest_component_size",
    "plot_timeseries",
    "plot_density",
    "plot_degree_summary",
    # timeseries — structural snapshot plots
    "plot_all_structural",
    "plot_shortest_path_distribution",
    "plot_hub_distribution",
    "plot_avg_path_length",
    # heatmap
    "plot_degree_temporal_heatmaps",
    # ego networks (1-hop)
    "EgoLayout",
    "build_node_attr",
    "build_shared_ego_layouts",
    "identify_top_concurrent_agents",
    "identify_agents_by_spec",
    "plot_ego_network_static_aggregate",
    # ego networks (multi-hop)
    "plot_ego_3hop_snapshots",
    "plot_ego_3hop_aggregate_per_agent",
]
