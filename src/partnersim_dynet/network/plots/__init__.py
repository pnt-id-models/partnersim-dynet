"""Plots for partnership network analysis (1st Sept 2026 version).

All plot functions:

- Take pre-computed metric DataFrames
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
from partnersim_dynet.network.plots.style import (
    PALETTE,
    NetworkPalette,
    OutputFormats,
    publication_style,
    save_figure,
)
from partnersim_dynet.network.plots.timeseries import (
    TimeseriesSpec,
    plot_degree_summary,
    plot_timeseries,
)

__all__ = [
    # style
    "PALETTE",
    "NetworkPalette",
    "OutputFormats",
    "publication_style",
    "save_figure",
    # timeseries
    "TimeseriesSpec",
    "plot_timeseries",
    "plot_degree_summary",
    # ego networks
    "EgoLayout",
    "build_node_attr",
    "build_shared_ego_layouts",
    "identify_top_concurrent_agents",
    "identify_agents_by_spec",
    "plot_ego_network_static_aggregate",
    "plot_ego_3hop_snapshots",
    "plot_ego_3hop_aggregate_per_agent",
]
