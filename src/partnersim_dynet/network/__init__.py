"""Network analysis for partnersim_dynet.

Builds dynamic graphs from partnership data + agent log, computes
temporal network metrics, and produces plots.

All functions take a partnership DataFrame and an agent log
DataFrame (as produced by `PartnershipGenerator`).
"""

from partnersim_dynet.network.active_intervals import ActiveIntervals
from partnersim_dynet.network.graph_builder import (
    PartnershipArrays,
    PartnershipEvent,
    build_graph_at,
    iter_partnership_events,
    prepare_partnerships,
)
from partnersim_dynet.network.metrics import (
    component_stats,
    compute_temporal_metrics,
    degree_at_snapshots,
    degree_by_demographic_over_time,
    degree_in_window,
    degree_stats,
    sampled_avg_path_length,
    transitivity,
)

__all__ = [
    "ActiveIntervals",
    "PartnershipArrays",
    "PartnershipEvent",
    "build_graph_at",
    "iter_partnership_events",
    "prepare_partnerships",
    "component_stats",
    "compute_temporal_metrics",
    "degree_stats",
    "sampled_avg_path_length",
    "transitivity",
    "degree_at_snapshots",
    "degree_by_demographic_over_time",
    "degree_in_window",
]
