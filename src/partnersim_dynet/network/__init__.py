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
    compute_temporal_metrics,
    degree_at_snapshots,
    degree_by_demographic_over_time,
    degree_in_window,
    degree_stats,
    steady_state_summary_table,
)

__all__ = [
    "ActiveIntervals",
    "PartnershipArrays",
    "PartnershipEvent",
    "build_graph_at",
    "iter_partnership_events",
    "prepare_partnerships",
    "compute_temporal_metrics",
    "degree_stats",
    "degree_at_snapshots",
    "degree_by_demographic_over_time",
    "degree_in_window",
    "steady_state_summary_table",
]
