"""Build NetworkX graphs from partnership data + active intervals.

The `build_graph_at` function is the snapshot network generator:
given a timestep, it returns a `networkx.Graph` where the node set
is agents active at that timestep, and the edges are the partnerships
active at that timestep.

For time-series metrics that need to track edges entering and leaving
the graph (rather than recomputing from scratch each timestep), use
`iter_partnership_events`, which yields start/end events in time order.

"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from partnersim_dynet.network.active_intervals import ActiveIntervals

# Preprocessing the partnership DataFrame into aligned NumPy arrays allows fast filtering by time.


@dataclass
class PartnershipArrays:
    """Partnership data converted to NumPy arrays once for fast filtering.

    Built via `prepare_partnerships`. All arrays are aligned: position `i`
    in each refers to the same partnership row in the original DataFrame.

    Singleton rows (agents who never partnered) are excluded — they have
    no edge information.

    Attributes
    ----------
    agent : ndarray of int64
        Agent ID of the focal agent in each partnership.
    partner : ndarray of int64
        Agent ID of the partner.
    start : ndarray of int32
        StartTime of each partnership.
    end : ndarray of int32
        EndTime of each partnership. NaN EndTime is filled with
        `total_timesteps + 1` (this is done for consistency with
        ActiveIntervals — partnerships still active at end of simulation
        get a value beyond the last timestep).
    """

    agent: np.ndarray
    partner: np.ndarray
    start: np.ndarray
    end: np.ndarray


def prepare_partnerships(partnerships: pd.DataFrame, total_timesteps: int) -> PartnershipArrays:
    """Convert a partnership DataFrame to aligned NumPy arrays.

    Call this once per analysis, not per snapshot. The returned arrays
    are then reused by `build_graph_at` and `iter_partnership_events`.

    Parameters
    ----------
    partnerships : DataFrame
        Output of `PartnershipGenerator.simulate_partnerships()`. Must
        contain columns: Agent, PartnerAgent, StartTime, EndTime.
        Rows representing singleton agents (no partner) are excluded.
    total_timesteps : int
        Used as the sentinel value for partnerships still active at end
        (so `start <= t < end` queries work uniformly). We do not include
        external partners in the network, so we do not filter them out here.

    Returns
    -------
    PartnershipArrays
        Aligned arrays ready for fast time-based filtering.
    """
    # required = {"Agent", "PartnerAgent", "StartTime", "EndTime", "ExternalPartner"} # Not including external partners for networks currently
    required = {"Agent", "PartnerAgent", "StartTime", "EndTime"}
    missing = required - set(partnerships.columns)
    if missing:
        raise ValueError(f"partnerships DataFrame missing columns: {sorted(missing)}")

    if total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")

    # Filter out singleton rows: agents with no partner
    real = partnerships[
        partnerships["PartnerAgent"].notna() & partnerships["StartTime"].notna()
        # & ~partnerships["ExternalPartner"].fillna(False)
    ].copy()

    agent = real["Agent"].to_numpy(dtype=np.int64)
    partner = real["PartnerAgent"].to_numpy(dtype=np.int64)
    start = real["StartTime"].to_numpy(dtype=np.int32)

    sentinel = total_timesteps + 1
    end = real["EndTime"].fillna(sentinel).to_numpy(dtype=np.int32)

    return PartnershipArrays(agent=agent, partner=partner, start=start, end=end)


# Snapshot graph construction


def build_graph_at(
    t: int,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
) -> nx.Graph:
    """Return the partnership network at timestep t.

    Nodes are every agent active at t (including those with no current
    partnerships — they appear as isolated nodes). Edges are every
    partnership active at t, defined as start_time <= t < end_time.

    The returned graph is a simple `nx.Graph`. If a pair appears in
    multiple partnership rows (re-partnered after dissolving), the edge
    appears once. For multiplicity-aware analysis use
    `iter_partnership_events` instead.

    Parameters
    ----------
    t : int
        Timestep to snapshot at.
    partnerships : PartnershipArrays
        Output of `prepare_partnerships`.
    active : ActiveIntervals
        Output of `ActiveIntervals.from_agent_log`.

    Returns
    -------
    networkx.Graph
        With one node per agent active at t and one edge per active
        partnership.
    """
    G = nx.Graph()

    # Nodes first: every agent active at t, including isolated ones
    active_nodes = active.active_at_array(t)
    G.add_nodes_from(active_nodes.tolist())

    # Edges: active partnerships at t, restricted to active-active pairs
    active_mask = (partnerships.start <= t) & (t < partnerships.end)
    if not active_mask.any():
        return G

    a = partnerships.agent[active_mask]
    b = partnerships.partner[active_mask]

    # Filter for dropping external partnerships as these are not in the agent log.
    # This would indicate inconsistency between partnerships and the agent log
    active_set = set(active_nodes.tolist())
    edges = [
        (int(ai), int(bi))
        for ai, bi in zip(a, b, strict=False)
        if int(ai) in active_set and int(bi) in active_set and ai != bi
    ]
    G.add_edges_from(edges)
    return G


# Event based iteration for time-series metrics
# This is for plotting metrics over time without having to rebuild the graph at every timestep.
@dataclass(slots=True)
class PartnershipEvent:
    """One edge-level event in the partnership timeline."""

    t: int
    agent_a: int
    agent_b: int
    kind: str  # "start" or "end"


def iter_partnership_events(
    partnerships: PartnershipArrays,
) -> Iterator[PartnershipEvent]:
    """Yield partnership start/end events in chronological order.

    Use for metrics that maintain running state across time to avoid
    rebuilding the graph from scratch at every timestep.

    Events at the same timestep are ordered: all ends before any starts.

    """
    n = len(partnerships.agent)
    if n == 0:
        return

    # Build (t, kind_order, i) tuples and sort
    events = np.empty(2 * n, dtype=[("t", np.int32), ("kind", np.int8), ("i", np.int32)])
    events["t"][:n] = partnerships.end
    events["kind"][:n] = 0  # end
    events["i"][:n] = np.arange(n, dtype=np.int32)
    events["t"][n:] = partnerships.start
    events["kind"][n:] = 1  # start
    events["i"][n:] = np.arange(n, dtype=np.int32)

    order = np.lexsort((events["kind"], events["t"]))

    for idx in order:
        e = events[idx]
        i = int(e["i"])
        yield PartnershipEvent(
            t=int(e["t"]),
            agent_a=int(partnerships.agent[i]),
            agent_b=int(partnerships.partner[i]),
            kind="end" if e["kind"] == 0 else "start",
        )
