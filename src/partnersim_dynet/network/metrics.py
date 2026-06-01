"""Temporal network metrics for partnership simulations.

The main entry point is `compute_temporal_metrics`, which produces a
per-timestep DataFrame with degree statistics, connected
components, clustering, and path-length metrics.

For static (single-timestep) analysis, the per-metric functions in this
module work on any `networkx.Graph`. Pass them a graph from
`build_graph_at`

Design
------
- Components are tracked event-by-event using a counter of edges.
  Graph is not rebuilt from scratch each timestep, instead we apply the start/end events for that step.
- APL uses sampled BFS from a random sample of LCC nodes. Each source's
  mean distance to others is computed, then averaged across sources.
- Clustering uses transitivity (3 * triangles / triads), not the per-node
  average.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from partnersim_dynet.network.active_intervals import ActiveIntervals
from partnersim_dynet.network.graph_builder import (
    PartnershipArrays,
    build_graph_at,
    iter_partnership_events,
)


def _attach_demographics(
    degrees: pd.DataFrame,
    agent_log: pd.DataFrame,
    snapshot_t: int | None = None,
) -> pd.DataFrame:
    """Join agent demographics onto a degree DataFrame.

    Parameters
    ----------
    degrees : DataFrame
        Must have columns `Agent` and `Degree`.
    agent_log : DataFrame
        Output of `PartnershipGenerator.get_agent_log()`.
    snapshot_t : int or None
        If provided, the AgentAge column is computed as
        EntryAge + (snapshot_t - EntryTimestep). If None, EntryAge is
        copied through unchanged (useful for window-aggregated views
        where there's no single timestamp).

    Returns
    -------
    DataFrame
        Original columns + AgentSex, AgentOrientation, AgentAge, AgentAgeGroup.
    """
    from partnersim_dynet.config import age_to_group

    demo = agent_log[["Agent", "Sex", "Orientation", "EntryAge", "EntryTimestep"]].copy()
    demo = demo.rename(columns={"Sex": "AgentSex", "Orientation": "AgentOrientation"})

    if snapshot_t is not None:
        demo["AgentAge"] = demo["EntryAge"] + (snapshot_t - demo["EntryTimestep"])
    else:
        demo["AgentAge"] = demo["EntryAge"]

    demo["AgentAgeGroup"] = demo["AgentAge"].apply(age_to_group)
    demo = demo.drop(columns=["EntryAge", "EntryTimestep"])
    return degrees.merge(demo, on="Agent", how="left")


# Per-graph metric functions


def degree_stats(G: nx.Graph) -> tuple[float, int, int]:
    """Return (avg_degree, max_degree, n_active_nodes).

    avg_degree is mean over all nodes (including isolates).
    n_active_nodes is the count of nodes with degree >= 1.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0, 0, 0
    degs = dict(G.degree())
    deg_values = list(degs.values())
    avg = sum(deg_values) / n
    mx = max(deg_values) if deg_values else 0
    active = sum(1 for d in deg_values if d > 0)
    return avg, mx, active


def component_stats(G: nx.Graph) -> tuple[int, int, float]:
    """Return (num_components, largest_component_size, mean_component_size)."""
    comps = list(nx.connected_components(G))
    if not comps:
        return 0, 0, 0.0
    sizes = [len(c) for c in comps]
    return len(comps), max(sizes), float(np.mean(sizes))


def transitivity(G: nx.Graph) -> float:
    """Global clustering: 3 * triangles / triads.

    More informative than per-node average clustering on sparse networks
    because it ignores nodes that can't form triangles (degree < 2).
    Returns 0 for graphs with no edges.
    """
    if G.number_of_edges() == 0:
        return 0.0
    return nx.transitivity(G)


def sampled_avg_path_length(G: nx.Graph, sample_size: int, rng: np.random.Generator) -> float:
    """True mean pairwise shortest-path length in the LCC, sampled.

    Algorithm:
    1. Find the largest connected component (LCC).
    2. Sample up to `sample_size` source nodes uniformly from the LCC.
    3. For each source, compute mean distance to all other LCC nodes.
    4. Return the average of those per-source means.

    Returns 0.0 if the graph has no edges or the LCC is a single node.
    """
    if G.number_of_edges() == 0:
        return 0.0
    comps = list(nx.connected_components(G))
    if not comps:
        return 0.0

    lcc_nodes = max(comps, key=len)
    if len(lcc_nodes) <= 1:
        return 0.0

    lcc = G.subgraph(lcc_nodes)
    k = min(sample_size, len(lcc_nodes))
    sources = rng.choice(list(lcc_nodes), size=k, replace=False)

    source_means: list[float] = []
    for src in sources:
        sp = nx.single_source_shortest_path_length(lcc, int(src))
        # Distances to other nodes (skip the source itself, which is 0)
        others = [v for v in sp.values() if v > 0]
        if others:
            source_means.append(float(np.mean(others)))

    return float(np.mean(source_means)) if source_means else 0.0


# Main driver: time series of metrics


def compute_temporal_metrics(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    total_timesteps: int,
    apl_sample_size: int = 50,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Compute per-timestep network metrics for the full simulation.

    The graph is maintained incrementally: edge multiplicities are
    tracked in a Counter, and the simple `nx.Graph` is updated whenever
    multiplicity transitions 0→1 (add edge) or 1→0 (remove edge). Nodes
    are synced to `active.active_at(t)` once per timestep.

    Parameters
    ----------
    partnerships : PartnershipArrays
        Output of `prepare_partnerships`.
    active : ActiveIntervals
        Output of `ActiveIntervals.from_agent_log`. Defines the node universe.
    total_timesteps : int
        Number of timesteps to compute metrics for (1..total_timesteps).
    apl_sample_size : int
        Number of LCC source nodes to sample for the mean-pairwise-distance
        estimate. 50 is a good default; larger sizes reduce variance but
        increase per-timestep cost linearly.
    _sample_size : int
        Number of LCC source nodes to sample
    rng_seed : int
        Seed for the source-node sampling. Different seeds give different
        APL estimates within sampling noise.

    Returns
    -------
    DataFrame
        Columns: t, num_nodes, num_edges, active_nodes, avg_degree,
        max_degree,new_edges, lost_edges, num_components,
        largest_component_size, mean_component_size, transitivity,
        and avg_path_length
    """
    rng = np.random.default_rng(rng_seed)

    G = nx.Graph()
    edge_mult: Counter[tuple[int, int]] = Counter()
    currently_in_G: set[int] = set()

    # Index events by timestep for fast per-step processing
    events_by_t = _bucket_events_by_t(partnerships)

    metrics: dict[str, list] = defaultdict(list)

    for t in range(1, total_timesteps + 1):
        # ── (1) Process end events at this timestep ─────────────────
        # Ends come before starts at the same t (see iter_partnership_events).
        step_lost = 0
        for ev in events_by_t.get(t, []):
            if ev.kind != "end":
                continue
            key = _edge_key(ev.agent_a, ev.agent_b)
            if edge_mult[key] > 0:
                edge_mult[key] -= 1
                if edge_mult[key] == 0:
                    if G.has_edge(ev.agent_a, ev.agent_b):
                        G.remove_edge(ev.agent_a, ev.agent_b)
                    step_lost += 1

        # ── (2) Sync node universe to active-at-t ───────────────────
        active_now = active.active_at(t)
        to_add = active_now - currently_in_G
        to_remove = currently_in_G - active_now
        if to_add:
            G.add_nodes_from(to_add)
        if to_remove:
            for n in to_remove:
                # Defensive: clear any remaining multiplicities for
                # edges incident on removed nodes.
                for a, b in list(G.edges(n)):
                    edge_mult[_edge_key(a, b)] = 0
                G.remove_node(n)
        currently_in_G = active_now

        # ── (3) Process start events at this timestep ───────────────
        step_new = 0
        for ev in events_by_t.get(t, []):
            if ev.kind != "start":
                continue
            if ev.agent_a == ev.agent_b:
                continue
            # Only add edges where both endpoints are currently active
            if ev.agent_a not in currently_in_G or ev.agent_b not in currently_in_G:
                continue
            key = _edge_key(ev.agent_a, ev.agent_b)
            if edge_mult[key] == 0:
                G.add_edge(ev.agent_a, ev.agent_b)
                step_new += 1
            edge_mult[key] += 1

        # ── (4) Compute per-step metrics ────────────────────────────
        avg_deg, max_deg, n_active = degree_stats(G)
        n_comp, lcc_size, mean_comp = component_stats(G)

        metrics["t"].append(t)
        metrics["num_nodes"].append(G.number_of_nodes())
        metrics["num_edges"].append(G.number_of_edges())
        metrics["active_nodes"].append(n_active)
        metrics["avg_degree"].append(avg_deg)
        metrics["max_degree"].append(max_deg)
        metrics["new_edges"].append(step_new)
        metrics["lost_edges"].append(step_lost)
        metrics["num_components"].append(n_comp)
        metrics["largest_component_size"].append(lcc_size)
        metrics["mean_component_size"].append(mean_comp)
        metrics["transitivity"].append(transitivity(G))
        metrics["avg_path_length"].append(sampled_avg_path_length(G, apl_sample_size, rng))

    return pd.DataFrame(metrics)


# Add after compute_temporal_metrics
def degree_at_snapshots(
    snapshot_times: list[int],
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
) -> pd.DataFrame:
    """Per-agent degree at specific timesteps, with demographics attached.

    For each timestep in `snapshot_times`, build the network graph and
    record every active agent's degree plus their demographic attributes
    at that moment. Useful for static cross-sectional analysis: degree
    distributions, age-degree scatter plots, demographic stratification
    at a specific point in time.

    Parameters
    ----------
    snapshot_times : list of int
        Timesteps to snapshot at. Each must be in [1, total_timesteps].
    partnerships : PartnershipArrays
        From `prepare_partnerships`.
    active : ActiveIntervals
        From `ActiveIntervals.from_agent_log`.
    agent_log : DataFrame
        Output of `PartnershipGenerator.get_agent_log()`. Used to
        compute each agent's age at the snapshot timestep.

    Returns
    -------
    DataFrame with columns:
        t, Agent, Degree, AgentSex, AgentOrientation, AgentAge, AgentAgeGroup

    Notes
    -----
    Row count is approximately `len(snapshot_times) * num_active_agents`.
    For a 5-snapshot, 15k-agent run that's ~75k rows — well within
    memory.
    """
    if not snapshot_times:
        raise ValueError("snapshot_times must be a non-empty list")

    pieces: list[pd.DataFrame] = []
    for t in snapshot_times:
        G = build_graph_at(t, partnerships, active)
        if G.number_of_nodes() == 0:
            continue

        rows = [{"t": t, "Agent": int(n), "Degree": int(d)} for n, d in G.degree()]
        df_t = pd.DataFrame(rows)
        df_t = _attach_demographics(df_t, agent_log, snapshot_t=t)
        pieces.append(df_t)

    if not pieces:
        return pd.DataFrame(
            columns=[
                "t",
                "Agent",
                "Degree",
                "AgentSex",
                "AgentOrientation",
                "AgentAge",
                "AgentAgeGroup",
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def degree_in_window(
    t_start: int,
    t_end: int,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
) -> pd.DataFrame:
    """Per-agent count of distinct partners during [t_start, t_end].

    For each agent active during any part of the window, count the
    number of distinct other agents they had a partnership with at
    some point in [t_start, t_end]. Used for static / aggregated
    network views: ego-network drawings, cumulative degree distributions,
    "how many partners did each agent have during this 5-year period".

    Parameters
    ----------
    t_start, t_end : int
        Inclusive window bounds, in timesteps. A partnership is
        included if it overlaps the window at all — i.e. if
        `start <= t_end and end > t_start`.
    partnerships : PartnershipArrays
        From `prepare_partnerships`.
    active : ActiveIntervals
        From `ActiveIntervals.from_agent_log`.
    agent_log : DataFrame
        For demographic attachment.

    Returns
    -------
    DataFrame with columns:
        Agent, Degree, AgentSex, AgentOrientation, AgentAge, AgentAgeGroup
        Note: no `t` column — this is a single aggregated row per agent.

    Notes
    -----
    `Degree` counts DISTINCT partners. If two agents partnered, broke
    up, and re-partnered within the window, they count once each.
    `AgentAge` is the agent's age at the start of the window.
    """
    if t_start > t_end:
        raise ValueError(f"t_start ({t_start}) must be <= t_end ({t_end})")

    # Filter partnerships overlapping the window
    overlap = (partnerships.start <= t_end) & (partnerships.end > t_start)
    a = partnerships.agent[overlap]
    b = partnerships.partner[overlap]

    # Build a dict of agent -> set of distinct partners (both directions)
    partners: dict[int, set[int]] = defaultdict(set)
    for ai, bi in zip(a.tolist(), b.tolist(), strict=False):
        if ai == bi:
            continue
        partners[ai].add(bi)
        partners[bi].add(ai)

    # The node universe: agents active at any point in the window
    active_in_window: set[int] = set()
    for t in range(t_start, t_end + 1):
        active_in_window |= active.active_at(t)

    rows = [
        {"Agent": aid, "Degree": len(partners.get(aid, set()))} for aid in sorted(active_in_window)
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Agent",
                "Degree",
                "AgentSex",
                "AgentOrientation",
                "AgentAge",
                "AgentAgeGroup",
            ]
        )

    return _attach_demographics(df, agent_log, snapshot_t=t_start)


def degree_by_demographic_over_time(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    total_timesteps: int,
) -> pd.DataFrame:
    """Per-(timestep, demographic combo): degree summary statistics.

    For each timestep and each (AgeGroup, Sex, Orientation) combo,
    computes the mean, median, p90, and count of agents' current
    degrees. Suitable for temporal stratified plots: "average degree
    of bisexual males 25-34 over time", per-demographic heatmaps,
    or comparing degree dynamics across demographic groups.

    Parameters
    ----------
    partnerships : PartnershipArrays
    active : activeIntervals
    agent_log : DataFrame
        Used to look up each agent's demographic attributes.
    total_timesteps : int

    Returns
    -------
    DataFrame with columns:
        t, AgentSex, AgentOrientation, AgentAgeGroup,
        MeanDegree, P50Degree, P90Degree, N

    Notes
    -----
    Rows where no agents matched a combo (e.g. no female bisexuals
    65-74) are omitted. AgentAge is implicit: every agent contributes
    to whichever AgeGroup they were in at the start of that timestep.
    """
    from partnersim_dynet.config import age_to_group

    # Pre-build agent -> (sex, orientation, entry_age, entry_t) lookup
    # so we can compute age_group efficiently per timestep.
    demo_lookup = {
        int(row["Agent"]): (
            row["Sex"],
            row["Orientation"],
            int(row["EntryAge"]),
            int(row["EntryTimestep"]),
        )
        for _, row in agent_log.iterrows()
    }

    # Reuse the same event-driven graph maintenance as compute_temporal_metrics
    G = nx.Graph()
    edge_mult: Counter[tuple[int, int]] = Counter()
    currently_in_G: set[int] = set()
    events_by_t = _bucket_events_by_t(partnerships)

    rows: list[dict] = []

    for t in range(1, total_timesteps + 1):
        # Ends first (same logic as compute_temporal_metrics)
        for ev in events_by_t.get(t, []):
            if ev.kind != "end":
                continue
            key = _edge_key(ev.agent_a, ev.agent_b)
            if edge_mult[key] > 0:
                edge_mult[key] -= 1
                if edge_mult[key] == 0 and G.has_edge(ev.agent_a, ev.agent_b):
                    G.remove_edge(ev.agent_a, ev.agent_b)

        # Sync nodes
        active_now = active.active_at(t)
        to_add = active_now - currently_in_G
        to_remove = currently_in_G - active_now
        if to_add:
            G.add_nodes_from(to_add)
        if to_remove:
            for n in to_remove:
                for a, b in list(G.edges(n)):
                    edge_mult[_edge_key(a, b)] = 0
                G.remove_node(n)
        currently_in_G = active_now

        # Starts
        for ev in events_by_t.get(t, []):
            if ev.kind != "start":
                continue
            if (
                ev.agent_a == ev.agent_b
                or ev.agent_a not in currently_in_G
                or ev.agent_b not in currently_in_G
            ):
                continue
            key = _edge_key(ev.agent_a, ev.agent_b)
            if edge_mult[key] == 0:
                G.add_edge(ev.agent_a, ev.agent_b)
            edge_mult[key] += 1

        # ── Per-demographic aggregation ──────────────────────────
        # Bucket every active agent's degree by their current combo.
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for node, deg in G.degree():
            info = demo_lookup.get(int(node))
            if info is None:
                continue
            sex, orientation, entry_age, entry_t = info
            current_age = entry_age + (t - entry_t)
            age_group = age_to_group(current_age)
            buckets[(sex, orientation, age_group)].append(int(deg))

        for (sex, orientation, age_group), degs in buckets.items():
            arr = np.asarray(degs)
            rows.append(
                {
                    "t": t,
                    "AgentSex": sex,
                    "AgentOrientation": orientation,
                    "AgentAgeGroup": age_group,
                    "MeanDegree": float(arr.mean()),
                    "P50Degree": float(np.percentile(arr, 50)),
                    "P90Degree": float(np.percentile(arr, 90)),
                    "N": int(len(arr)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "t",
                "AgentSex",
                "AgentOrientation",
                "AgentAgeGroup",
                "MeanDegree",
                "P50Degree",
                "P90Degree",
                "N",
            ]
        )
    return pd.DataFrame(rows)


# Internal helpers


def _edge_key(a: int, b: int) -> tuple[int, int]:
    """Canonical undirected edge key (smaller, larger)."""
    return (a, b) if a < b else (b, a)


def _bucket_events_by_t(
    partnerships: PartnershipArrays,
) -> dict[int, list]:
    """Group partnership events by their timestep.

    Within each timestep, events are in the iterator's order (ends
    before starts). Returns a dict mapping t → list of events.
    """
    buckets: dict[int, list] = defaultdict(list)
    for ev in iter_partnership_events(partnerships):
        buckets[ev.t].append(ev)
    return buckets
