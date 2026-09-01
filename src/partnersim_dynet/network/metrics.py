"""Temporal network metrics for partnership simulations.

NOTE: Not all metrics are used in the current plots; some are placeholders for future work.

"""

from __future__ import annotations

import os
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


# Attach agent demographics to a degree DataFrame (used by both snapshot and window views)
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


def degree_stats(G: nx.Graph) -> tuple[float, int, int, float]:
    """Return (avg_degree, max_degree, n_active_nodes, median_degree).

    avg_degree is mean over all nodes (including isolates).
    n_active_nodes is the count of nodes with degree >= 1.
    median_degree is the median degree across all nodes.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0, 0, 0, 0.0
    degs = dict(G.degree())
    deg_values = list(degs.values())
    avg = sum(deg_values) / n
    mx = max(deg_values) if deg_values else 0
    active = sum(1 for d in deg_values if d > 0)
    med = float(np.median(deg_values))
    return avg, mx, active, med


# Degree stats for concurrent and monogamous agents only (degree >= 2 or degree == 1, respectively)
def degree_stats_concurrent(G: nx.Graph) -> tuple[float | None, float | None, int]:
    """Mean/median degree among concurrent agents only (degree >= 2).

    Returns (mean, median, n_concurrent). mean/median are None if no
    agent currently holds 2+ simultaneous partners (e.g. true 0%
    concurrency scenarios).
    """
    concurrent_degs = [d for _, d in G.degree() if d >= 2]
    if not concurrent_degs:
        return None, None, 0
    return float(np.mean(concurrent_degs)), float(np.median(concurrent_degs)), len(concurrent_degs)


# Degree stats for monogamous agents only (degree == 1)
def degree_stats_monogamous(G: nx.Graph) -> tuple[float | None, float | None, int]:
    """Mean/median degree among monogamous agents only (degree == 1).

    Returns (mean, median, n_monogamous). mean/median are None if no
    agent currently has exactly one simultaneous partner.
    """
    monogamous_degs = [d for _, d in G.degree() if d == 1]
    if not monogamous_degs:
        return None, None, 0
    return float(np.mean(monogamous_degs)), float(np.median(monogamous_degs)), len(monogamous_degs)


# Weighted average path length across all components, size-weighted (UNUSED in current plots PLACEHOLDER ONLY)
def weighted_avg_path_length(
    G: nx.Graph,
    sample_size: int,
    rng: np.random.Generator,
) -> float:
    """Average shortest path length across all components, size-weighted."""
    if G.number_of_edges() == 0:
        return 0.0

    comps = list(nx.connected_components(G))
    if not comps:
        return 0.0

    total_weighted_sum = 0.0
    total_pairs = 0

    for comp_nodes in comps:
        n = len(comp_nodes)
        if n < 2:
            continue  # Singletons have no path
        comp_pairs = n * (n - 1) // 2

        subg = G.subgraph(comp_nodes)
        if n <= 5:
            #
            comp_apl = float(nx.average_shortest_path_length(subg))
        else:
            #
            k = min(sample_size, n)
            sources = rng.choice(list(comp_nodes), size=k, replace=False)
            source_means: list[float] = []
            for src in sources:
                sp = nx.single_source_shortest_path_length(subg, int(src))
                others = [v for v in sp.values() if v > 0]
                if others:
                    source_means.append(float(np.mean(others)))
            comp_apl = float(np.mean(source_means)) if source_means else 0.0

        total_weighted_sum += comp_apl * comp_pairs
        total_pairs += comp_pairs

    return total_weighted_sum / total_pairs if total_pairs > 0 else 0.0


# Main driver function for temporal metrics
def compute_temporal_metrics(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    total_timesteps: int,
    apl_sample_size: int = 50,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Compute per-timestep network metrics for the full simulation."""
    rng = np.random.default_rng(rng_seed)

    G = nx.Graph()
    edge_mult: Counter[tuple[int, int]] = Counter()
    currently_in_G: set[int] = set()

    # Index events by timestep for fast per-step processing
    events_by_t = _bucket_events_by_t(partnerships)
    PATH_LENGTH_STRIDE = 10

    metrics: dict[str, list] = defaultdict(list)

    for t in range(1, total_timesteps + 1):
        # Process end events first, then sync node universe, then process start events.
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

        # Sync the node universe to match the current active set at this timestep
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

        # Process start events, adding edges only if both endpoints are currently active
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

        # Compute metrics for this timestep
        avg_deg, max_deg, n_active, med_deg = degree_stats(G)
        mean_c, median_c, n_c = degree_stats_concurrent(G)
        mean_m, median_m, n_m = degree_stats_monogamous(G)
        if t % PATH_LENGTH_STRIDE == 0:
            apl_w = weighted_avg_path_length(G, apl_sample_size, rng)
        else:
            apl_w = np.nan
        metrics["t"].append(t)
        metrics["num_nodes"].append(G.number_of_nodes())
        metrics["num_edges"].append(G.number_of_edges())
        metrics["active_nodes"].append(n_active)
        metrics["avg_degree"].append(avg_deg)
        metrics["max_degree"].append(max_deg)
        metrics["new_edges"].append(step_new)
        metrics["lost_edges"].append(step_lost)
        metrics["avg_path_length_weighted"].append(apl_w)
        metrics["median_degree"].append(med_deg)
        metrics["mean_degree_concurrent"].append(mean_c)
        metrics["median_degree_concurrent"].append(median_c)
        metrics["n_concurrent"].append(n_c)
        metrics["mean_degree_monogamous"].append(mean_m)
        metrics["median_degree_monogamous"].append(median_m)
        metrics["n_monogamous"].append(n_m)

    return pd.DataFrame(metrics)


# Degree at specific snapshots, with demographics attached
def degree_at_snapshots(
    snapshot_times: list[int],
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
) -> pd.DataFrame:
    """Per-agent degree at specific timesteps, with demographics attached.

    For each timestep in `snapshot_times`, build the network graph and
    record every active agent's degree plus their demographic attributes


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
        t, Agent, Degree, median_degree, AgentSex, AgentOrientation, AgentAge, AgentAgeGroup

    Notes
    -----
    Row count is approximately `len(snapshot_times) * num_active_agents`.

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


# Degree in a time window, with demographics attached
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
    some point in [t_start, t_end].

    Parameters
    ----------
    t_start, t_end : int
        Inclusive window bounds, in timesteps. A partnership is
        included if it overlaps the window i.e. if
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


# Degree by demographic combo over time
def degree_by_demographic_over_time(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    total_timesteps: int,
) -> pd.DataFrame:
    """Per-(timestep, demographic combo): degree summary statistics.

    For each timestep and each (AgeGroup, Sex, Orientation) combo,
    computes the mean, median, and count of agents' current
    degrees.

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
    65-74) are omitted.
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

        # Aggregate degree stats by demographic combo at this timestep
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


# edge key for undirected edges: always (smaller, larger) to avoid duplicates
def _edge_key(a: int, b: int) -> tuple[int, int]:
    """Canonical undirected edge key (smaller, larger)."""
    return (a, b) if a < b else (b, a)


# Bucket partnership events by timestep for fast per-step processing
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


# Steady-state summary table (one row, mean/SD stats)
def steady_state_summary_table(
    metrics_df: pd.DataFrame,
    output_dir: str,
    filename_stem: str = "summary_table",
    g_agg: nx.Graph | None = None,
    # top_n_hub: int = 30,
    write_tex: bool = True,
) -> list[str]:
    """Write a one-row publication summary table (csv + optional tex).

    Collapses the steady-state window of `metrics_df` (same window used
    by the timeseries plots' stats boxes) into scalar mean/SD stats.
    """
    from partnersim_dynet.network.plots.timeseries import _steady_state_mask

    mask, *_ = _steady_state_mask(metrics_df)
    w = metrics_df.loc[mask]

    def _mstat(col):
        s = w[col].dropna()
        return (float(s.mean()), float(s.std())) if len(s) else (np.nan, np.nan)

    row: dict = {}
    for col, label in [
        ("avg_degree", "mean_degree"),
        ("median_degree", "median_degree"),
        ("max_degree", "max_degree"),
        ("mean_degree_concurrent", "mean_degree_concurrent"),
        ("median_degree_concurrent", "median_degree_concurrent"),
        ("n_concurrent", "n_concurrent"),
        ("mean_degree_monogamous", "mean_degree_monogamous"),
        ("median_degree_monogamous", "median_degree_monogamous"),
        ("n_monogamous", "n_monogamous"),
        ("avg_path_length_weighted", "avg_path_length"),
    ]:
        if col in w.columns:
            mean_, std_ = _mstat(col)
            row[f"{label}_mean"], row[f"{label}_sd"] = mean_, std_

    if g_agg is not None:
        # row["hub_share_topN"] = float(degs[:top_n_hub].sum() / total) if total else np.nan
        if g_agg.number_of_edges() > 0:
            lcc = g_agg.subgraph(max(nx.connected_components(g_agg), key=len))
            rng = np.random.default_rng(42)
            sources = list(lcc.nodes())
            if len(sources) > 2000:
                sources = rng.choice(sources, size=2000, replace=False).tolist()
            lengths = [
                v
                for src in sources
                for v in nx.single_source_shortest_path_length(lcc, src).values()
                if v > 0
            ]
            if lengths:
                row["shortest_path_mean"] = float(np.mean(lengths))
                row["shortest_path_median"] = float(np.median(lengths))

    df = pd.DataFrame([row])
    os.makedirs(output_dir, exist_ok=True)
    written = []

    csv_path = os.path.join(output_dir, f"{filename_stem}.csv")
    df.to_csv(csv_path, index=False)
    written.append(csv_path)

    if write_tex:
        tex_path = os.path.join(output_dir, f"{filename_stem}.tex")
        df.to_latex(tex_path, index=False, float_format="%.3g")
        written.append(tex_path)

    return written
