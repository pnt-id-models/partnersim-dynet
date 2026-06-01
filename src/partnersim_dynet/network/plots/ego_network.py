"""Ego-network plots for top-concurrent-partnership agents.

Three plot variants share the same drawing settings and node layout:

- ``plot_ego_network_dynamic``: multi-panel evolution. Rows = top
  agents, columns = timesteps. Each panel shows the ego subgraph at
  that one timestep (only currently-active partnerships).

- ``plot_ego_network_active_snapshot``: single column. One ego subgraph
  per top agent, as of one chosen snapshot timestep.

- ``plot_ego_network_static_aggregate``: single column. One ego
  subgraph per top agent, aggregating EVERY partnership during
  ``[t_start, t_end]`` into a single graph.

All three render top agents' ego subgraphs using consistent visual
encoding:
  - Marker shape: sex (circle = Males, square = Females)
  - Color:        orientation (blue/green/purple, see PALETTE)
  - Size:         local degree within the ego subgraph
  - Label:        age (years), only on sparse networks

Node positions are computed once from the time-aggregated graph
(superset of all three views), so an agent appears in the same place
across all three plots — making them directly comparable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from partnersim_dynet.network.active_intervals import ActiveIntervals
from partnersim_dynet.network.graph_builder import (
    PartnershipArrays,
    build_graph_at,
)
from partnersim_dynet.network.plots.style import (
    PALETTE,
    OutputFormats,
    publication_style,
    save_figure,
)

# Identification: which agents had the most concurrent partnerships?


def identify_top_concurrent_agents(
    partnerships: pd.DataFrame,
    top_n: int,
) -> list[int]:
    """Return the top-N agent IDs by maximum simultaneous partnerships.

    For each agent, computes the maximum number of partnerships they
    held simultaneously at any point during the simulation. Ranks by
    that maximum; ties broken by total partnership count. Filters to
    agents who held >= 2 simultaneous partners (since "concurrent" by
    definition); if fewer than ``top_n`` qualify, fills from the rest
    of the ranking.

    Parameters
    ----------
    partnerships : DataFrame
        Output of ``PartnershipGenerator.simulate_partnerships()``.
        Singleton rows (PartnerAgent NaN) are excluded.
    top_n : int
        How many agents to return.

    Returns
    -------
    list of int
        Agent IDs in descending rank order. May be shorter than
        ``top_n`` if there aren't enough agents in the data.
    """
    real = partnerships[
        partnerships["PartnerAgent"].notna() & partnerships["StartTime"].notna()
    ].copy()
    # Need both start and end times for overlap computation
    real = real[real["EndTime"].notna()]
    if real.empty:
        return []

    real["StartTime"] = real["StartTime"].astype(int)
    real["EndTime"] = real["EndTime"].astype(int)

    rows = []
    for agent, grp in real.groupby("Agent"):
        # Sweepline algorithm: process +1 at start, -1 at end, track max
        events = []
        for s, e in zip(grp["StartTime"], grp["EndTime"], strict=False):
            events.append((s, +1))
            events.append((e, -1))
        events.sort()
        max_simul = cur = 0
        for _, delta in events:
            cur += delta
            max_simul = max(max_simul, cur)
        rows.append(
            {
                "Agent": int(agent),
                "MaxSimultaneous": max_simul,
                "TotalPartnerships": len(grp),
            }
        )

    rank_df = pd.DataFrame(rows).sort_values(
        ["MaxSimultaneous", "TotalPartnerships"], ascending=False
    )
    concurrent = rank_df[rank_df["MaxSimultaneous"] >= 2]

    if len(concurrent) >= top_n:
        return concurrent.head(top_n)["Agent"].tolist()

    # Pad from non-concurrent agents to reach top_n
    extra = rank_df[~rank_df["Agent"].isin(concurrent["Agent"])]
    combined = pd.concat([concurrent, extra]).head(top_n)
    return combined["Agent"].tolist()


# Node layout: shared across the three plot variants


@dataclass(frozen=True)
class EgoLayout:
    """Computed positions for one agent's ego network.

    Shared across the three plot variants so the agent and their
    partners appear in the same location in all three figures.
    """

    positions: dict[int, np.ndarray]  # node id -> (x, y)
    bounds: tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)


def build_shared_ego_layouts(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    ego_agents: list[int],
    t_start: int,
    t_end: int,
    ego_radius: int = 1,
) -> dict[int, EgoLayout]:
    """Compute one positions+bounds layout per ego agent.

    Computed on the time-aggregated graph (every partnership in
    ``[t_start, t_end]``) so the layout is a superset of every snapshot.
    Each agent and every potential partner gets a fixed (x, y), and the
    same coordinates are reused by the three ego-network plot functions.

    Parameters
    ----------
    partnerships, active : as in the metrics module.
    ego_agents : list of int
        Agent IDs for which to build layouts.
    t_start, t_end : int
        Window over which the aggregated graph is built.
    ego_radius : int
        Ego graph radius (1 = direct partners only).

    Returns
    -------
    dict mapping each ego agent ID to its ``EgoLayout``.
    """
    # Build the aggregated graph once, then extract per-ego subgraphs
    overlap = (partnerships.start <= t_end) & (partnerships.end > t_start)
    a = partnerships.agent[overlap]
    b = partnerships.partner[overlap]

    # Node universe: agents active at any point in the window
    node_universe: set[int] = set()
    for t in range(t_start, t_end + 1):
        node_universe |= active.active_at(t)

    G_agg = nx.Graph()
    G_agg.add_nodes_from(node_universe)
    for ai, bi in zip(a.tolist(), b.tolist(), strict=False):
        if int(ai) in node_universe and int(bi) in node_universe and ai != bi:
            G_agg.add_edge(int(ai), int(bi))

    layouts: dict[int, EgoLayout] = {}
    for ego in ego_agents:
        if ego in G_agg:
            sub = nx.ego_graph(G_agg, ego, ego_radius)
        else:
            sub = nx.Graph()
            sub.add_node(ego)

        if sub.number_of_nodes() == 1:
            layouts[ego] = EgoLayout(
                positions={ego: np.array([0.0, 0.0])},
                bounds=(-1.0, 1.0, -1.0, 1.0),
            )
            continue

        positions = nx.spring_layout(
            sub,
            seed=int(ego) % (2**31),
            k=1.2 / np.sqrt(len(sub)),
            iterations=250,
            fixed=[ego],
            pos={ego: np.array([0.0, 0.0])},
        )
        xs = np.array([p[0] for p in positions.values()])
        ys = np.array([p[1] for p in positions.values()])
        pad = 0.15
        bounds = (
            float(xs.min()) - pad,
            float(xs.max()) + pad,
            float(ys.min()) - pad,
            float(ys.max()) + pad,
        )
        layouts[ego] = EgoLayout(positions=positions, bounds=bounds)

    return layouts


# Drawing settings — shared by all three plot variants


def _draw_ego_panel(
    ax,
    ego_subgraph: nx.Graph,
    layout: EgoLayout,
    node_attr: Mapping[int, Mapping] | None,
    edge_color: str = "#222222",
    edge_alpha: float = 0.85,
    edge_width: float = 1.5,
    show_age_labels: bool = True,
    age_label_max_nodes: int = 30,
    custom_bounds: tuple | None = None,
) -> None:
    """Draw one ego subgraph onto an axes using the shared layout.

    ``node_attr`` maps each agent id to a dict with keys ``Sex``,
    ``Orientation``, ``Age``. Missing entries fall back to defaults
    (gray color, "o" shape, no age label).

    ``custom_bounds`` overrides ``layout.bounds`` — used by the active-
    snapshot variant to zoom in on a small set of active nodes within
    a layout sized for the much larger aggregate graph.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    lo_x, hi_x, lo_y, hi_y = custom_bounds if custom_bounds else layout.bounds
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_aspect("equal")

    nodes = list(ego_subgraph.nodes())
    pos = dict(layout.positions)

    # Fill in random positions for nodes that aren't in the shared layout
    rng = np.random.default_rng(0)
    for n in nodes:
        if n not in pos:
            pos[n] = np.array(
                [
                    rng.uniform(lo_x + 0.1, hi_x - 0.1),
                    rng.uniform(lo_y + 0.1, hi_y - 0.1),
                ]
            )

    # Edges
    if ego_subgraph.number_of_edges() > 0:
        nx.draw_networkx_edges(
            ego_subgraph,
            pos,
            ax=ax,
            edge_color=edge_color,
            alpha=edge_alpha,
            width=edge_width,
        )

    # Nodes — grouped by sex (for marker shape)
    local_deg = dict(ego_subgraph.degree())
    local_max_deg = max(max(local_deg.values()) if local_deg else 1, 1)

    def _attr(n):
        return node_attr.get(n, {}) if node_attr else {}

    for sex in ("Male", "Female"):
        marker = PALETTE.sex_shape(sex)
        group = [n for n in nodes if _attr(n).get("Sex") == sex]
        if not group:
            continue

        colors = [PALETTE.orientation_color(_attr(n).get("Orientation", "")) for n in group]
        sizes = [80 + 220 * (local_deg.get(n, 0) / local_max_deg) for n in group]
        nx.draw_networkx_nodes(
            ego_subgraph,
            pos,
            nodelist=group,
            ax=ax,
            node_color=colors,
            node_size=sizes,
            node_shape=marker,
            alpha=1.0,
            linewidths=0.5,
            edgecolors="black",
        )

    # Age labels — only on sparse networks
    if show_age_labels and len(nodes) <= age_label_max_nodes:
        for idx, n in enumerate(nodes):
            age = _attr(n).get("Age", "")
            if age == "":
                continue
            x, y = pos[n]
            offset_y = 0.06 if idx % 2 == 0 else -0.06
            offset_x = 0.03 if idx % 3 == 0 else (-0.03 if idx % 3 == 1 else 0)
            ax.text(
                x + offset_x,
                y + offset_y,
                str(age),
                fontsize=6.5,
                ha="center",
                va="bottom" if offset_y > 0 else "top",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                ),
            )


def _add_ego_legend(fig, y_anchor: float = 0.02) -> None:
    """Add a unified legend at the bottom of an ego-network figure."""
    handles = []

    handles.append(Line2D([0], [0], color="none", label="Sex:"))
    for sex in ("Male", "Female"):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=PALETTE.sex_shape(sex),
                color="black",
                linestyle="None",
                markersize=7,
                label=sex,
            )
        )

    handles.append(Line2D([0], [0], color="none", label="   Orientation:"))
    for ori in ("Opposite-sex", "Same-sex", "Bisexual"):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PALETTE.orientation_color(ori),
                markersize=8,
                label=ori,
            )
        )

    handles.append(Line2D([0], [0], color="none", label="   Degree:"))
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=4,
            label="low",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="high",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color="none",
            label="   Numeric labels = age (years)",
        )
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, y_anchor),
        handletextpad=0.4,
        columnspacing=0.8,
    )


# Variant 1: dynamic — multi-panel evolution across timesteps
def plot_ego_network_dynamic(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    output_dir: str,
    top_n: int,
    timesteps: list[int],
    node_attr: Mapping[int, Mapping],
    ego_radius: int = 1,
    filename_stem: str = "ego_network_dynamic",
    formats: OutputFormats = OutputFormats(),
    shared_layouts: dict[int, EgoLayout] | None = None,
) -> list[str]:
    """Multi-panel evolution: ego networks across multiple timesteps.

    Rows = top-N most concurrent agents, columns = timesteps. Each cell
    shows the ego subgraph as of that timestep (only partnerships
    currently active).

    Parameters
    ----------
    partnerships_df : DataFrame
        The original partnership DataFrame (for identifying top agents).
    partnerships : PartnershipArrays
        Preprocessed partnerships.
    active : ActiveIntervals
        Agent activity windows.
    output_dir : str
    top_n : int
    timesteps : list of int
        Columns of the figure. Order is preserved.
    node_attr : mapping from agent_id to dict
        Each dict can contain Sex, Orientation, Age. Build this once
        from the agent log and reuse across the three plot variants.
    ego_radius : int
    filename_stem : str
    formats : OutputFormats
    shared_layouts : dict[int, EgoLayout] or None
        If supplied, used for node positioning. If None, computed
        internally from the aggregated graph across ``timesteps``.

    Returns
    -------
    list of file paths written.
    """
    if not timesteps:
        raise ValueError("timesteps must be a non-empty list")

    top_agents = identify_top_concurrent_agents(partnerships_df, top_n=top_n)
    if not top_agents:
        return []

    if shared_layouts is None:
        shared_layouts = build_shared_ego_layouts(
            partnerships,
            active,
            top_agents,
            t_start=min(timesteps),
            t_end=max(timesteps),
            ego_radius=ego_radius,
        )

    n_rows = len(top_agents)
    n_cols = len(timesteps)

    with publication_style():
        fig = plt.figure(
            figsize=(3.6 * n_cols, 4.0 * n_rows + 1.2),
            constrained_layout=False,
        )
        gs = gridspec.GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            hspace=0.10,
            wspace=0.05,
            left=0.06,
            right=0.97,
            top=0.86,
            bottom=0.18,
        )

        for i, ego in enumerate(top_agents):
            for j, t in enumerate(timesteps):
                ax = fig.add_subplot(gs[i, j])
                G_t = build_graph_at(t, partnerships, active)
                if ego in G_t:
                    sub = nx.ego_graph(G_t, ego, ego_radius)
                else:
                    sub = nx.Graph()
                    sub.add_node(ego)

                _draw_ego_panel(
                    ax,
                    sub,
                    shared_layouts[ego],
                    node_attr,
                    edge_color="#222222",
                    edge_alpha=0.85,
                    edge_width=1.4,
                )

                if i == 0:
                    ax.set_title(f"t = {t}", fontsize=11, fontweight="bold", pad=6)
                if j == 0:
                    ax.set_ylabel(f"Agent {ego}", fontsize=10, labelpad=8)

                n_e = sub.number_of_edges()
                n_p = max(sub.number_of_nodes() - 1, 0)
                ax.text(
                    0.98,
                    0.98,
                    f"{n_e} edges • {n_p} partners",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color=PALETTE.annotation,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.75,
                        edgecolor="none",
                    ),
                )

        _add_ego_legend(fig, y_anchor=0.04)
        fig.suptitle(
            "Dynamic ego networks — partnerships active at each timestep",
            fontsize=12,
            fontweight="bold",
            y=0.96,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# Variant 2: active snapshot — one timestep, one column


def plot_ego_network_active_snapshot(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    output_dir: str,
    top_n: int,
    snapshot_t: int,
    node_attr: Mapping[int, Mapping],
    ego_radius: int = 1,
    zoom_to_active: bool = True,
    filename_stem: str = "ego_network_active_snapshot",
    formats: OutputFormats = OutputFormats(),
    shared_layouts: dict[int, EgoLayout] | None = None,
) -> list[str]:
    """Single-column ego networks at one snapshot timestep.

    For each top-N agent, draw the ego subgraph showing only the
    partnerships ACTIVE at ``snapshot_t``. Optionally zoom the
    viewport to the active subset (otherwise inherits the full
    aggregated layout's bounds).
    """
    top_agents = identify_top_concurrent_agents(partnerships_df, top_n=top_n)
    if not top_agents:
        return []

    if shared_layouts is None:
        shared_layouts = build_shared_ego_layouts(
            partnerships,
            active,
            top_agents,
            t_start=1,
            t_end=snapshot_t,
            ego_radius=ego_radius,
        )

    n_rows = len(top_agents)

    with publication_style():
        fig = plt.figure(
            figsize=(6.0, 4.5 * n_rows + 1.2),
            constrained_layout=False,
        )
        gs = gridspec.GridSpec(
            n_rows,
            1,
            figure=fig,
            hspace=0.18,
            left=0.10,
            right=0.94,
            top=0.88,
            bottom=0.18,
        )

        for i, ego in enumerate(top_agents):
            ax = fig.add_subplot(gs[i, 0])

            G_t = build_graph_at(snapshot_t, partnerships, active)
            if ego in G_t:
                sub = nx.ego_graph(G_t, ego, ego_radius)
            else:
                sub = nx.Graph()
                sub.add_node(ego)

            # Compute zoomed bounds from the active subset of nodes
            custom_bounds = None
            if zoom_to_active and sub.number_of_nodes() > 1:
                pos = shared_layouts[ego].positions
                xs = np.array([pos[n][0] for n in sub.nodes() if n in pos])
                ys = np.array([pos[n][1] for n in sub.nodes() if n in pos])
                if len(xs) > 0:
                    pad = max(0.15, 0.3 * max(xs.max() - xs.min(), ys.max() - ys.min()))
                    custom_bounds = (
                        float(xs.min() - pad),
                        float(xs.max() + pad),
                        float(ys.min() - pad),
                        float(ys.max() + pad),
                    )

            _draw_ego_panel(
                ax,
                sub,
                shared_layouts[ego],
                node_attr,
                edge_color="#0E3926",
                edge_alpha=0.9,
                edge_width=1.6,
                custom_bounds=custom_bounds,
            )

            ax.set_ylabel(f"Agent {ego}", fontsize=10, labelpad=8)
            n_e = sub.number_of_edges()
            n_p = max(sub.number_of_nodes() - 1, 0)
            ax.text(
                0.98,
                0.98,
                f"{n_e} active edges • {n_p} current partners",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=PALETTE.annotation,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                ),
            )

        _add_ego_legend(fig, y_anchor=0.04)
        fig.suptitle(
            f"Active-snapshot ego networks — partnerships active at t = {snapshot_t}",
            fontsize=12,
            fontweight="bold",
            y=0.95,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# Variant 3: static aggregate — every partnership in [t_start, t_end]


def plot_ego_network_static_aggregate(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    output_dir: str,
    top_n: int,
    t_start: int,
    t_end: int,
    node_attr: Mapping[int, Mapping],
    ego_radius: int = 1,
    filename_stem: str = "ego_network_static_aggregate",
    formats: OutputFormats = OutputFormats(),
    shared_layouts: dict[int, EgoLayout] | None = None,
) -> list[str]:
    """Static aggregate: every partnership in [t_start, t_end] on one figure.

    For each top-N agent, draw the ego subgraph aggregating ALL
    partnerships that occurred during ``[t_start, t_end]`` into one
    graph. Edge weights are not depicted; this is a presence-only view.
    """
    if t_start > t_end:
        raise ValueError(f"t_start ({t_start}) must be <= t_end ({t_end})")

    top_agents = identify_top_concurrent_agents(partnerships_df, top_n=top_n)
    if not top_agents:
        return []

    if shared_layouts is None:
        shared_layouts = build_shared_ego_layouts(
            partnerships,
            active,
            top_agents,
            t_start=t_start,
            t_end=t_end,
            ego_radius=ego_radius,
        )

    # Build the aggregated graph (every partnership in window)
    overlap = (partnerships.start <= t_end) & (partnerships.end > t_start)
    a = partnerships.agent[overlap]
    b = partnerships.partner[overlap]

    node_universe: set[int] = set()
    for t in range(t_start, t_end + 1):
        node_universe |= active.active_at(t)

    G_agg = nx.Graph()
    G_agg.add_nodes_from(node_universe)
    for ai, bi in zip(a.tolist(), b.tolist(), strict=False):
        if int(ai) in node_universe and int(bi) in node_universe and ai != bi:
            G_agg.add_edge(int(ai), int(bi))

    n_rows = len(top_agents)

    with publication_style():
        fig = plt.figure(
            figsize=(8.5, 7.5 * n_rows + 1.2),
            constrained_layout=False,
        )
        gs = gridspec.GridSpec(
            n_rows,
            1,
            figure=fig,
            hspace=0.18,
            left=0.08,
            right=0.95,
            top=0.92,
            bottom=0.13,
        )

        for i, ego in enumerate(top_agents):
            ax = fig.add_subplot(gs[i, 0])

            if ego in G_agg:
                sub = nx.ego_graph(G_agg, ego, ego_radius)
            else:
                sub = nx.Graph()
                sub.add_node(ego)

            n_partners = max(sub.number_of_nodes() - 1, 0)
            edge_alpha = 0.5 if n_partners > 40 else 0.7
            edge_width = 0.6 if n_partners > 40 else 1.0

            _draw_ego_panel(
                ax,
                sub,
                shared_layouts[ego],
                node_attr,
                edge_color="#333333",
                edge_alpha=edge_alpha,
                edge_width=edge_width,
                age_label_max_nodes=30,
            )

            ax.set_ylabel(f"Agent {ego}", fontsize=10, labelpad=8)
            n_e = sub.number_of_edges()
            ax.text(
                0.98,
                0.98,
                f"{n_e} aggregated edges • {n_partners} unique partners",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=PALETTE.annotation,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                ),
            )

        _add_ego_legend(fig, y_anchor=0.03)
        fig.suptitle(
            f"Static-aggregate ego networks — all partnerships in [{t_start}, {t_end}]",
            fontsize=12,
            fontweight="bold",
            y=0.96,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# Helper: build node_attr from agent log


def build_node_attr(
    agent_log: pd.DataFrame,
    snapshot_t: int | None = None,
) -> dict[int, dict]:
    """Construct a node_attr dict from an agent log.

    Each agent gets a dict with Sex, Orientation, and Age. If
    ``snapshot_t`` is provided, Age is computed at that timestep
    (EntryAge + (snapshot_t - EntryTimestep)); otherwise the
    EntryAge is used unchanged.

    Convenience function — callers can also build node_attr themselves
    if they want richer attributes than these three.
    """
    out: dict[int, dict] = {}
    for _, row in agent_log.iterrows():
        aid = int(row["Agent"])
        entry_age = int(row["EntryAge"])
        entry_t = int(row["EntryTimestep"])
        age = entry_age + (snapshot_t - entry_t) if snapshot_t is not None else entry_age
        out[aid] = {
            "Sex": row["Sex"],
            "Orientation": row["Orientation"],
            "Age": age,
        }
    return out
