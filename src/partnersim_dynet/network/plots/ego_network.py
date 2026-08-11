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
  - Size:         local degree within the ego subgraph (sqrt-scaled)
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

# ---------------------------------------------------------------------------
# Identification: which agents had the most concurrent partnerships?
# ---------------------------------------------------------------------------


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
    real = real[real["EndTime"].notna()]
    if real.empty:
        return []

    real["StartTime"] = real["StartTime"].astype(int)
    real["EndTime"] = real["EndTime"].astype(int)

    rows = []
    for agent, grp in real.groupby("Agent"):
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

    extra = rank_df[~rank_df["Agent"].isin(concurrent["Agent"])]
    combined = pd.concat([concurrent, extra]).head(top_n)
    return combined["Agent"].tolist()


# ---------------------------------------------------------------------------
# Node layout: shared across the three plot variants
# ---------------------------------------------------------------------------


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
            seed=hash(ego) % (2**31 - 1),
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


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------


def _node_sizes(
    nodes: list[int],
    ego: int,
    local_deg: dict[int, int],
    local_max_deg: int,
    ego_size: float = 260,
    min_size: float = 25,
    max_size: float = 130,
) -> list[float]:
    """Return perceptually honest node sizes using sqrt-of-degree scaling.

    Area encodes degree rather than radius. The ego is always the
    largest node. Sizes calibrated for 2.4" panels at 300 dpi.
    """
    sizes = []
    for n in nodes:
        if n == ego:
            sizes.append(ego_size)
        else:
            frac = local_deg.get(n, 0) / local_max_deg
            sizes.append(min_size + (max_size - min_size) * np.sqrt(frac))
    return sizes


def _row_label(ax, ego: int) -> None:
    """Agent ID label inside the top-left corner of the panel."""
    ax.text(
        0.02,
        0.97,
        f"Agent {ego}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.80, edgecolor="none"),
    )


def _annotate_panel(ax, label: str) -> None:
    """Compact stats annotation at the bottom-right corner."""
    ax.text(
        0.98,
        0.02,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.5,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.80, edgecolor="none"),
    )


def _draw_ego_panel(
    ax,
    ego_subgraph: nx.Graph,
    layout: EgoLayout,
    node_attr: Mapping[int, Mapping] | None,
    ego: int,
    edge_color: str = "#222222",
    edge_alpha: float = 0.55,
    edge_width: float = 0.8,
    show_age_labels: bool = True,
    age_label_max_nodes: int = 30,
    custom_bounds: tuple | None = None,
) -> None:
    """Draw one ego subgraph onto an axes using the shared layout.

    ``node_attr`` maps each agent id to a dict with keys ``Sex``,
    ``Orientation``, ``Age``. Missing entries fall back to defaults.

    ``custom_bounds`` overrides ``layout.bounds`` — used by the active-
    snapshot variant to zoom in on the active subset of nodes.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    lo_x, hi_x, lo_y, hi_y = custom_bounds if custom_bounds else layout.bounds
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)

    nodes = list(ego_subgraph.nodes())
    pos = dict(layout.positions)

    rng = np.random.default_rng(hash(ego) % (2**31 - 1))
    for n in nodes:
        if n not in pos:
            pos[n] = np.array(
                [
                    rng.uniform(lo_x + 0.1, hi_x - 0.1),
                    rng.uniform(lo_y + 0.1, hi_y - 0.1),
                ]
            )

    if ego_subgraph.number_of_edges() > 0:
        nx.draw_networkx_edges(
            ego_subgraph,
            pos,
            ax=ax,
            edge_color=edge_color,
            alpha=edge_alpha,
            width=edge_width,
        )

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
        sizes = _node_sizes(group, ego, local_deg, local_max_deg)
        nx.draw_networkx_nodes(
            ego_subgraph,
            pos,
            nodelist=group,
            ax=ax,
            node_color=colors,
            node_size=sizes,
            node_shape=marker,
            alpha=1.0,
            linewidths=[2.0 if n == ego else 0.5 for n in group],
            edgecolors=["#FFD700" if n == ego else "#222222" for n in group],
        )

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
                fontsize=12,
                ha="center",
                va="bottom" if offset_y > 0 else "top",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"
                ),
            )


def _add_ego_legend(fig, y_anchor: float = 0.02) -> None:
    """Unified legend at the bottom of an ego-network figure."""
    handles = []

    handles.append(Line2D([0], [0], color="none", label="Sex:"))
    for sex in ("Female", "Male"):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=PALETTE.sex_shape(sex),
                color="#222222",
                linestyle="None",
                markersize=8,
                label=sex,
            )
        )

    handles.append(Line2D([0], [0], color="none", label=" "))

    handles.append(Line2D([0], [0], color="none", label="Orientation:"))
    for ori in ("Opposite-sex", "Same-sex", "Bisexual"):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PALETTE.orientation_color(ori),
                markersize=9,
                label=ori,
                markeredgecolor="#222222",
                markeredgewidth=0.5,
            )
        )

    handles.append(Line2D([0], [0], color="none", label=" "))

    handles.append(Line2D([0], [0], color="none", label="Node size:"))
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=13,
            markeredgecolor="#FFD700",
            markeredgewidth=2,
            label="ego",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=5,
            markeredgecolor="#222222",
            markeredgewidth=0.5,
            label="low degree",
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
            markeredgecolor="#222222",
            markeredgewidth=0.5,
            label="high degree",
        )
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, y_anchor),
        handletextpad=0.45,
        columnspacing=0.9,
    )


# ---------------------------------------------------------------------------
# Variant 1: dynamic — multi-panel evolution across timesteps
# ---------------------------------------------------------------------------


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
    _panel = 2.4

    with publication_style():
        fig = plt.figure(figsize=(_panel * n_cols, _panel * n_rows + 1.0))
        gs = gridspec.GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            hspace=0.08,
            wspace=0.04,
            left=0.04,
            right=0.98,
            top=0.90,
            bottom=0.11,
        )

        for i, ego in enumerate(top_agents):
            for j, t in enumerate(timesteps):
                ax = fig.add_subplot(gs[i, j])
                G_t = build_graph_at(t, partnerships, active)
                sub = nx.ego_graph(G_t, ego, ego_radius) if ego in G_t else _empty(ego)

                _draw_ego_panel(
                    ax,
                    sub,
                    shared_layouts[ego],
                    node_attr,
                    ego,
                    edge_color="#555555",
                    edge_alpha=0.55,
                    edge_width=0.8,
                )

                if i == 0:
                    ax.set_title(f"t = {t}", fontsize=12, fontweight="bold", pad=4)
                if j == 0:
                    _row_label(ax, ego)

                n_e = sub.number_of_edges()
                n_p = max(sub.number_of_nodes() - 1, 0)
                _annotate_panel(ax, f"{n_e}e · {n_p}p")

        _add_ego_legend(fig, y_anchor=0.02)
        fig.suptitle(
            "Dynamic ego networks — partnerships active at each timestep",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# Variant 2: active snapshot — one timestep, one column
# ---------------------------------------------------------------------------


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
    """Single-column ego networks at one snapshot timestep."""
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
        fig = plt.figure(figsize=(4.8, 2.4 * n_rows + 1.0))
        gs = gridspec.GridSpec(
            n_rows,
            1,
            figure=fig,
            hspace=0.10,
            left=0.04,
            right=0.98,
            top=0.90,
            bottom=0.11,
        )

        for i, ego in enumerate(top_agents):
            ax = fig.add_subplot(gs[i, 0])
            G_t = build_graph_at(snapshot_t, partnerships, active)
            sub = nx.ego_graph(G_t, ego, ego_radius) if ego in G_t else _empty(ego)

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
                ego,
                edge_color="#555555",
                edge_alpha=0.55,
                edge_width=0.8,
                custom_bounds=custom_bounds,
            )

            _row_label(ax, ego)
            n_e = sub.number_of_edges()
            n_p = max(sub.number_of_nodes() - 1, 0)
            _annotate_panel(ax, f"{n_e}e · {n_p}p")

        _add_ego_legend(fig, y_anchor=0.02)
        fig.suptitle(
            f"Active-snapshot ego networks — t = {snapshot_t}",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# Variant 3: static aggregate — every partnership in [t_start, t_end]
# ---------------------------------------------------------------------------


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
    """Static aggregate: every partnership in [t_start, t_end]."""
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
        fig = plt.figure(figsize=(9.0, 3.0 * n_rows + 1.0))
        gs = gridspec.GridSpec(
            n_rows,
            1,
            figure=fig,
            hspace=0.14,
            left=0.08,
            right=0.97,
            top=0.92,
            bottom=0.10,
        )

        for i, ego in enumerate(top_agents):
            ax = fig.add_subplot(gs[i, 0])
            sub = nx.ego_graph(G_agg, ego, ego_radius) if ego in G_agg else _empty(ego)

            n_partners = max(sub.number_of_nodes() - 1, 0)
            edge_alpha = 0.4 if n_partners > 40 else 0.55
            edge_width = 0.5 if n_partners > 40 else 0.8

            _draw_ego_panel(
                ax,
                sub,
                shared_layouts[ego],
                node_attr,
                ego,
                edge_color="#555555",
                edge_alpha=edge_alpha,
                edge_width=edge_width,
                age_label_max_nodes=30,
            )

            _row_label(ax, ego)
            n_e = sub.number_of_edges()
            _annotate_panel(ax, f"{n_e}e · {n_partners}p")

        _add_ego_legend(fig, y_anchor=0.02)
        fig.suptitle(
            f"Aggregate ego networks — all partnerships in [{t_start}, {t_end}]",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty(ego: int) -> nx.Graph:
    """Return a single-node graph for an ego absent from the network."""
    g = nx.Graph()
    g.add_node(ego)
    return g


# ---------------------------------------------------------------------------
# Helper: build node_attr from agent log
# ---------------------------------------------------------------------------


def build_node_attr(
    agent_log: pd.DataFrame,
    snapshot_t: int | None = None,
) -> dict[int, dict]:
    """Construct a node_attr dict from an agent log.

    Each agent gets a dict with Sex, Orientation, and Age. If
    ``snapshot_t`` is provided, Age is computed at that timestep
    (EntryAge + (snapshot_t - EntryTimestep)); otherwise EntryAge
    is used unchanged.
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
