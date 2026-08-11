"""Multi-hop ego network visualisations.

Three plot variants extend the 1-hop ego networks in ``ego_network.py``
to multiple hops out. Each shares a common k-hop subgraph computation
and a 150-node cap that prunes farthest nodes if the neighborhood
grows too large.

Variants:

- ``plot_ego_kshell_distance_coloured``: 4 levels (ego + 3 hops),
  coloured by distance from ego (navy → light blue). Sex maps to
  marker shape. No orientation colour — sacrificed for the distance
  encoding. Produces panels for snapshots t=0, 500, 1000, 1500 plus
  one aggregate panel.

- ``plot_ego_3hop_orientation_coloured``: 3 hops out, using the
  existing palette (orientation = colour, sex = shape). The ego is
  visually distinguished by a heavy outline. Same panel structure
  as above.

Both share the same top-N agent selection (most-concurrent) and the
same 150-node visibility cap.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

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
from partnersim_dynet.network.plots.ego_network import (
    identify_top_concurrent_agents,
)
from partnersim_dynet.network.plots.style import (
    PALETTE,
    OutputFormats,
    publication_style,
    save_figure,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ColorBrewer Blues-4: perceptually uniform, greyscale-safe, print-tested.
DISTANCE_COLOURS = {
    0: "#084594",  # darkest blue  — ego
    1: "#2171B5",  # medium-dark   — 1 hop
    2: "#6BAED6",  # medium-light  — 2 hops
    3: "#C6DBEF",  # lightest      — 3 hops
}

MAX_NODES_PER_PANEL = 100

# Journal column widths in inches (IEEE / Nature / PLOS convention)
_SINGLE_COL = 3.5
_DOUBLE_COL = 7.2

# ---------------------------------------------------------------------------
# Shared helper: build k-hop subgraph with size cap
# ---------------------------------------------------------------------------


def _khop_subgraph_capped(
    g: nx.Graph,
    ego: int,
    k: int,
    max_nodes: int = MAX_NODES_PER_PANEL,
) -> tuple[nx.Graph, dict[int, int]]:
    """Compute the k-hop ego subgraph, capped at ``max_nodes`` nodes.

    Uses BFS from the ego to compute geodesic distance. Includes all
    nodes at distance <= k. If the result exceeds ``max_nodes``, prunes
    by removing the farthest nodes first (and any orphans created by
    that removal).

    Parameters
    ----------
    g : nx.Graph
        The graph containing the ego.
    ego : int
        Agent ID at the centre.
    k : int
        Maximum number of hops out from ego.
    max_nodes : int
        Hard cap on the returned subgraph's node count.

    Returns
    -------
    tuple of (subgraph, distance_map)
        ``subgraph`` is an induced subgraph. ``distance_map`` maps each
        node ID to its hop count from the ego.
    """
    if ego not in g:
        sub = nx.Graph()
        sub.add_node(ego)
        return sub, {ego: 0}

    # BFS to compute distances up to k
    distance_map: dict[int, int] = {ego: 0}
    frontier = {ego}
    for hop in range(1, k + 1):
        next_frontier: set[int] = set()
        for node in frontier:
            for neighbor in g.neighbors(node):
                if neighbor not in distance_map:
                    distance_map[neighbor] = hop
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    # Prune if too many nodes (keep closest first)
    if len(distance_map) > max_nodes:
        sorted_nodes = sorted(distance_map.items(), key=lambda x: x[1])
        kept_nodes = {node for node, _ in sorted_nodes[:max_nodes]}
        distance_map = {n: d for n, d in distance_map.items() if n in kept_nodes}

    sub = g.subgraph(distance_map.keys()).copy()
    return sub, distance_map


# ---------------------------------------------------------------------------
# Shared helper: build aggregate graph for [t_start, t_end]
# ---------------------------------------------------------------------------


def _build_aggregate_graph(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    t_start: int,
    t_end: int,
) -> nx.Graph:
    """Build the aggregate undirected graph of every partnership active
    in ``[t_start, t_end]``.
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
# Shared helper: compute layout for a k-hop subgraph
# ---------------------------------------------------------------------------


def _compute_khop_layout(
    sub: nx.Graph,
    ego: int,
    distance_map: dict[int, int],
) -> dict[int, np.ndarray]:
    """Spring layout with the ego fixed at origin."""
    if sub.number_of_nodes() == 1:
        return {ego: np.array([0.0, 0.0])}

    # return nx.spring_layout(
    #     sub,
    #     seed=hash(ego) % (2**31 - 1),
    #     k=2.5 / np.sqrt(len(sub)),
    #     iterations=500,
    #     scale=1.5,
    #     fixed=[ego],
    #     pos={ego: np.array([0.0, 0.0])},
    # )
    return nx.spring_layout(
        sub,
        seed=hash(ego) % (2**31 - 1),
        k=2.5 / np.sqrt(len(sub)),
        iterations=500,
        scale=1.5,
        fixed=[ego],
        pos={ego: np.array([0.0, 0.0])},
    )


# ---------------------------------------------------------------------------
# Shared helper: node attribute lookup
# ---------------------------------------------------------------------------


def _build_node_attr_simple(agent_log: pd.DataFrame) -> dict[int, dict]:
    """Build {agent_id: {Sex, Orientation}} from the agent log."""
    out: dict[int, dict] = {}
    for _, row in agent_log.iterrows():
        out[int(row["Agent"])] = {
            "Sex": row["Sex"],
            "Orientation": row["Orientation"],
        }
    return out


# ---------------------------------------------------------------------------
# Shared helper: node size using sqrt-of-degree scaling
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
    """Return perceptually honest node sizes.

    Uses sqrt scaling so that area encodes degree rather than radius.
    Sizes are calibrated for 2.4" panels at 300 dpi: the ego is clearly
    dominant but not overwhelming, and leaf nodes remain visible.
    """
    sizes = []
    for n in nodes:
        if n == ego:
            sizes.append(ego_size)
        else:
            frac = local_deg.get(n, 0) / local_max_deg
            sizes.append(min_size + (max_size - min_size) * np.sqrt(frac))
    return sizes


# ---------------------------------------------------------------------------
# Shared helper: annotate panel with node/edge count
# ---------------------------------------------------------------------------


def _annotate_panel(ax, n_nodes: int, n_edges: int) -> None:
    """Add a compact node/edge count to the bottom-right corner.

    Placed bottom-right so it does not overlap the agent label in the
    top-left corner.
    """
    ax.text(
        0.98,
        0.02,
        f"{n_nodes}n · {n_edges}e",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="#444444",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.80,
            edgecolor="none",
        ),
    )


# ---------------------------------------------------------------------------
# Shared helper: row label (replaces set_ylabel)
# ---------------------------------------------------------------------------


def _row_label(ax, ego: int) -> None:
    """Write the agent ID as a small, unrotated label above the leftmost panel.

    Placed just inside the top-left corner of the axes so it is always
    anchored to the panel regardless of figure layout. Unrotated text
    is easier to read and avoids the size/placement issues caused by
    rotated ax.transAxes labels when panel heights vary.
    """
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


# ---------------------------------------------------------------------------
# Shared helper: configure axes for a network panel
# ---------------------------------------------------------------------------


def _configure_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    # Do NOT use adjustable="datalim" — it makes each panel a different size.
    # Fixed axes boxes keep the grid uniform; padding in _set_ax_limits fills space.


# ---------------------------------------------------------------------------
# Shared helper: set axis limits with consistent padding
# ---------------------------------------------------------------------------


def _set_ax_limits(ax, pos: dict, pad: float = 0.35) -> None:
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)


# ---------------------------------------------------------------------------
# Drawing helper: distance-coloured panel
# ---------------------------------------------------------------------------


def _draw_distance_coloured_panel(
    ax,
    sub: nx.Graph,
    distance_map: dict[int, int],
    layout: dict[int, np.ndarray],
    node_attr: Mapping[int, Mapping],
    ego: int,
) -> None:
    """Draw one k-shell ego subgraph with distance-coloured nodes."""
    _configure_ax(ax)

    if sub.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    # Fallback positions for any node missing from layout
    pos = dict(layout)
    rng = np.random.default_rng(hash(ego) % (2**31 - 1))
    for n in sub.nodes():
        if n not in pos:
            pos[n] = rng.uniform(-1, 1, size=2)

    _set_ax_limits(ax, pos)

    if sub.number_of_edges() > 0:
        nx.draw_networkx_edges(
            sub,
            pos,
            ax=ax,
            edge_color="#555555",
            alpha=0.55,
            width=0.8,
        )

    local_deg = dict(sub.degree())
    local_max_deg = max(max(local_deg.values()) if local_deg else 1, 1)

    for sex in ("Male", "Female"):
        marker = PALETTE.sex_shape(sex)
        group = [n for n in sub.nodes() if node_attr.get(n, {}).get("Sex") == sex]
        if not group:
            continue

        colors = [DISTANCE_COLOURS.get(distance_map[n], "#888888") for n in group]
        sizes = _node_sizes(group, ego, local_deg, local_max_deg)

        nx.draw_networkx_nodes(
            sub,
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


# ---------------------------------------------------------------------------
# Legend: distance-coloured
# ---------------------------------------------------------------------------


def _add_distance_legend(fig, y_anchor: float = 0.03) -> None:
    handles = []

    # Distance colour swatch group
    handles.append(Line2D([0], [0], color="none", label="Distance:"))
    for _d, c in DISTANCE_COLOURS.items():
        label = "ego" if _d == 0 else f"{_d} hop{'s' if _d > 1 else ''}"
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=c,
                markersize=9,
                label=label,
                markeredgecolor="#FFD700" if _d == 0 else "#222222",
                markeredgewidth=2.0 if _d == 0 else 0.5,
            )
        )

    # Spacer
    handles.append(Line2D([0], [0], color="none", label=" "))

    # Sex group
    handles.append(Line2D([0], [0], color="none", label="Sex:"))
    for sex in ("Male", "Female"):
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
# Drawing helper: orientation-coloured panel
# ---------------------------------------------------------------------------


def _draw_orientation_coloured_panel(
    ax,
    sub: nx.Graph,
    layout: dict[int, np.ndarray],
    node_attr: Mapping[int, Mapping],
    ego: int,
    distance_map: dict[int, int] | None = None,
    fade_with_distance: bool = False,
    ego_size: float = 260,
    min_size: float = 25,
    max_size: float = 130,
) -> None:
    """Draw one k-hop ego subgraph with orientation-coloured nodes.

    Parameters
    ----------
    distance_map : dict[int, int] or None
        Required when ``fade_with_distance=True``.
    fade_with_distance : bool
        If True, node alpha decreases with hop distance from ego.
    ego_size, min_size, max_size : float
        Node size parameters passed to ``_node_sizes``. Use larger
        values for wide/tall aggregate panels.
    """
    _configure_ax(ax)

    if sub.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    pos = dict(layout)
    rng = np.random.default_rng(hash(ego) % (2**31 - 1))
    for n in sub.nodes():
        if n not in pos:
            pos[n] = rng.uniform(-1, 1, size=2)

    _set_ax_limits(ax, pos)

    if sub.number_of_edges() > 0:
        nx.draw_networkx_edges(
            sub,
            pos,
            ax=ax,
            edge_color="#555555",
            alpha=0.55,
            width=0.8,
        )

    local_deg = dict(sub.degree())
    local_max_deg = max(max(local_deg.values()) if local_deg else 1, 1)

    def _alpha(n: int) -> float:
        if not fade_with_distance or distance_map is None:
            return 1.0
        return max(0.4, 1.0 - 0.2 * distance_map.get(n, 0))

    for sex in ("Male", "Female"):
        marker = PALETTE.sex_shape(sex)
        sex_nodes = [n for n in sub.nodes() if node_attr.get(n, {}).get("Sex") == sex]
        if not sex_nodes:
            continue

        if not fade_with_distance or distance_map is None:
            colors = [
                PALETTE.orientation_color(node_attr.get(n, {}).get("Orientation", ""))
                for n in sex_nodes
            ]
            sizes = _node_sizes(
                sex_nodes,
                ego,
                local_deg,
                local_max_deg,
                ego_size=ego_size,
                min_size=min_size,
                max_size=max_size,
            )
            nx.draw_networkx_nodes(
                sub,
                pos,
                nodelist=sex_nodes,
                ax=ax,
                node_color=colors,
                node_size=sizes,
                node_shape=marker,
                alpha=1.0,
                linewidths=[2.0 if n == ego else 0.5 for n in sex_nodes],
                edgecolors=["#FFD700" if n == ego else "#222222" for n in sex_nodes],
            )
        else:
            # Group by distance so each distance level gets its own alpha call
            distance_groups: dict[int, list[int]] = {}
            for n in sex_nodes:
                d = distance_map.get(n, 0)
                distance_groups.setdefault(d, []).append(n)

            for _d, group in sorted(distance_groups.items()):
                colors = [
                    PALETTE.orientation_color(node_attr.get(n, {}).get("Orientation", ""))
                    for n in group
                ]
                sizes = _node_sizes(
                    group,
                    ego,
                    local_deg,
                    local_max_deg,
                    ego_size=ego_size,
                    min_size=min_size,
                    max_size=max_size,
                )
                nx.draw_networkx_nodes(
                    sub,
                    pos,
                    nodelist=group,
                    ax=ax,
                    node_color=colors,
                    node_size=sizes,
                    node_shape=marker,
                    alpha=_alpha(group[0]),
                    linewidths=[2.0 if n == ego else 0.5 for n in group],
                    edgecolors=["#FFD700" if n == ego else "#222222" for n in group],
                )


# ---------------------------------------------------------------------------
# Legend: orientation-coloured
# ---------------------------------------------------------------------------


def _add_orientation_legend(fig, y_anchor: float = 0.03) -> None:
    handles = []

    # Sex
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

    # Spacer
    handles.append(Line2D([0], [0], color="none", label=" "))

    # Orientation
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

    # Spacer
    handles.append(Line2D([0], [0], color="none", label=" "))

    # Marker size encoding
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
# GridSpec factory helpers
# ---------------------------------------------------------------------------


def _snapshot_gridspec(fig, n_rows: int, n_cols: int) -> gridspec.GridSpec:
    return gridspec.GridSpec(
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


def _aggregate_gridspec(fig, n_rows: int) -> gridspec.GridSpec:
    return gridspec.GridSpec(
        n_rows,
        1,
        figure=fig,
        hspace=0.14,
        left=0.08,
        right=0.97,
        top=0.92,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Snapshot plot — distance-coloured
# ---------------------------------------------------------------------------


def plot_ego_kshell_snapshots(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    output_dir: str,
    top_n: int,
    timesteps: list[int],
    k_hops: int = 3,
    max_nodes: int = MAX_NODES_PER_PANEL,
    filename_stem: str = "ego_kshell_snapshots",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """K-shell ego plots across snapshot timesteps. Distance-coloured.

    Rows = top-N agents, columns = snapshot timesteps. Each panel shows
    the ego subgraph with currently active partnerships, coloured by
    distance from ego.

    Parameters
    ----------
    partnerships_df : DataFrame
    partnerships : PartnershipArrays
    active : ActiveIntervals
    agent_log : DataFrame
    output_dir : str
    top_n : int
    timesteps : list of int
        Snapshot columns. With 4 entries the figure is 4 columns wide.
    k_hops : int
        Hops from ego. Default 3 → 4 distance levels (ego + 3).
    max_nodes : int
    filename_stem : str
    formats : OutputFormats
    """
    if not timesteps:
        raise ValueError("timesteps must be a non-empty list")

    top_agents = identify_top_concurrent_agents(partnerships_df, top_n=top_n)
    if not top_agents:
        return []

    node_attr = _build_node_attr_simple(agent_log)
    n_rows = len(top_agents)
    n_cols = len(timesteps)

    with publication_style():
        # Square panels: 2.4" per panel, legend strip at bottom
        _panel = 2.4
        fig = plt.figure(figsize=(_panel * n_cols, _panel * n_rows + 1.0))
        gs = _snapshot_gridspec(fig, n_rows, n_cols)

        for i, ego in enumerate(top_agents):
            for j, t in enumerate(timesteps):
                ax = fig.add_subplot(gs[i, j])
                g_t = build_graph_at(t, partnerships, active)
                sub, dist_map = _khop_subgraph_capped(g_t, ego, k_hops, max_nodes)
                layout = _compute_khop_layout(sub, ego, dist_map)
                _draw_distance_coloured_panel(ax, sub, dist_map, layout, node_attr, ego)
                _annotate_panel(ax, sub.number_of_nodes(), sub.number_of_edges())
                if j == 0:
                    _row_label(ax, ego)
                if i == 0:
                    ax.set_title(f"t = {t}", fontsize=12, fontweight="bold", pad=4)

        _add_distance_legend(fig, y_anchor=0.02)
        fig.suptitle(
            f"Snapshot ego networks ({k_hops + 1}-shell, distance-coloured, cap {max_nodes} nodes)",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# Aggregate plot — distance-coloured
# ---------------------------------------------------------------------------


def plot_ego_kshell_aggregate(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    output_dir: str,
    top_n: int,
    total_timesteps: int,
    t_start: int = 1,
    t_end: int | None = None,
    k_hops: int = 3,
    max_nodes: int = MAX_NODES_PER_PANEL,
    filename_stem: str = "ego_kshell_aggregate",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """K-shell ego plots aggregated over a time window. Distance-coloured.

    One wide rectangular panel per top-N agent showing the union of all
    partnerships in ``[t_start, t_end]``.

    Parameters
    ----------
    t_start : int
        Window start. Default 1 (simulation beginning).
    t_end : int or None
        Window end. Default ``total_timesteps`` (simulation end).
    """
    if t_end is None:
        t_end = total_timesteps

    top_agents = identify_top_concurrent_agents(partnerships_df, top_n=top_n)
    if not top_agents:
        return []

    node_attr = _build_node_attr_simple(agent_log)
    g_agg = _build_aggregate_graph(partnerships, active, t_start, t_end)
    n_rows = len(top_agents)

    with publication_style():
        # Aggregate: wide single strip per agent, 3.0" tall each
        fig = plt.figure(figsize=(9.0, 3.0 * n_rows + 1.0))
        gs = _aggregate_gridspec(fig, n_rows)

        for i, ego in enumerate(top_agents):
            ax = fig.add_subplot(gs[i, 0])
            sub, dist_map = _khop_subgraph_capped(g_agg, ego, k_hops, max_nodes)
            layout = _compute_khop_layout(sub, ego, dist_map)
            _draw_distance_coloured_panel(ax, sub, dist_map, layout, node_attr, ego)
            _annotate_panel(ax, sub.number_of_nodes(), sub.number_of_edges())
            _row_label(ax, ego)

        _add_distance_legend(fig, y_anchor=0.02)
        fig.suptitle(
            f"Aggregate ego networks ({k_hops + 1}-shell, distance-coloured) — "
            f"partnerships [{t_start}, {t_end}]",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# Snapshot plot — orientation-coloured
# ---------------------------------------------------------------------------


def plot_ego_3hop_snapshots(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    output_dir: str,
    top_n: int,
    timesteps: list[int],
    k_hops: int = 3,
    max_nodes: int = MAX_NODES_PER_PANEL,
    filename_stem: str = "ego_3hop_snapshots",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """3-hop ego plots across snapshot timesteps. Orientation-coloured.

    Same panel layout as ``plot_ego_kshell_snapshots`` but coloured by
    orientation using the existing palette. The ego is distinguished
    by a thick gold outline.
    """
    if not timesteps:
        raise ValueError("timesteps must be a non-empty list")

    top_agents = identify_top_concurrent_agents(partnerships_df, top_n=top_n)
    if not top_agents:
        return []

    node_attr = _build_node_attr_simple(agent_log)
    n_rows = len(top_agents)
    n_cols = len(timesteps)

    with publication_style():
        _panel = 2.4
        fig = plt.figure(figsize=(_panel * n_cols, _panel * n_rows + 1.0))
        gs = _snapshot_gridspec(fig, n_rows, n_cols)

        for i, ego in enumerate(top_agents):
            for j, t in enumerate(timesteps):
                ax = fig.add_subplot(gs[i, j])
                g_t = build_graph_at(t, partnerships, active)
                sub, dist_map = _khop_subgraph_capped(g_t, ego, k_hops, max_nodes)
                layout = _compute_khop_layout(sub, ego, dist_map)
                _draw_orientation_coloured_panel(ax, sub, layout, node_attr, ego)
                _annotate_panel(ax, sub.number_of_nodes(), sub.number_of_edges())

                if i == 0:
                    ax.set_title(f"t = {t}", fontsize=12, fontweight="bold", pad=4)
                if j == 0:
                    _row_label(ax, ego)

        _add_orientation_legend(fig, y_anchor=0.02)
        fig.suptitle(
            f"Snapshot ego networks ({k_hops}-hop, orientation-coloured, cap {max_nodes} nodes)",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# Aggregate plot — orientation-coloured
# ---------------------------------------------------------------------------


def plot_ego_3hop_aggregate(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    output_dir: str,
    top_n: int,
    total_timesteps: int,
    t_start: int = 1,
    t_end: int | None = None,
    k_hops: int = 3,
    max_nodes: int = MAX_NODES_PER_PANEL,
    filename_stem: str = "ego_3hop_aggregate",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """3-hop ego plots aggregated over a time window. Orientation-coloured.

    One wide rectangular panel per top-N agent showing the union of all
    partnerships in ``[t_start, t_end]``, using the orientation/sex
    palette.
    """
    if t_end is None:
        t_end = total_timesteps

    top_agents = identify_top_concurrent_agents(partnerships_df, top_n=top_n)
    if not top_agents:
        return []

    node_attr = _build_node_attr_simple(agent_log)
    g_agg = _build_aggregate_graph(partnerships, active, t_start, t_end)
    n_rows = len(top_agents)

    with publication_style():
        # Tall panels give the spring layout room to spread; rotated ylabel
        # works well at this aspect ratio and matches the original style.
        fig = plt.figure(figsize=(12.0, 4.5 * n_rows + 1.5))
        gs = gridspec.GridSpec(
            n_rows,
            1,
            figure=fig,
            hspace=0.18,
            left=0.06,
            right=0.97,
            top=0.92,
            bottom=0.10,
        )

        for i, ego in enumerate(top_agents):
            ax = fig.add_subplot(gs[i, 0])
            sub, dist_map = _khop_subgraph_capped(g_agg, ego, k_hops, max_nodes)
            layout = _compute_khop_layout(sub, ego, dist_map)
            _draw_orientation_coloured_panel(
                ax,
                sub,
                layout,
                node_attr,
                ego,
                distance_map=dist_map,
                fade_with_distance=False,
                ego_size=520,
                min_size=60,
                max_size=320,
            )

            ax.set_ylabel(f"Agent {ego}", fontsize=16, fontweight="bold", labelpad=10)
            ax.text(
                0.99,
                0.98,
                f"{sub.number_of_nodes()} nodes • {sub.number_of_edges()} edges",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=16,
                color=PALETTE.annotation,
                bbox=dict(
                    boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"
                ),
            )

        _add_orientation_legend(fig, y_anchor=0.03)
        fig.suptitle(
            f"Aggregate ego networks ({k_hops}-hop, orientation-coloured) — "
            f"all partnerships in [{t_start}, {t_end}]",
            fontsize=16,
            fontweight="bold",
            y=0.97,
        )

        output_base = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, output_base, formats)
        plt.close(fig)

    return written
