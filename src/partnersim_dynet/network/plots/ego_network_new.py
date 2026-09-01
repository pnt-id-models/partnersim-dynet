"""Ego-network plotting script/

Consolidated module: 1-hop ego networks (dynamic / snapshot / static
aggregate / multi-window aggregate) plus multi-hop k-shell and
orientation-coloured variants.

Visual encoding (all panels):
  - Marker shape: sex (circle = Male, square = Female)
  - Color:        orientation (PALETTE)
  - Size:         fixed (ego larger than partners; no degree scaling)
  - No age labels, no per-panel node/edge count annotations.
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

from partnersim_dynet.config import age_to_group
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

# Constant node sizes for ego vs. partner nodes

EGO_NODE_SIZE = 150
PARTNER_NODE_SIZE = 60


# Colourblind-safe qualitative palette chosen for maximum
# separation between the three orientation categories at small marker
# sizes and in greyscale print.
ORIENTATION_COLORS_ACCESSIBLE = {
    "Opposite-sex": "#2081D6",  # blue
    "Same-sex": "#109E00",  # bluish-green
    "Bisexual": "#C000D5",  # pink
}

# Standard working-age group boundaries, matching PartnershipGenerator's
# initial-cohort age groups (16-24 ... 65-74).
AGE_GROUP_BOUNDS: dict[str, tuple[int, int]] = {
    "16-24": (16, 24),
    "25-34": (25, 34),
    "35-44": (35, 44),
    "45-54": (45, 54),
    "55-64": (55, 64),
    "65-74": (65, 74),
}
MAX_NODES_PER_PANEL = 25

DEFAULT_SEXES = ("Male", "Female")
DEFAULT_ORIENTATIONS = ("Opposite-sex", "Same-sex", "Bisexual")

# Shared legend style for all ego-network plots, with larger font and marker sizes
LEGEND_STYLE_LARGE: dict = dict(
    ncol=8,
    fontsize=14,
    legend_marker_size=17,
    handletextpad=0.35,
    columnspacing=0.8,
    labelspacing=0.15,
    compact=True,
)
# specs = [
#     ("Male", "Bisexual", "25-34"),
#     ("Female", "Same-sex", "35-44"),
#     ("Male", "Opposite-sex", "16-24"),
# ]

specs = [
    ("Male", "Bisexual"),
    ("Female", "Same-sex"),
    ("Male", "Opposite-sex"),
]


def identify_agents_by_spec(
    partnerships: pd.DataFrame,
    node_attr: Mapping[int, Mapping],
    specs: list[tuple[str, str]],
    eligible_agents: set[int] | None = None,
) -> list[int | None]:
    """Return one agent per (sex, orientation, age_group) spec.

    For each spec, ranks eligible candidates by (MaxSimultaneous,
    TotalPartnerships) — same ranking as identify_top_concurrent_agents —
    and picks the top-ranked agent matching that exact combination.

    Parameters
    ----------
    partnerships : pd.DataFrame
        Partnership event rows with Agent/PartnerAgent/StartTime/EndTime.
    node_attr : Mapping[int, Mapping]
        {agent_id: {"Sex": ..., "Orientation": ..., "Age": ...}}, as
        produced by build_node_attr(agent_log, snapshot_t=...). AgeGroup
        is derived here from Age via age_to_group
    specs : list of (sex, orientation) tuples
        One tuple per desired agent, e.g.
        [("Male", "Bisexual"),
         ("Female", "Same-sex"),
         ("Male", "Opposite-sex")].
    eligible_agents : set[int] | None
        If given, restricts the candidate pool to these agent IDs (e.g.
        active at a reference timestep) before matching against specs.

    Returns
    -------
    list of int or None, same length and order as `specs`.
    """

    # Real partnerships only (ignore missing PartnerAgent or StartTime/EndTime)
    real = partnerships[
        partnerships["PartnerAgent"].notna() & partnerships["StartTime"].notna()
    ].copy()
    real = real[real["EndTime"].notna()]
    if real.empty:
        return [None] * len(specs)

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
            {"Agent": int(agent), "MaxSimultaneous": max_simul, "TotalPartnerships": len(grp)}
        )

    rank_df = (
        pd.DataFrame(rows)
        .sort_values(["MaxSimultaneous", "TotalPartnerships"], ascending=False)
        .reset_index(drop=True)
    )

    if eligible_agents is not None:
        rank_df = rank_df[rank_df["Agent"].isin(eligible_agents)]

    rank_df["Sex"] = rank_df["Agent"].map(lambda a: node_attr.get(a, {}).get("Sex"))
    rank_df["Orientation"] = rank_df["Agent"].map(lambda a: node_attr.get(a, {}).get("Orientation"))
    rank_df["AgeGroup"] = rank_df["Agent"].map(
        lambda a: age_to_group(node_attr[a]["Age"]) if a in node_attr else None
    )
    rank_df = rank_df.dropna(subset=["Sex", "Orientation"])

    selected: list[int | None] = []
    claimed: set[int] = set()

    for sex, orientation in specs:
        pool = rank_df[
            (rank_df["Sex"] == sex)
            & (rank_df["Orientation"] == orientation)
            & (~rank_df["Agent"].isin(claimed))
        ]
        if pool.empty:
            selected.append(None)
            continue
        agent = int(pool.iloc[0]["Agent"])
        selected.append(agent)
        claimed.add(agent)

    return selected


# Identification: top concurrent agents (UNUSED)
def identify_top_concurrent_agents(
    partnerships: pd.DataFrame,
    node_attr: Mapping[int, Mapping],
    top_n: int,
    sexes: tuple[str, ...] = DEFAULT_SEXES,
    orientations: tuple[str, ...] = DEFAULT_ORIENTATIONS,
    eligible_agents: set[int] | None = None,
) -> list[int]:
    """Return top_n agent IDs by max simultaneous partnerships,

    Parameters
    ----------
    partnerships : pd.DataFrame
        Partnership event rows with Agent/PartnerAgent/StartTime/EndTime.
    node_attr : Mapping[int, Mapping]
        {agent_id: {"Sex": ..., "Orientation": ..., ...}}, as produced by
        build_node_attr(agent_log). Agents missing from this mapping are
        excluded from consideration.
    top_n : int
        Total number of agents to return.
    sexes : tuple[str, ...]
        Sex categories to balance across. Defaults to ("Male", "Female").
    orientations : tuple[str, ...]
        Orientation categories to guarantee representation for, within
        each sex's quota. Defaults to ("Opposite-sex", "Same-sex",
        "Bisexual").
    eligible_agents : set[int] | None
        If given, restricts the candidate pool to these agent IDs before
        ranking/quota selection — e.g. agents of a specific age or age
        group at a reference timestep.
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
            {"Agent": int(agent), "MaxSimultaneous": max_simul, "TotalPartnerships": len(grp)}
        )

    rank_df = (
        pd.DataFrame(rows)
        .sort_values(["MaxSimultaneous", "TotalPartnerships"], ascending=False)
        .reset_index(drop=True)
    )

    if eligible_agents is not None:
        rank_df = rank_df[rank_df["Agent"].isin(eligible_agents)]
        if rank_df.empty:
            return []

    # Attach demographics; drop agents we have no Sex/Orientation for.
    rank_df["Sex"] = rank_df["Agent"].map(lambda a: node_attr.get(a, {}).get("Sex"))
    rank_df["Orientation"] = rank_df["Agent"].map(lambda a: node_attr.get(a, {}).get("Orientation"))
    rank_df = rank_df.dropna(subset=["Sex", "Orientation"])
    if rank_df.empty:
        return []

    # Split top_n as evenly as possible across sexes (extra slot, if any,
    # goes to the first entry in `sexes`).
    base, rem = divmod(top_n, len(sexes))
    quotas = {sex: base + (1 if i < rem else 0) for i, sex in enumerate(sexes)}

    selected: list[int] = []

    for sex in sexes:
        quota = quotas[sex]
        if quota <= 0:
            continue
        pool = rank_df[rank_df["Sex"] == sex]
        if pool.empty:
            continue

        sex_selected: list[int] = []

        # Step 1: guarantee one agent per orientation (best-ranked within
        # that orientation), independent of partner count.
        for ori in orientations:
            if len(sex_selected) >= quota:
                break
            ori_pool = pool[(pool["Orientation"] == ori) & (~pool["Agent"].isin(sex_selected))]
            if not ori_pool.empty:
                sex_selected.append(int(ori_pool.iloc[0]["Agent"]))

        # Step 2: fill remaining slots in this sex's quota by overall rank.
        if len(sex_selected) < quota:
            remaining = pool[~pool["Agent"].isin(sex_selected)]
            needed = quota - len(sex_selected)
            sex_selected.extend(int(a) for a in remaining.head(needed)["Agent"])

        selected.extend(sex_selected[:quota])

    # Step 3: if some sex/orientation combos didn't have enough agents,
    # backfill from the overall pool (ignoring quotas) so we still return
    # top_n agents when possible.
    if len(selected) < top_n:
        remaining = rank_df[~rank_df["Agent"].isin(selected)]
        needed = top_n - len(selected)
        selected.extend(int(a) for a in remaining.head(needed)["Agent"])

    return selected[:top_n]


# Node lookup: demographics (Sex, Orientation, Age)


def build_node_attr(agent_log: pd.DataFrame, snapshot_t: int | None = None) -> dict[int, dict]:
    """Build {agent_id: {Sex, Orientation, Age}} from the agent log."""
    out: dict[int, dict] = {}
    for _, row in agent_log.iterrows():
        aid = int(row["Agent"])
        entry_age = int(row["EntryAge"])
        entry_t = int(row["EntryTimestep"])
        age = entry_age + (snapshot_t - entry_t) if snapshot_t is not None else entry_age
        out[aid] = {"Sex": row["Sex"], "Orientation": row["Orientation"], "Age": age}
    return out


# Aggregate graph construction (used by static-aggregate and multi-window-aggregate plots)
def _build_aggregate_graph(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    t_start: int,
    t_end: int,
) -> nx.Graph:
    """Union of every partnership active at any point in [t_start, t_end]."""
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


# k-hop subgraph extraction (used by multi-hop k-shell plots)
def _khop_subgraph_capped(
    g: nx.Graph,
    ego: int,
    k: int,
    max_nodes: int,
) -> tuple[nx.Graph, dict[int, int]]:
    """Breadth-first search for k hops from ego, capped at max_nodes total.

    Returns (subgraph, distance_map).
    """
    if ego not in g:
        return _empty(ego), {ego: 0}

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

    if len(distance_map) > max_nodes:
        # keep closest nodes first
        keep = dict(sorted(distance_map.items(), key=lambda kv: kv[1])[:max_nodes])
        distance_map = keep

    sub = g.subgraph(distance_map.keys()).copy()
    return sub, distance_map


# Shared layout computation for all ego agents (used by static-aggregate and multi-window-aggregate plots)
@dataclass(frozen=True)
class EgoLayout:
    positions: dict[int, np.ndarray]
    bounds: tuple[float, float, float, float]


def build_shared_ego_layouts(
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    ego_agents: list[int],
    t_start: int,
    t_end: int,
    ego_radius: int = 1,
) -> dict[int, EgoLayout]:
    """One positions+bounds layout per ego agent, computed on the aggregate
    graph over [t_start, t_end] so it's a superset of every snapshot."""
    g_agg = _build_aggregate_graph(partnerships, active, t_start, t_end)

    layouts: dict[int, EgoLayout] = {}
    for ego in ego_agents:
        sub = nx.ego_graph(g_agg, ego, ego_radius) if ego in g_agg else _empty(ego)
        layouts[ego] = _layout_for_subgraph(sub, ego)
    return layouts


def _layout_for_subgraph(sub: nx.Graph, ego: int, pad: float = 0.3) -> EgoLayout:
    if sub.number_of_nodes() <= 1:
        return EgoLayout(positions={ego: np.array([0.0, 0.0])}, bounds=(-1.0, 1.0, -1.0, 1.0))

    positions = nx.spring_layout(
        sub,
        seed=hash(ego) % (2**31 - 1),
        k=3.0 / np.sqrt(len(sub)),
        iterations=400,
        fixed=[ego],
        pos={ego: np.array([0.0, 0.0])},
    )
    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])
    bounds = (
        float(xs.min() - pad),
        float(xs.max() + pad),
        float(ys.min() - pad),
        float(ys.max() + pad),
    )
    return EgoLayout(positions=positions, bounds=bounds)


def _empty(ego: int) -> nx.Graph:
    g = nx.Graph()
    g.add_node(ego)
    return g


# Drawing helpers (used by all ego-network plots)
def _configure_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


# def _row_label(ax, ego: int) -> None:
#     ax.text(
#         0.02, 0.97, f"Agent {ego}", transform=ax.transAxes, ha="left", va="top",
#         fontsize=12, fontweight="bold", color="#111111",
#         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.80, edgecolor="none"),
#     )
def _row_label(ax, ego: int, age: int | None = None, extra_offset: bool = False) -> None:
    """Row label placed ABOVE the axes (like a title), never overlapping
    plotted nodes regardless of layout. `extra_offset` lifts it slightly
    higher for row 0, so it doesn't collide with the column header
    ("t = ...") that also sits above that same row's axes.
    """
    label = f"Agent {ego}" if age is None else f"Agent {ego}, age {age}"
    y = 1.5 if extra_offset else 1.04
    ax.text(
        0.0,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        zorder=10,
    )


# Draw one ego subgraph. No age labels, no node/edge count annotation.
def _draw_ego_panel(
    ax,
    sub: nx.Graph,
    layout: EgoLayout,
    node_attr: Mapping[int, Mapping],
    ego: int,
    color_by: str = "orientation",  # "orientation" or "distance"
    edge_color: str = "#555555",
    edge_alpha: float = 0.55,
    edge_width: float = 0.8,
    custom_bounds: tuple | None = None,
    ego_node_size: int = EGO_NODE_SIZE,
    partner_node_size: int = PARTNER_NODE_SIZE,
    orientation_colors: dict[str, str] | None = None,
    ego_highlight_color: str = "#E69F00",
) -> None:
    """Draw one ego subgraph.
    Ego is distinguished by an orange outline.
    """
    _configure_ax(ax)

    if sub.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    lo_x, hi_x, lo_y, hi_y = custom_bounds if custom_bounds else layout.bounds
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)

    pos = dict(layout.positions)
    rng = np.random.default_rng(hash(ego) % (2**31 - 1))
    for n in sub.nodes():
        if n not in pos:
            pos[n] = np.array(
                [rng.uniform(lo_x + 0.1, hi_x - 0.1), rng.uniform(lo_y + 0.1, hi_y - 0.1)]
            )

    if sub.number_of_edges() > 0:
        edge_collection = nx.draw_networkx_edges(
            sub, pos, ax=ax, edge_color=edge_color, alpha=edge_alpha, width=edge_width
        )
        if edge_collection is not None:
            if isinstance(edge_collection, list):
                for artist in edge_collection:
                    artist.set_clip_on(False)
            else:
                edge_collection.set_clip_on(False)

    def _attr(n):
        return node_attr.get(n, {}) if node_attr else {}

    colors_lookup = orientation_colors or ORIENTATION_COLORS_ACCESSIBLE

    for sex in ("Male", "Female"):
        marker = PALETTE.sex_shape(sex)
        group = [n for n in sub.nodes() if _attr(n).get("Sex") == sex]
        if not group:
            continue

        sizes = [ego_node_size if n == ego else partner_node_size for n in group]
        colors = [colors_lookup.get(_attr(n).get("Orientation", ""), "#888888") for n in group]
        edge_colors = [ego_highlight_color if n == ego else "none" for n in group]
        line_widths = [2.4 if n == ego else 0.0 for n in group]

        node_collection = nx.draw_networkx_nodes(
            sub,
            pos,
            nodelist=group,
            ax=ax,
            node_color=colors,
            node_size=sizes,
            node_shape=marker,
            alpha=1.0,
            linewidths=line_widths,  # [2.4 if n == ego else 0.6 for n in group],
            edgecolors=edge_colors,  # ["#000000" if n == ego else "#000000" for n in group],
        )
        if node_collection is not None:
            node_collection.set_clip_on(False)


# Add legend to the figure, with sex shapes and orientation colors, plus ego highlight. No legend frame.
def _add_legend(
    fig,
    color_by: str = "orientation",
    y_anchor: float = 0.02,
    orientation_colors: dict[str, str] | None = None,
    ego_highlight_color: str = "#E69F00",
    ncol: int = 4,
    fontsize: int = 12,
    legend_marker_size: int = 14,
    handletextpad: float = 0.45,
    columnspacing: float = 0.9,
    labelspacing: float = 0.3,
    compact: bool = False,
) -> None:
    handles = [Line2D([0], [0], color="none", label="Sex:")]
    for sex in ("Female", "Male"):
        handles.append(
            Line2D(
                [0],
                [0],
                marker=PALETTE.sex_shape(sex),
                color="#222222",
                linestyle="None",
                markersize=legend_marker_size,
                markeredgecolor="none",
                label=sex,
            )
        )
    if not compact:
        handles.append(Line2D([0], [0], color="none", label=" "))

    if color_by == "orientation":
        colors_lookup = orientation_colors or ORIENTATION_COLORS_ACCESSIBLE
        handles.append(Line2D([0], [0], color="none", label="Orientation:"))
        for ori in ("Opposite-sex", "Same-sex", "Bisexual"):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",  # square marker for orientation
                    color="w",
                    markerfacecolor=colors_lookup[ori],
                    markersize=legend_marker_size,
                    label=ori,
                    markeredgecolor="none",
                    markeredgewidth=0,
                )
            )
    if not compact:
        handles.append(Line2D([0], [0], color="none", label=" "))
    handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=legend_marker_size,
            markeredgecolor=ego_highlight_color,
            markeredgewidth=2.8,
            label="ego",
        )
    )

    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, y_anchor),
        handletextpad=handletextpad,
        columnspacing=columnspacing,
        labelspacing=labelspacing,
        borderaxespad=0.0,
    )
    if legend.get_frame() is not None:
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_linewidth(0.0)


# Static-aggregate ego-network plot: one panel per agent, showing all partnerships active at any point in [t_start, t_end].
def plot_ego_network_static_aggregate(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    output_dir: str,
    agents: list[int],
    t_start: int = 0,
    t_end: int = 1875,
    node_attr: Mapping[int, Mapping] | None = None,
    ego_radius: int = 1,
    filename_stem: str = "ego_network_static_aggregate",
    formats: OutputFormats = OutputFormats(),
    shared_layouts: dict[int, EgoLayout] | None = None,
) -> list[str]:
    if t_start > t_end:
        raise ValueError(f"t_start ({t_start}) must be <= t_end ({t_end})")
    if not agents:
        return []
    if shared_layouts is None:
        shared_layouts = build_shared_ego_layouts(
            partnerships,
            active,
            agents,
            t_start=t_start,
            t_end=t_end,
            ego_radius=ego_radius,
        )

    g_agg = _build_aggregate_graph(partnerships, active, t_start, t_end)
    n_rows = len(agents)

    with publication_style():
        fig_height = max(7.5, 2.6 * n_rows + 2.7)
        fig = plt.figure(figsize=(8.95, min(fig_height, 11.69)))
        gs = gridspec.GridSpec(
            n_rows,
            1,
            figure=fig,
            hspace=0.28,
            left=0.08,
            right=0.97,
            top=0.94,
            bottom=0.04,
        )
        for i, ego in enumerate(agents):
            ax = fig.add_subplot(gs[i, 0])
            sub = nx.ego_graph(g_agg, ego, ego_radius) if ego in g_agg else _empty(ego)
            n_partners = max(sub.number_of_nodes() - 1, 0)
            edge_alpha = 0.4 if n_partners > 40 else 0.55
            edge_width = 0.5 if n_partners > 40 else 0.8
            _draw_ego_panel(
                ax,
                sub,
                shared_layouts[ego],
                node_attr,
                ego,
                edge_alpha=edge_alpha,
                edge_width=edge_width,
                ego_node_size=180,
                partner_node_size=120,
            )
            _row_label(ax, ego)

        _add_legend(
            fig,
            y_anchor=0.0,
            **LEGEND_STYLE_LARGE,
        )
        fig.suptitle(
            f"Aggregate ego networks — all partnerships in [{t_start}, {t_end}]",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )
        out = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, out, formats)
        plt.close(fig)
    return written


# Add some constants for A4 page layout, to keep the figure size to a full page and derive row height from the number
# of agents rather than fixing row height and letting the figure shrink for small agent counts.
A4_WIDTH_IN = 8.95
A4_HEIGHT_IN = 11.69
TOP_MARGIN_IN = 1.0  # title + column headers ("t = ...")
BOTTOM_MARGIN_IN = 0.78  # legend, with clearance above it
MIN_ROW_HEIGHT_IN = 1.7  # floor, in case someone passes many agents


# Plot multi-agent, multi-timestep snapshots of k-hop ego networks, with one row per agent and one column per timestep.
def plot_ego_3hop_snapshots(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    output_dir: str,
    agents: list[int],
    timesteps: list[int],
    k_hops: int = 3,
    max_nodes: int = MAX_NODES_PER_PANEL,
    filename_stem: str = "ego_3hop_snapshots",
    formats: OutputFormats = OutputFormats(),
    agent_ages: Mapping[int, int] | None = None,
    title_suffix: str = "",
    show_title: bool = False,
    page_height_in: float = A4_HEIGHT_IN,
) -> list[str]:
    if not timesteps:
        raise ValueError("timesteps must be a non-empty list")
    if not agents:
        return []
    node_attr = build_node_attr(agent_log)

    n_rows, n_cols = len(agents), len(timesteps)

    # Fix the figure to a full A4 page and derive row height

    available_height = page_height_in - TOP_MARGIN_IN - BOTTOM_MARGIN_IN
    row_height_in = max(available_height / n_rows, MIN_ROW_HEIGHT_IN)
    fig_height = TOP_MARGIN_IN + n_rows * row_height_in + BOTTOM_MARGIN_IN
    top_frac = 1 - TOP_MARGIN_IN / fig_height
    bottom_frac = BOTTOM_MARGIN_IN / fig_height

    with publication_style():
        fig = plt.figure(figsize=(10.2, fig_height))
        gs = gridspec.GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            hspace=0.22,
            wspace=0.06,
            left=0.03,
            right=0.98,
            top=top_frac,
            bottom=bottom_frac,
        )
        for i, ego in enumerate(agents):
            for j, t in enumerate(timesteps):
                ax = fig.add_subplot(gs[i, j])
                g_t = build_graph_at(t, partnerships, active)
                sub, dist_map = _khop_subgraph_capped(g_t, ego, k_hops, max_nodes)
                layout = _layout_for_subgraph(sub, ego)
                _draw_ego_panel(
                    ax,
                    sub,
                    layout,
                    node_attr,
                    ego,
                    edge_color="#333333",
                    edge_alpha=0.75,
                    edge_width=1.6,
                    ego_node_size=180,
                    partner_node_size=120,
                )
                if i == 0:
                    ax.set_title(f"t = {t}", fontsize=13, fontweight="bold", pad=20)
                if j == 0:
                    age = agent_ages.get(ego) if agent_ages else None
                    _row_label(ax, ego, age=age)
        _add_legend(
            fig,
            y_anchor=0.005,
            **LEGEND_STYLE_LARGE,
        )
        if show_title:
            fig.suptitle(
                f"Snapshot ego networks ({k_hops}-hop, orientation-coloured){title_suffix}",
                fontsize=14,
                fontweight="bold",
                y=1 - (TOP_MARGIN_IN * 0.25) / fig_height,
            )
        out = os.path.join(output_dir, filename_stem)
        written = save_figure(fig, out, formats)
        plt.close(fig)
    return written


# Draw one figure per agent, showing the union of all partnerships active at any point in [t_start, t_end], with k-hop ego networks and orientation-coloured nodes.
def plot_ego_3hop_aggregate_per_agent(
    partnerships_df: pd.DataFrame,
    partnerships: PartnershipArrays,
    active: ActiveIntervals,
    agent_log: pd.DataFrame,
    output_dir: str,
    agents: list[int],
    total_timesteps: int,
    t_start: int = 1,
    t_end: int | None = None,
    k_hops: int = 3,
    max_nodes: int = MAX_NODES_PER_PANEL,
    filename_prefix: str = "ego_3hop_aggregate",
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """One figure per agent, sized to that agent's own subgraph density.

    Filenames are ``{filename_prefix}_agent{ego}.{ext}``.
    """
    if t_end is None:
        t_end = total_timesteps
    if not agents:
        return []
    node_attr = build_node_attr(agent_log)
    g_agg = _build_aggregate_graph(partnerships, active, t_start, t_end)

    written: list[str] = []

    with publication_style():
        for ego in agents:
            sub, dist_map = _khop_subgraph_capped(g_agg, ego, k_hops, max_nodes)
            layout = _layout_for_subgraph(sub, ego)

            n_nodes = sub.number_of_nodes()
            n_edges = sub.number_of_edges()
            # Scale figure size depending on how dense the focal agent's
            # subgraph is, rather than a fixed size for every agent.
            density_factor = 1.0 + min(n_edges / 30, 1.5)
            fig_w = 11.2 * density_factor
            fig_h = 6.0 * density_factor

            edge_alpha = 0.6 if n_edges > 40 else 0.55
            edge_width = 1 if n_edges > 40 else 0.8

            fig = plt.figure(figsize=(fig_w, fig_h))
            gs = gridspec.GridSpec(1, 1, figure=fig, left=0.05, right=0.97, top=0.90, bottom=0.08)
            ax = fig.add_subplot(gs[0, 0])
            _draw_ego_panel(
                ax,
                sub,
                layout,
                node_attr,
                ego,
                edge_alpha=edge_alpha,
                edge_width=edge_width,
                ego_node_size=500,
                partner_node_size=350,
            )

            fig.suptitle(
                f"Agent {ego} — {k_hops}-hop aggregate ego network "
                f"[{t_start}, {t_end}] ({n_nodes} nodes, {n_edges} edges)",
                fontsize=13,
                fontweight="bold",
                y=0.97,
            )
            _add_legend(
                fig,
                y_anchor=0.008,
                **LEGEND_STYLE_LARGE,
            )

            out = os.path.join(output_dir, f"{filename_prefix}_agent{ego}")
            written.extend(save_figure(fig, out, formats))
            plt.close(fig)

    return written
