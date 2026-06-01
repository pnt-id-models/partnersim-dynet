"""Tests for the graph_builder module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partnersim_dynet.network import (
    ActiveIntervals,
    build_graph_at,
    iter_partnership_events,
    prepare_partnerships,
)

# Helpers


def _make_partnership_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal partnership DataFrame with the required columns."""
    if not rows:
        return pd.DataFrame(columns=["Agent", "PartnerAgent", "StartTime", "EndTime"])
    df = pd.DataFrame(rows)
    df["Agent"] = df["Agent"].astype("int64")
    df["PartnerAgent"] = df["PartnerAgent"].astype("float64")
    df["StartTime"] = df["StartTime"].astype("float64")
    df["EndTime"] = df["EndTime"].astype("float64")
    return df


def _make_active(agent_ids: list[int], total_timesteps: int = 100) -> ActiveIntervals:
    """Build an ActiveIntervals where every agent is active throughout."""
    log = pd.DataFrame(
        {
            "Agent": agent_ids,
            "EntryTimestep": [1] * len(agent_ids),
            "ExitTimestep": [float("nan")] * len(agent_ids),
        }
    )
    log["Agent"] = log["Agent"].astype("int64")
    log["EntryTimestep"] = log["EntryTimestep"].astype("int32")
    log["ExitTimestep"] = log["ExitTimestep"].astype("float64")
    return ActiveIntervals.from_agent_log(log, total_timesteps=total_timesteps)


# prepare_partnerships


class TestPreparePartnerships:
    def test_basic_conversion(self):
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
                {"Agent": 3, "PartnerAgent": 4, "StartTime": 20, "EndTime": 80},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)
        np.testing.assert_array_equal(arr.agent, [1, 3])
        np.testing.assert_array_equal(arr.partner, [2, 4])
        np.testing.assert_array_equal(arr.start, [10, 20])
        np.testing.assert_array_equal(arr.end, [50, 80])

    def test_singleton_rows_filtered(self):
        # Singletons: PartnerAgent is NaN
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
                {
                    "Agent": 5,
                    "PartnerAgent": float("nan"),
                    "StartTime": float("nan"),
                    "EndTime": float("nan"),
                },
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)
        assert len(arr.agent) == 1
        assert arr.agent[0] == 1

    def test_nan_end_becomes_sentinel(self):
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": float("nan")},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)
        assert arr.end[0] == 101  # total_timesteps + 1

    def test_missing_column_raises(self):
        df = pd.DataFrame({"Agent": [1], "PartnerAgent": [2], "StartTime": [10]})
        # EndTime missing
        with pytest.raises(ValueError, match="missing columns"):
            prepare_partnerships(df, total_timesteps=100)

    def test_zero_total_timesteps_raises(self):
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
            ]
        )
        with pytest.raises(ValueError, match="total_timesteps"):
            prepare_partnerships(df, total_timesteps=0)


# build_graph_at


class TestBuildGraphAt:
    def test_isolated_nodes_included(self):
        """An agent active at t with no partnerships must still be a node."""
        active = _make_active([1, 2, 3], total_timesteps=100)
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)

        G = build_graph_at(t=20, partnerships=arr, active=active)
        # Agent 3 has no edges but should still be present
        assert 3 in G.nodes
        assert G.degree(3) == 0
        # Agents 1 and 2 are connected
        assert G.has_edge(1, 2)

    def test_only_alive_nodes_present(self):
        # Agent 99 is in a partnership row but not in the agent log →
        # they shouldn't appear in the graph.
        active = _make_active([1, 2], total_timesteps=100)
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 99, "StartTime": 10, "EndTime": 50},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)

        G = build_graph_at(t=20, partnerships=arr, active=active)
        assert 99 not in G.nodes
        # And the edge is dropped because one endpoint isn't active
        assert not G.has_edge(1, 99)

    def test_partnership_ended_before_t_excluded(self):
        active = _make_active([1, 2], total_timesteps=100)
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 30},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)

        # At t=50, partnership has already ended
        G = build_graph_at(t=50, partnerships=arr, active=active)
        assert not G.has_edge(1, 2)

    def test_partnership_not_started_yet_excluded(self):
        active = _make_active([1, 2], total_timesteps=100)
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 50, "EndTime": 80},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)

        # At t=20, partnership hasn't started
        G = build_graph_at(t=20, partnerships=arr, active=active)
        assert not G.has_edge(1, 2)

    def test_partnership_boundaries(self):
        """StartTime is inclusive, EndTime is exclusive."""
        active = _make_active([1, 2], total_timesteps=100)
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)

        assert build_graph_at(t=9, partnerships=arr, active=active).has_edge(1, 2) is False
        assert build_graph_at(t=10, partnerships=arr, active=active).has_edge(1, 2) is True
        assert build_graph_at(t=49, partnerships=arr, active=active).has_edge(1, 2) is True
        assert build_graph_at(t=50, partnerships=arr, active=active).has_edge(1, 2) is False

    def test_empty_partnerships(self):
        active = _make_active([1, 2, 3], total_timesteps=100)
        # Empty DataFrame with the required columns
        df = pd.DataFrame(columns=["Agent", "PartnerAgent", "StartTime", "EndTime"])
        arr = prepare_partnerships(df, total_timesteps=100)

        G = build_graph_at(t=50, partnerships=arr, active=active)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 0

    def test_returns_simple_graph(self):
        """Even if the same pair appears in two partnership rows, the
        resulting graph has only one edge between them."""
        active = _make_active([1, 2], total_timesteps=100)
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},  # duplicate
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)

        G = build_graph_at(t=20, partnerships=arr, active=active)
        assert G.number_of_edges() == 1


# iter_partnership_events


class TestIterPartnershipEvents:
    def test_yields_starts_and_ends(self):
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)
        events = list(iter_partnership_events(arr))
        kinds = [e.kind for e in events]
        # Each partnership produces both a start and an end event
        assert sorted(kinds) == ["end", "start"]

    def test_events_in_chronological_order(self):
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
                {"Agent": 3, "PartnerAgent": 4, "StartTime": 5, "EndTime": 80},
                {"Agent": 5, "PartnerAgent": 6, "StartTime": 20, "EndTime": 30},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)
        events = list(iter_partnership_events(arr))
        times = [e.t for e in events]
        assert times == sorted(times)

    def test_ends_before_starts_at_same_t(self):
        """At the same timestep, end events come before start events."""
        df = _make_partnership_df(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 50},
                {"Agent": 3, "PartnerAgent": 4, "StartTime": 50, "EndTime": 80},
            ]
        )
        arr = prepare_partnerships(df, total_timesteps=100)
        events_at_50 = [e for e in iter_partnership_events(arr) if e.t == 50]
        # End of partnership 1-2 must come before start of 3-4
        assert events_at_50[0].kind == "end"
        assert events_at_50[1].kind == "start"

    def test_empty_partnerships(self):
        df = pd.DataFrame(columns=["Agent", "PartnerAgent", "StartTime", "EndTime"])
        arr = prepare_partnerships(df, total_timesteps=100)
        events = list(iter_partnership_events(arr))
        assert events == []


# Integration with generator


class TestIntegrationWithGenerator:
    def test_graph_at_steady_state(self):
        """A graph built mid-simulation should have num_agents nodes."""
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator

        cfg = PartnershipConfig(num_agents=200, total_timesteps=200)
        gen = PartnershipGenerator(cfg, seed=0)
        df = gen.simulate_partnerships()
        active = ActiveIntervals.from_agent_log(
            gen.get_agent_log(), total_timesteps=cfg.total_timesteps
        )
        arr = prepare_partnerships(df, total_timesteps=cfg.total_timesteps)

        G = build_graph_at(t=100, partnerships=arr, active=active)
        assert G.number_of_nodes() == cfg.num_agents

    def test_isolated_node_count_makes_sense(self):
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator

        cfg = PartnershipConfig(num_agents=200, total_timesteps=200)
        gen = PartnershipGenerator(cfg, seed=0)
        df = gen.simulate_partnerships()
        active = ActiveIntervals.from_agent_log(
            gen.get_agent_log(), total_timesteps=cfg.total_timesteps
        )
        arr = prepare_partnerships(df, total_timesteps=cfg.total_timesteps)

        G = build_graph_at(t=100, partnerships=arr, active=active)
        # Some nodes will be isolated (no active partnerships at t=100), the total node count is num_agents, not just non-isolated ones.
        isolated = sum(1 for n in G.nodes if G.degree(n) == 0)
        connected = G.number_of_nodes() - isolated
        # Both should be > 0
        assert isolated > 0, "no isolated nodes — verify partnership coverage"
        assert connected > 0, "no connected nodes — verify formation worked"
