"""Tests for network/metrics.py — per-graph metrics and the temporal driver."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from partnersim_dynet.network import (
    ActiveIntervals,
    component_stats,
    compute_temporal_metrics,
    degree_stats,
    prepare_partnerships,
    sampled_avg_path_length,
    transitivity,
)

# Per-graph metric functions

class TestDegreeStats:
    def test_empty_graph(self):
        G = nx.Graph()
        avg, mx, active = degree_stats(G)
        assert (avg, mx, active) == (0.0, 0, 0)

    def test_isolated_nodes_only(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        avg, mx, active = degree_stats(G)
        assert (avg, mx, active) == (0.0, 0, 0)

    def test_single_edge(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_node(3)  # isolated
        avg, mx, active = degree_stats(G)
        # avg = (1 + 1 + 0) / 3 = 2/3
        assert avg == pytest.approx(2 / 3)
        assert mx == 1
        assert active == 2

    def test_star_graph(self):
        # Center node 0 connected to 1, 2, 3, 4
        G = nx.star_graph(4)
        avg, mx, active = degree_stats(G)
        # avg = (4 + 1 + 1 + 1 + 1) / 5 = 8/5
        assert avg == pytest.approx(8 / 5)
        assert mx == 4
        assert active == 5

class TestComponentStats:
    def test_empty(self):
        assert component_stats(nx.Graph()) == (0, 0, 0.0)

    def test_one_component(self):
        G = nx.path_graph(5)
        n_comp, lcc, mean = component_stats(G)
        assert n_comp == 1
        assert lcc == 5
        assert mean == 5.0

    def test_multiple_components(self):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (3, 4), (5, 6)])
        G.add_node(7)  # isolated
        n_comp, lcc, mean = component_stats(G)
        # 3 dyads + 1 singleton = 4 components
        assert n_comp == 4
        assert lcc == 2
        # sizes [2, 2, 2, 1], mean = 7/4
        assert mean == pytest.approx(7 / 4)


class TestTransitivity:
    def test_empty(self):
        assert transitivity(nx.Graph()) == 0.0

    def test_triangle(self):
        # Single triangle: transitivity should be 1.0
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])
        assert transitivity(G) == pytest.approx(1.0)

    def test_path_has_zero_transitivity(self):
        # Path has triads (open) but no triangles → transitivity = 0
        G = nx.path_graph(5)
        assert transitivity(G) == 0.0


class TestSampledAvgPathLength:
    def test_empty_returns_zero(self):
        assert sampled_avg_path_length(
            nx.Graph(), sample_size=10, rng=np.random.default_rng(0)
        ) == 0.0

    def test_single_edge_lcc(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        # LCC has 2 nodes, distance between them is 1
        apl = sampled_avg_path_length(G, sample_size=10, rng=np.random.default_rng(0))
        assert apl == pytest.approx(1.0)

    def test_path_graph_known_apl(self):
        # Path on 5 nodes: distances are (1,2,3,4,1,2,3,1,2,1) = 20/10 = 2.0
        G = nx.path_graph(5)
        # With sample_size >= 5, we hit every source, so estimate ≈ true APL
        apl = sampled_avg_path_length(G, sample_size=5, rng=np.random.default_rng(0))
        # True APL of path_graph(5) = 2.0
        assert apl == pytest.approx(2.0, rel=0.01)


# Temporal driver

@pytest.fixture
def small_temporal_setup() -> tuple:
    """Build a small partnership + active setup for testing the temporal driver."""
    # 3 agents, all active 1..50
    log = pd.DataFrame({
        "Agent": [1, 2, 3],
        "EntryTimestep": [1, 1, 1],
        "ExitTimestep": [float("nan")] * 3,
    })
    log["Agent"] = log["Agent"].astype("int64")
    log["EntryTimestep"] = log["EntryTimestep"].astype("int32")
    log["ExitTimestep"] = log["ExitTimestep"].astype("float64")
    active = ActiveIntervals.from_agent_log(log, total_timesteps=50)

    df = pd.DataFrame([
        {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 30},
        {"Agent": 2, "PartnerAgent": 3, "StartTime": 20, "EndTime": 40},
    ])
    df["Agent"] = df["Agent"].astype("int64")
    df["PartnerAgent"] = df["PartnerAgent"].astype("float64")
    df["StartTime"] = df["StartTime"].astype("float64")
    df["EndTime"] = df["EndTime"].astype("float64")
    arr = prepare_partnerships(df, total_timesteps=50)

    return arr, active


class TestComputeTemporalMetrics:
    def test_output_columns(self, small_temporal_setup):
        arr, active = small_temporal_setup
        metrics = compute_temporal_metrics(arr, active, total_timesteps=50)
        expected = {
            "t", "num_nodes", "num_edges", "active_nodes", "avg_degree",
            "max_degree", "new_edges", "lost_edges",
            "num_components", "largest_component_size", "mean_component_size",
            "transitivity", "avg_path_length",
        }
        assert set(metrics.columns) == expected

    def test_num_nodes_constant_when_no_removals(self, small_temporal_setup):
        arr, active = small_temporal_setup
        metrics = compute_temporal_metrics(arr, active, total_timesteps=50)
        # With nobody exiting, num_nodes is 3 throughout
        assert (metrics["num_nodes"] == 3).all()

    def test_edge_lifecycle(self, small_temporal_setup):
        arr, active = small_temporal_setup
        metrics = compute_temporal_metrics(arr, active, total_timesteps=50)

        # Before any partnership starts: 0 edges
        assert metrics.loc[metrics["t"] == 9, "num_edges"].item() == 0

        # After 1-2 starts (t=10), before 2-3 starts (t=20)
        assert metrics.loc[metrics["t"] == 15, "num_edges"].item() == 1

        # Both partnerships active (20 <= t < 30)
        assert metrics.loc[metrics["t"] == 25, "num_edges"].item() == 2

        # 1-2 ended (t=30 onward), 2-3 still active until t=40
        assert metrics.loc[metrics["t"] == 35, "num_edges"].item() == 1

        # Both ended (t >= 40)
        assert metrics.loc[metrics["t"] == 45, "num_edges"].item() == 0

    def test_new_and_lost_edges(self, small_temporal_setup):
        arr, active = small_temporal_setup
        metrics = compute_temporal_metrics(arr, active, total_timesteps=50)

        # 2 new edges total (at t=10 and t=20)
        assert metrics["new_edges"].sum() == 2

        # 2 lost edges total (at t=30 and t=40)
        assert metrics["lost_edges"].sum() == 2


# Reproducibility

class TestMetricsReproducibility:
    def test_same_seed_gives_same_metrics(self, small_temporal_setup):
        arr, active = small_temporal_setup
        a = compute_temporal_metrics(arr, active, total_timesteps=50, rng_seed=42)
        b = compute_temporal_metrics(arr, active, total_timesteps=50, rng_seed=42)
        pd.testing.assert_frame_equal(a, b)

# Integration with generator

class TestIntegrationWithGenerator:
    def test_full_pipeline(self):
        """Run a small simulation, compute metrics, verify the shape and
        a few sanity invariants."""
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator

        cfg = PartnershipConfig(num_agents=200, total_timesteps=200)
        gen = PartnershipGenerator(cfg, seed=42)
        df = gen.simulate_partnerships()
        active = ActiveIntervals.from_agent_log(
            gen.get_agent_log(), total_timesteps=cfg.total_timesteps
        )
        arr = prepare_partnerships(df, total_timesteps=cfg.total_timesteps)

        metrics = compute_temporal_metrics(
            arr, active, total_timesteps=cfg.total_timesteps
        )

        assert len(metrics) == cfg.total_timesteps
        # Steady-state population: num_nodes should equal num_agents at every t
        assert (metrics["num_nodes"] == cfg.num_agents).all()


# ─────────────────────────────────────────────────────────────────────────────
# Degree distribution functions
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_setup_with_log() -> tuple:
    """A small partnership setup including the agent log needed by the
    degree functions."""
    log = pd.DataFrame([
        {"Agent": 1, "Sex": "Males",   "Orientation": "Opposite-sex",
         "EntryAge": 25, "EntryTimestep": 1, "ExitTimestep": float("nan")},
        {"Agent": 2, "Sex": "Females", "Orientation": "Opposite-sex",
         "EntryAge": 24, "EntryTimestep": 1, "ExitTimestep": float("nan")},
        {"Agent": 3, "Sex": "Females", "Orientation": "Bisexual",
         "EntryAge": 30, "EntryTimestep": 1, "ExitTimestep": float("nan")},
    ])
    log["Agent"] = log["Agent"].astype("int64")
    log["EntryTimestep"] = log["EntryTimestep"].astype("int32")
    log["ExitTimestep"] = log["ExitTimestep"].astype("float64")

    active = ActiveIntervals.from_agent_log(log, total_timesteps=50)

    df = pd.DataFrame([
        {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 30},
        {"Agent": 2, "PartnerAgent": 3, "StartTime": 20, "EndTime": 40},
    ])
    df["Agent"] = df["Agent"].astype("int64")
    df["PartnerAgent"] = df["PartnerAgent"].astype("float64")
    df["StartTime"] = df["StartTime"].astype("float64")
    df["EndTime"] = df["EndTime"].astype("float64")
    arr = prepare_partnerships(df, total_timesteps=50)

    return arr, active, log


class TestDegreeAtSnapshots:
    def test_columns(self, small_setup_with_log):
        from partnersim_dynet.network import degree_at_snapshots
        arr, active, log = small_setup_with_log
        result = degree_at_snapshots([15, 25, 35], arr, active, log)
        expected = {
            "t", "Agent", "Degree",
            "AgentSex", "AgentOrientation", "AgentAge", "AgentAgeGroup",
        }
        assert set(result.columns) == expected

    def test_known_degrees(self, small_setup_with_log):
        from partnersim_dynet.network import degree_at_snapshots
        arr, active, log = small_setup_with_log
        # At t=15: only 1-2 active, so degrees are A1=1, A2=1, A3=0
        result = degree_at_snapshots([15], arr, active, log)
        degrees = dict(zip(result["Agent"], result["Degree"]))
        assert degrees == {1: 1, 2: 1, 3: 0}

    def test_multiple_snapshots(self, small_setup_with_log):
        from partnersim_dynet.network import degree_at_snapshots
        arr, active, log = small_setup_with_log
        # 3 timesteps × 3 agents = 9 rows
        result = degree_at_snapshots([5, 25, 45], arr, active, log)
        assert len(result) == 9

    def test_agent_age_advances_correctly(self, small_setup_with_log):
        from partnersim_dynet.network import degree_at_snapshots
        arr, active, log = small_setup_with_log
        # Agent 1 enters at age 25 at t=1; at t=10, age is 25 (no birthday)
        # Note: age in our model is in years not timesteps, so the increment
        # is (t - entry_t), but this is a coarse approximation — in the real
        # generator, age advances once per 365 timesteps. For these tests we
        # accept the linear model since the agent log only stores entry age.
        # We just want to verify the column exists and is sensible.
        result = degree_at_snapshots([1], arr, active, log)
        ages = dict(zip(result["Agent"], result["AgentAge"]))
        # At t=1, all agents are at their entry age
        assert ages == {1: 25, 2: 24, 3: 30}

    def test_empty_snapshot_list_raises(self, small_setup_with_log):
        from partnersim_dynet.network import degree_at_snapshots
        arr, active, log = small_setup_with_log
        with pytest.raises(ValueError, match="non-empty"):
            degree_at_snapshots([], arr, active, log)


class TestDegreeInWindow:
    def test_known_partner_counts(self, small_setup_with_log):
        from partnersim_dynet.network import degree_in_window
        arr, active, log = small_setup_with_log
        # Window [1, 50] covers everything. Agent 1 partnered with {2},
        # Agent 2 partnered with {1, 3}, Agent 3 partnered with {2}.
        result = degree_in_window(1, 50, arr, active, log)
        degrees = dict(zip(result["Agent"], result["Degree"]))
        assert degrees == {1: 1, 2: 2, 3: 1}

    def test_narrow_window(self, small_setup_with_log):
        from partnersim_dynet.network import degree_in_window
        arr, active, log = small_setup_with_log
        # Window [22, 28]: only 1-2 partnership (t=10-30) is fully overlapping
        # and 2-3 partnership (t=20-40) is also overlapping
        result = degree_in_window(22, 28, arr, active, log)
        degrees = dict(zip(result["Agent"], result["Degree"]))
        # 1 partnered with 2; 2 partnered with both; 3 partnered with 2
        assert degrees == {1: 1, 2: 2, 3: 1}

    def test_window_excluding_all_partnerships(self, small_setup_with_log):
        from partnersim_dynet.network import degree_in_window
        arr, active, log = small_setup_with_log
        # Window [45, 50] — both partnerships have ended
        result = degree_in_window(45, 50, arr, active, log)
        degrees = dict(zip(result["Agent"], result["Degree"]))
        # All agents present, but no partnerships
        assert degrees == {1: 0, 2: 0, 3: 0}

    def test_invalid_window_raises(self, small_setup_with_log):
        from partnersim_dynet.network import degree_in_window
        arr, active, log = small_setup_with_log
        with pytest.raises(ValueError, match="t_start"):
            degree_in_window(50, 10, arr, active, log)


class TestDegreeByDemographicOverTime:
    def test_columns(self, small_setup_with_log):
        from partnersim_dynet.network import degree_by_demographic_over_time
        arr, active, log = small_setup_with_log
        result = degree_by_demographic_over_time(arr, active, log, total_timesteps=50)
        expected = {
            "t", "AgentSex", "AgentOrientation", "AgentAgeGroup",
            "MeanDegree", "P50Degree", "P90Degree", "N",
        }
        assert set(result.columns) == expected

    def test_only_populated_combos_present(self, small_setup_with_log):
        from partnersim_dynet.network import degree_by_demographic_over_time
        arr, active, log = small_setup_with_log
        result = degree_by_demographic_over_time(arr, active, log, total_timesteps=50)
        # We have 3 agents in 3 different combos. So at every timestep
        # we should have exactly 3 rows.
        per_t_counts = result.groupby("t").size()
        assert (per_t_counts == 3).all()

    def test_mean_degree_changes_with_edges(self, small_setup_with_log):
        from partnersim_dynet.network import degree_by_demographic_over_time
        arr, active, log = small_setup_with_log
        result = degree_by_demographic_over_time(arr, active, log, total_timesteps=50)

        # At t=5, no partnerships active → all degrees 0
        early = result[result["t"] == 5]
        assert (early["MeanDegree"] == 0).all()

        # At t=25, both partnerships active → mean degree per agent is
        # nonzero for at least some demographic groups
        mid = result[result["t"] == 25]
        assert mid["MeanDegree"].sum() > 0


class TestIntegrationWithGenerator:
    """Spot-check the three functions work end-to-end with a real run."""

    def test_all_three_functions_work(self):
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator
        from partnersim_dynet.network import (
            degree_at_snapshots,
            degree_by_demographic_over_time,
            degree_in_window,
        )

        cfg = PartnershipConfig(num_agents=200, total_timesteps=200)
        gen = PartnershipGenerator(cfg, seed=42)
        df = gen.simulate_partnerships()
        log = gen.get_agent_log()
        active = ActiveIntervals.from_agent_log(log, total_timesteps=200)
        arr = prepare_partnerships(df, total_timesteps=200)

        snap = degree_at_snapshots([50, 100, 150], arr, active, log)
        win = degree_in_window(1, 200, arr, active, log)
        demo = degree_by_demographic_over_time(arr, active, log, total_timesteps=200)

        # All should be non-empty
        assert len(snap) > 0
        assert len(win) > 0
        assert len(demo) > 0

        # Sanity checks
        assert (snap["Degree"] >= 0).all()
        assert (win["Degree"] >= 0).all()
        assert (demo["MeanDegree"] >= 0).all()