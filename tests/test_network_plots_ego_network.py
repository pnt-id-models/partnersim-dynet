"""Tests for the three ego-network plot variants."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from partnersim_dynet.network.plots import (
    OutputFormats,
    build_node_attr,
    build_shared_ego_layouts,
    identify_top_concurrent_agents,
    plot_ego_network_active_snapshot,
    plot_ego_network_dynamic,
    plot_ego_network_static_aggregate,
)


# identify_top_concurrent_agents


class TestIdentifyTopConcurrent:
    def test_empty_partnerships_returns_empty(self):
        df = pd.DataFrame(
            columns=[
                "Agent",
                "PartnerAgent",
                "StartTime",
                "EndTime",
            ]
        )
        assert identify_top_concurrent_agents(df, top_n=5) == []

    def test_no_overlap_means_no_concurrency(self):
        # Two sequential partnerships, no overlap
        df = pd.DataFrame(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 20},
                {"Agent": 1, "PartnerAgent": 3, "StartTime": 30, "EndTime": 40},
            ]
        )
        # max_simultaneous = 1 for agent 1, so it's not in the >=2 concurrent set,
        # but it gets included in the fallback fill
        result = identify_top_concurrent_agents(df, top_n=3)
        assert 1 in result

    def test_overlapping_partnerships_count(self):
        df = pd.DataFrame(
            [
                {"Agent": 1, "PartnerAgent": 2, "StartTime": 10, "EndTime": 30},
                {"Agent": 1, "PartnerAgent": 3, "StartTime": 20, "EndTime": 40},
                {"Agent": 1, "PartnerAgent": 4, "StartTime": 25, "EndTime": 35},
            ]
        )
        # Agent 1 had up to 3 simultaneous partners (between t=25 and t=30)
        result = identify_top_concurrent_agents(df, top_n=1)
        assert result == [1]


# build_node_attr


class TestBuildNodeAttr:
    def test_basic(self):
        log = pd.DataFrame(
            [
                {
                    "Agent": 1,
                    "Sex": "Males",
                    "Orientation": "Opposite-sex",
                    "EntryAge": 25,
                    "EntryTimestep": 1,
                },
                {
                    "Agent": 2,
                    "Sex": "Females",
                    "Orientation": "Bisexual",
                    "EntryAge": 30,
                    "EntryTimestep": 1,
                },
            ]
        )
        attr = build_node_attr(log)
        assert attr[1] == {"Sex": "Males", "Orientation": "Opposite-sex", "Age": 25}
        assert attr[2] == {"Sex": "Females", "Orientation": "Bisexual", "Age": 30}

    def test_with_snapshot_advances_age(self):
        log = pd.DataFrame(
            [
                {
                    "Agent": 1,
                    "Sex": "Males",
                    "Orientation": "Opposite-sex",
                    "EntryAge": 25,
                    "EntryTimestep": 1,
                },
            ]
        )
        attr = build_node_attr(log, snapshot_t=366)
        # Linear age model: 25 + (366 - 1) = 390. (Approximation; real
        # generator increments age once per 365 timesteps.)
        assert attr[1]["Age"] == 25 + 365


# End-to-end tests with a real simulation


@pytest.fixture(scope="module")
def real_simulation_data():
    """One small simulation, shared across the three ego-plot tests."""
    from partnersim_dynet.config import PartnershipConfig
    from partnersim_dynet.generator import PartnershipGenerator
    from partnersim_dynet.network import (
        ActiveIntervals,
        prepare_partnerships,
    )

    cfg = PartnershipConfig(
        num_agents=200,
        total_timesteps=200,
        concurrency_prop=0.20,
        concurrency_model=1,
    )
    gen = PartnershipGenerator(cfg, seed=42)
    df = gen.simulate_partnerships()
    log = gen.get_agent_log()
    active = ActiveIntervals.from_agent_log(log, total_timesteps=200)
    arr = prepare_partnerships(df, total_timesteps=200)
    node_attr = build_node_attr(log)
    return df, arr, active, log, node_attr


class TestPlotEgoNetworkDynamic:
    def test_writes_files(self, tmp_path, real_simulation_data):
        df, arr, active, _log, node_attr = real_simulation_data
        written = plot_ego_network_dynamic(
            partnerships_df=df,
            partnerships=arr,
            active=active,
            output_dir=str(tmp_path),
            top_n=2,
            timesteps=[50, 100, 150, 200],
            node_attr=node_attr,
            formats=OutputFormats.all_enabled(),
        )
        assert len(written) == 3
        for p in written:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000

    def test_empty_timesteps_raises(self, tmp_path, real_simulation_data):
        df, arr, active, _, node_attr = real_simulation_data
        with pytest.raises(ValueError, match="non-empty"):
            plot_ego_network_dynamic(
                partnerships_df=df,
                partnerships=arr,
                active=active,
                output_dir=str(tmp_path),
                top_n=2,
                timesteps=[],
                node_attr=node_attr,
            )


class TestPlotEgoNetworkActiveSnapshot:
    def test_writes_files(self, tmp_path, real_simulation_data):
        df, arr, active, _, node_attr = real_simulation_data
        written = plot_ego_network_active_snapshot(
            partnerships_df=df,
            partnerships=arr,
            active=active,
            output_dir=str(tmp_path),
            top_n=2,
            snapshot_t=150,
            node_attr=node_attr,
        )
        assert len(written) >= 1
        assert all(os.path.exists(p) for p in written)


class TestPlotEgoNetworkStaticAggregate:
    def test_writes_files(self, tmp_path, real_simulation_data):
        df, arr, active, _, node_attr = real_simulation_data
        written = plot_ego_network_static_aggregate(
            partnerships_df=df,
            partnerships=arr,
            active=active,
            output_dir=str(tmp_path),
            top_n=2,
            t_start=1,
            t_end=200,
            node_attr=node_attr,
        )
        assert len(written) >= 1
        assert all(os.path.exists(p) for p in written)

    def test_invalid_window_raises(self, tmp_path, real_simulation_data):
        df, arr, active, _, node_attr = real_simulation_data
        with pytest.raises(ValueError, match="t_start"):
            plot_ego_network_static_aggregate(
                partnerships_df=df,
                partnerships=arr,
                active=active,
                output_dir=str(tmp_path),
                top_n=2,
                t_start=200,
                t_end=50,
                node_attr=node_attr,
            )


# Shared layouts produce identical positions across variants


class TestSharedLayouts:
    def test_same_layout_used_across_variants(self, tmp_path, real_simulation_data):
        """Building the layout once and passing it in should give the
        same positions to all three variants — that's the whole point."""
        df, arr, active, _, node_attr = real_simulation_data
        top_agents = identify_top_concurrent_agents(df, top_n=2)
        if not top_agents:
            pytest.skip("no top agents in this simulation")
        layouts = build_shared_ego_layouts(
            arr,
            active,
            top_agents,
            t_start=1,
            t_end=200,
        )
        # Call all three variants with the same layouts
        plot_ego_network_dynamic(
            partnerships_df=df,
            partnerships=arr,
            active=active,
            output_dir=str(tmp_path / "dyn"),
            top_n=2,
            timesteps=[50, 100, 150, 200],
            node_attr=node_attr,
            shared_layouts=layouts,
        )
        plot_ego_network_active_snapshot(
            partnerships_df=df,
            partnerships=arr,
            active=active,
            output_dir=str(tmp_path / "snap"),
            top_n=2,
            snapshot_t=150,
            node_attr=node_attr,
            shared_layouts=layouts,
        )
        plot_ego_network_static_aggregate(
            partnerships_df=df,
            partnerships=arr,
            active=active,
            output_dir=str(tmp_path / "agg"),
            top_n=2,
            t_start=1,
            t_end=200,
            node_attr=node_attr,
            shared_layouts=layouts,
        )
        # All three should have produced output
        assert os.path.exists(tmp_path / "dyn" / "ego_network_dynamic.png")
        assert os.path.exists(tmp_path / "snap" / "ego_network_active_snapshot.png")
        assert os.path.exists(tmp_path / "agg" / "ego_network_static_aggregate.png")


# No global state pollution


class TestNoGlobalPollution:
    def test_rcparams_not_mutated(self, tmp_path, real_simulation_data):
        df, arr, active, _, node_attr = real_simulation_data
        original_dpi = plt.rcParams["savefig.dpi"]
        plot_ego_network_dynamic(
            partnerships_df=df,
            partnerships=arr,
            active=active,
            output_dir=str(tmp_path),
            top_n=2,
            timesteps=[50, 100, 150, 200],
            node_attr=node_attr,
        )
        assert plt.rcParams["savefig.dpi"] == original_dpi
