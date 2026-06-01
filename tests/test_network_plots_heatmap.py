"""Tests for the degree heatmap plot."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from partnersim_dynet.config import AGE_GROUPS
from partnersim_dynet.network.plots import (
    OutputFormats,
    plot_degree_heatmap_evolution,
)

# Helpers


def _make_demographic_df(
    timesteps: list[int],
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic degree_by_demographic DataFrame.

    Has one row per (t, AgeGroup, Sex, Orientation) combination, with
    plausible random degree values.
    """
    rng = np.random.default_rng(seed)
    sexes = ("Male", "Female")
    oris = ("Opposite-sex", "Same-sex", "Bisexual")

    rows = []
    for t in timesteps:
        for ag in AGE_GROUPS:
            for sx in sexes:
                for ori in oris:
                    rows.append(
                        {
                            "t": t,
                            "AgentSex": sx,
                            "AgentOrientation": ori,
                            "AgentAgeGroup": ag,
                            "MeanDegree": float(rng.uniform(0, 2)),
                            "P50Degree": 0.0,
                            "P90Degree": 0.0,
                            "N": rng.integers(5, 50),
                        }
                    )
    return pd.DataFrame(rows)


# plot_degree_heatmap_evolution


class TestPlotDegreeHeatmap:
    def test_writes_png_by_default(self, tmp_path):
        df = _make_demographic_df([50, 100, 150])
        written = plot_degree_heatmap_evolution(
            df, snapshot_times=[50, 100, 150], output_dir=str(tmp_path)
        )
        assert len(written) == 1
        assert written[0].endswith(".png")
        assert os.path.exists(written[0])
        assert os.path.getsize(written[0]) > 1000

    def test_writes_all_formats(self, tmp_path):
        df = _make_demographic_df([50, 100, 150])
        written = plot_degree_heatmap_evolution(
            df,
            snapshot_times=[50, 100, 150],
            output_dir=str(tmp_path),
            formats=OutputFormats.all_enabled(),
        )
        assert len(written) == 3

    def test_custom_filename_stem(self, tmp_path):
        df = _make_demographic_df([50, 100, 150])
        plot_degree_heatmap_evolution(
            df,
            snapshot_times=[50, 100, 150],
            output_dir=str(tmp_path),
            filename_stem="custom_heatmap",
        )
        assert os.path.exists(os.path.join(str(tmp_path), "custom_heatmap.png"))

    def test_empty_snapshot_times_raises(self, tmp_path):
        df = _make_demographic_df([50, 100, 150])
        with pytest.raises(ValueError, match="non-empty"):
            plot_degree_heatmap_evolution(df, snapshot_times=[], output_dir=str(tmp_path))

    def test_missing_snapshot_t_raises(self, tmp_path):
        df = _make_demographic_df([50, 100, 150])
        with pytest.raises(ValueError, match="not in DataFrame"):
            plot_degree_heatmap_evolution(df, snapshot_times=[50, 999], output_dir=str(tmp_path))

    def test_missing_column_raises(self, tmp_path):
        df = pd.DataFrame({"t": [1], "AgentSex": ["Male"]})
        # Missing AgentOrientation, AgentAgeGroup, MeanDegree
        with pytest.raises(KeyError, match="missing columns"):
            plot_degree_heatmap_evolution(df, snapshot_times=[1], output_dir=str(tmp_path))

    def test_single_snapshot_works(self, tmp_path):
        """When n_cols=1, axes is 1-D not 2-D; the function should still work."""
        df = _make_demographic_df([50])
        written = plot_degree_heatmap_evolution(df, snapshot_times=[50], output_dir=str(tmp_path))
        assert os.path.exists(written[0])

    def test_no_figure_leaks(self, tmp_path):
        df = _make_demographic_df([50, 100])
        before = len(plt.get_fignums())
        plot_degree_heatmap_evolution(df, snapshot_times=[50, 100], output_dir=str(tmp_path))
        after = len(plt.get_fignums())
        assert after == before

    def test_handles_zero_degree_data(self, tmp_path):
        """If every cell is zero, vmax falls back to 1.0 to avoid a
        degenerate colorbar."""
        df = _make_demographic_df([50, 100])
        df["MeanDegree"] = 0.0
        written = plot_degree_heatmap_evolution(
            df, snapshot_times=[50, 100], output_dir=str(tmp_path)
        )
        assert os.path.exists(written[0])

    def test_handles_combos_missing_from_some_snapshots(self, tmp_path):
        """If a (t, ori, sex, age) combo is missing from the DataFrame
        (e.g. no agents in that demographic at that timestep), the
        heatmap should still render — that cell is just blank."""
        df = _make_demographic_df([50, 100])
        # Remove all rows for the (50, "Same-sex") subset
        df = df[~((df["t"] == 50) & (df["AgentOrientation"] == "Same-sex"))]
        written = plot_degree_heatmap_evolution(
            df, snapshot_times=[50, 100], output_dir=str(tmp_path)
        )
        assert os.path.exists(written[0])


# Integration with the full pipeline


class TestIntegrationWithRealMetrics:
    def test_runs_against_real_metrics_output(self, tmp_path):
        """End-to-end: simulate, compute degree_by_demographic, plot."""
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator
        from partnersim_dynet.network import (
            ActiveIntervals,
            degree_by_demographic_over_time,
            prepare_partnerships,
        )

        cfg = PartnershipConfig(num_agents=200, total_timesteps=200)
        gen = PartnershipGenerator(cfg, seed=42)
        df = gen.simulate_partnerships()
        log = gen.get_agent_log()
        active = ActiveIntervals.from_agent_log(log, total_timesteps=200)
        arr = prepare_partnerships(df, total_timesteps=200)

        demo = degree_by_demographic_over_time(arr, active, log, total_timesteps=200)

        written = plot_degree_heatmap_evolution(
            demo,
            snapshot_times=[50, 100, 150, 200],
            output_dir=str(tmp_path),
            formats=OutputFormats.all_enabled(),
        )
        assert len(written) == 3
        for p in written:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000
