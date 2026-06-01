"""Tests for the timeseries plot functions.

Plot tests can verify:
- The function runs without crashing
- Output files are written and non-empty
- The right formats are produced
- The right errors are raised

They CAN'T verify the plot is visually correct — that requires human
inspection.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from partnersim_dynet.network.plots import (
    SPEC_AVG_DEGREE,
    OutputFormats,
    TimeseriesSpec,
    plot_avg_degree,
    plot_avg_path_length,
    plot_max_degree,
    plot_timeseries,
    plot_transitivity,
)

# Helpers


def _make_metrics_df(n_timesteps: int = 100) -> pd.DataFrame:
    """Build a synthetic metrics DataFrame with all the columns the plot
    functions might want."""
    rng = np.random.default_rng(0)
    t = np.arange(1, n_timesteps + 1)
    return pd.DataFrame(
        {
            "t": t,
            "avg_degree": rng.uniform(0.0, 2.0, size=n_timesteps),
            "max_degree": rng.integers(0, 10, size=n_timesteps),
            "transitivity": rng.uniform(0.0, 0.3, size=n_timesteps),
            "avg_path_length": rng.uniform(1.0, 5.0, size=n_timesteps),
            "num_nodes": np.full(n_timesteps, 100),
            "num_edges": rng.integers(0, 50, size=n_timesteps),
        }
    )


# TimeseriesSpec
class TestTimeseriesSpec:
    def test_filename_stem_derived_from_column(self):
        spec = TimeseriesSpec(
            metric_column="my_metric",
            ylabel="y",
            title="t",
            color="#000000",
        )
        assert spec.filename_stem == "my_metric_over_time"

    def test_filename_stem_can_be_overridden(self):
        spec = TimeseriesSpec(
            metric_column="my_metric",
            ylabel="y",
            title="t",
            color="#000000",
            filename_stem="custom_name",
        )
        assert spec.filename_stem == "custom_name"

    def test_spec_is_frozen(self):
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):  # FrozenInstanceError
            SPEC_AVG_DEGREE.color = "#FF0000"  # type: ignore[misc]


# plot_timeseries — core function
class TestPlotTimeseries:
    def test_writes_png_by_default(self, tmp_path):
        metrics = _make_metrics_df()
        written = plot_timeseries(metrics, SPEC_AVG_DEGREE, str(tmp_path))
        assert len(written) == 1
        assert written[0].endswith(".png")
        assert os.path.exists(written[0])
        assert os.path.getsize(written[0]) > 100

    def test_writes_all_formats(self, tmp_path):
        metrics = _make_metrics_df()
        written = plot_timeseries(
            metrics,
            SPEC_AVG_DEGREE,
            str(tmp_path),
            formats=OutputFormats.all_enabled(),
        )
        assert len(written) == 3
        exts = {os.path.splitext(p)[1] for p in written}
        assert exts == {".png", ".pdf", ".svg"}

    def test_filename_matches_spec(self, tmp_path):
        metrics = _make_metrics_df()
        written = plot_timeseries(metrics, SPEC_AVG_DEGREE, str(tmp_path))
        assert os.path.basename(written[0]) == "avg_degree_over_time.png"

    def test_missing_metric_column_raises(self, tmp_path):
        metrics = pd.DataFrame({"t": [1, 2, 3]})
        with pytest.raises(KeyError, match="metric column"):
            plot_timeseries(metrics, SPEC_AVG_DEGREE, str(tmp_path))

    def test_t_window_cropping(self, tmp_path):
        """A narrower window should still produce a valid plot file."""
        metrics = _make_metrics_df()
        written = plot_timeseries(
            metrics,
            SPEC_AVG_DEGREE,
            str(tmp_path),
            t_start=20,
            t_end=80,
        )
        assert os.path.exists(written[0])

    def test_t_end_defaults_to_max(self, tmp_path):
        """When t_end is None, the plot should cover up to the last
        timestep without crashing."""
        metrics = _make_metrics_df()
        written = plot_timeseries(
            metrics,
            SPEC_AVG_DEGREE,
            str(tmp_path),
            t_start=1,
            t_end=None,
        )
        assert os.path.exists(written[0])

    def test_does_not_leave_open_figures(self, tmp_path):
        """plot_timeseries should close its figure (no leak)."""
        metrics = _make_metrics_df()
        before = len(plt.get_fignums())
        plot_timeseries(metrics, SPEC_AVG_DEGREE, str(tmp_path))
        after = len(plt.get_fignums())
        assert after == before


# Convenience wrappers


class TestConvenienceWrappers:
    @pytest.mark.parametrize(
        "fn, expected_stem",
        [
            (plot_avg_degree, "avg_degree_over_time"),
            (plot_max_degree, "max_degree_over_time"),
            (plot_transitivity, "transitivity_over_time"),
            (plot_avg_path_length, "avg_path_length_over_time"),
        ],
    )
    def test_wrapper_writes_correct_file(self, tmp_path, fn, expected_stem):
        metrics = _make_metrics_df()
        written = fn(metrics, str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), f"{expected_stem}.png"))
        assert any(expected_stem in p for p in written)


# No global state pollution
class TestNoGlobalPollution:
    def test_rcparams_not_mutated(self, tmp_path):
        """Calling a plot function should not mutate global rcParams.
        This is the whole point of the publication_style context manager."""
        original_dpi = plt.rcParams["savefig.dpi"]
        original_spines = plt.rcParams["axes.spines.top"]

        metrics = _make_metrics_df()
        plot_avg_degree(metrics, str(tmp_path))

        assert plt.rcParams["savefig.dpi"] == original_dpi
        assert plt.rcParams["axes.spines.top"] == original_spines


# Integration with generator
class TestIntegrationWithRealMetrics:
    def test_runs_against_real_metrics_output(self, tmp_path):
        """End-to-end: simulate, compute metrics, plot all four."""
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator
        from partnersim_dynet.network import (
            ActiveIntervals,
            compute_temporal_metrics,
            prepare_partnerships,
        )

        cfg = PartnershipConfig(num_agents=200, total_timesteps=200)
        gen = PartnershipGenerator(cfg, seed=42)
        df = gen.simulate_partnerships()
        active = ActiveIntervals.from_agent_log(gen.get_agent_log(), total_timesteps=200)
        arr = prepare_partnerships(df, total_timesteps=200)
        metrics = compute_temporal_metrics(arr, active, total_timesteps=200)

        for plot_fn in (plot_avg_degree, plot_max_degree, plot_transitivity, plot_avg_path_length):
            written = plot_fn(metrics, str(tmp_path))
            assert all(os.path.exists(p) for p in written)
