"""Tests for the simulator: run_single and run_replicates."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from partnersim_dynet import RunResult, run_replicates, run_single
from partnersim_dynet.config import PartnershipConfig, SimulationConfig


# Minimal config for fast tests


def _tiny_cfg() -> PartnershipConfig:
    return PartnershipConfig(num_agents=80, total_timesteps=60)


# run_single: default (no analysis)


class TestRunSingleDefault:
    def test_basic_run_writes_two_files(self, tmp_path):
        result = run_single(_tiny_cfg(), seed=42, output_dir=str(tmp_path))
        assert isinstance(result, RunResult)
        assert result.seed == 42
        # Default: partnerships + agent_log
        assert len(result.files_written) == 2
        assert os.path.exists(tmp_path / "partnerships.parquet")
        assert os.path.exists(tmp_path / "agent_log.parquet")

    def test_csv_format(self, tmp_path):
        run_single(_tiny_cfg(), seed=42, output_dir=str(tmp_path), output_format="csv")
        assert os.path.exists(tmp_path / "partnerships.csv")
        assert not os.path.exists(tmp_path / "partnerships.parquet")

    def test_invalid_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unsupported format"):
            run_single(
                _tiny_cfg(),
                seed=42,
                output_dir=str(tmp_path),
                output_format="json",
            )


# run_single: with analysis flags


class TestRunSingleWithAnalysis:
    def test_run_metrics_writes_metrics(self, tmp_path):
        result = run_single(
            _tiny_cfg(),
            seed=42,
            output_dir=str(tmp_path),
            run_metrics=True,
        )
        assert os.path.exists(tmp_path / "metrics.parquet")
        # Quick sanity check on the metrics DataFrame
        m = pd.read_parquet(tmp_path / "metrics.parquet")
        assert "avg_degree" in m.columns
        assert len(m) == 60

    def test_run_degree_distributions(self, tmp_path):
        run_single(
            _tiny_cfg(),
            seed=42,
            output_dir=str(tmp_path),
            run_degree_distributions=True,
        )
        assert os.path.exists(tmp_path / "degree_by_demographic.parquet")
        assert os.path.exists(tmp_path / "degree_at_snapshots.parquet")
        assert os.path.exists(tmp_path / "degree_in_window.parquet")

    def test_run_plots(self, tmp_path):
        run_single(_tiny_cfg(), seed=42, output_dir=str(tmp_path), run_plots=True)
        # Metrics should also be saved (auto-required by plots)
        assert os.path.exists(tmp_path / "metrics.parquet")
        # Plots subdirectory should exist with files
        plots_dir = tmp_path / "plots"
        assert plots_dir.is_dir()
        png_files = list(plots_dir.glob("*.png"))
        # 4 timeseries + 1 heatmap + 3 ego = 8, minimum
        assert len(png_files) >= 8

    def test_run_diagnostics(self, tmp_path):
        run_single(_tiny_cfg(), seed=42, output_dir=str(tmp_path), run_diagnostics=True)
        diag_dir = tmp_path / "diagnostics"
        assert diag_dir.is_dir()
        assert (diag_dir / "base_probabilities.csv").exists()
        assert (diag_dir / "effective_bounds.csv").exists()
        # 6 demographic figures
        png_files = list(diag_dir.glob("*.png"))
        assert len(png_files) == 6

    def test_all_flags_together(self, tmp_path):
        result = run_single(
            _tiny_cfg(),
            seed=42,
            output_dir=str(tmp_path),
            run_metrics=True,
            run_degree_distributions=True,
            run_plots=True,
            run_diagnostics=True,
        )
        # Just verify we got a coherent result and many files
        assert len(result.files_written) > 15


0
# run_single: reproducibility


class TestReproducibility:
    def test_same_seed_same_output(self, tmp_path):
        run_single(_tiny_cfg(), seed=42, output_dir=str(tmp_path / "a"))
        run_single(_tiny_cfg(), seed=42, output_dir=str(tmp_path / "b"))
        a = pd.read_parquet(tmp_path / "a" / "partnerships.parquet")
        b = pd.read_parquet(tmp_path / "b" / "partnerships.parquet")
        pd.testing.assert_frame_equal(a, b)


# run_replicates


class TestRunReplicatesSerial:
    def test_runs_correct_number_of_replicates(self, tmp_path):
        sim_cfg = SimulationConfig(
            partnership=_tiny_cfg(),
            n_partnership_replicates=3,
            n_workers=1,
        )
        results = run_replicates(sim_cfg, str(tmp_path))
        assert len(results) == 3
        # Each result has its own output dir
        for r in results:
            assert os.path.isdir(r.output_dir)
            assert os.path.exists(os.path.join(r.output_dir, "partnerships.parquet"))

    def test_seeds_deterministic_from_base(self, tmp_path):
        sim_cfg = SimulationConfig(
            partnership=_tiny_cfg(),
            n_partnership_replicates=3,
            base_partnership_seed=42,
            n_workers=1,
        )
        a = run_replicates(sim_cfg, str(tmp_path / "a"))
        b = run_replicates(sim_cfg, str(tmp_path / "b"))
        assert [r.seed for r in a] == [r.seed for r in b]

    def test_results_sorted_by_seed(self, tmp_path):
        sim_cfg = SimulationConfig(
            partnership=_tiny_cfg(),
            n_partnership_replicates=4,
            n_workers=1,
        )
        results = run_replicates(sim_cfg, str(tmp_path))
        seeds = [r.seed for r in results]
        assert seeds == sorted(seeds)


class TestRunReplicatesParallel:
    """Parallel execution. Marked slow because it pays ProcessPool overhead."""

    @pytest.mark.slow
    def test_parallel_matches_serial(self, tmp_path):
        """The same SimulationConfig should produce the same outputs
        whether run serially or in parallel."""
        sim_cfg_a = SimulationConfig(
            partnership=_tiny_cfg(),
            n_partnership_replicates=3,
            n_workers=1,
        )
        sim_cfg_b = SimulationConfig(
            partnership=_tiny_cfg(),
            n_partnership_replicates=3,
            n_workers=2,
        )
        a = run_replicates(sim_cfg_a, str(tmp_path / "a"))
        b = run_replicates(sim_cfg_b, str(tmp_path / "b"))

        # Same seeds
        assert [r.seed for r in a] == [r.seed for r in b]

        # Same outputs per seed
        for ra, rb in zip(a, b, strict=False):
            da = pd.read_parquet(os.path.join(ra.output_dir, "partnerships.parquet"))
            db = pd.read_parquet(os.path.join(rb.output_dir, "partnerships.parquet"))
            pd.testing.assert_frame_equal(da, db)


# run_replicates


class TestRunReplicatesFlagsPropagate:
    def test_metrics_flag_applies_to_all_replicates(self, tmp_path):
        sim_cfg = SimulationConfig(
            partnership=_tiny_cfg(),
            n_partnership_replicates=2,
            n_workers=1,
            run_metrics=True,
        )
        results = run_replicates(sim_cfg, str(tmp_path))
        for r in results:
            assert os.path.exists(os.path.join(r.output_dir, "metrics.parquet"))
