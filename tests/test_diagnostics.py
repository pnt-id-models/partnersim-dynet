"""Tests for the diagnostics module."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from partnersim_dynet.config import PartnershipConfig
from partnersim_dynet.diagnostics import (
    export_probability_bounds,
    export_probability_bounds_csv,
    plot_agent_probability_distributions,
    print_probability_table,
    save_probability_table,
)

# Helpers


def _real_agent_log():
    """Run a small simulation and return the agent log."""
    from partnersim_dynet.generator import PartnershipGenerator

    cfg = PartnershipConfig(num_agents=300, total_timesteps=100)
    gen = PartnershipGenerator(cfg, seed=42)
    gen.simulate_partnerships()
    return cfg, gen.get_agent_log()


# Probability tables (base rates, config only)


class TestPrintProbabilityTable:
    def test_prints_without_crashing(self, capsys):
        cfg = PartnershipConfig()
        print_probability_table(cfg)
        captured = capsys.readouterr()
        assert "Calibrated probabilities" in captured.out
        assert "Male" in captured.out
        assert "Female" in captured.out
        assert "16-24" in captured.out

    def test_without_breakage(self, capsys):
        cfg = PartnershipConfig()
        print_probability_table(cfg, include_breakage=False)
        captured = capsys.readouterr()
        assert "Formation prob" in captured.out
        assert "Breakage prob" not in captured.out


class TestSaveProbabilityTable:
    def test_writes_csv(self, tmp_path):
        cfg = PartnershipConfig()
        path = save_probability_table(cfg, str(tmp_path / "probs.csv"))
        assert os.path.exists(path)

        df = pd.read_csv(path)
        assert set(df.columns) == {"Type", "Sex", "Orientation", "AgeGroup", "Probability"}
        # Both types present
        assert {"Formation", "Breakage"} <= set(df["Type"])
        # All age groups present
        from partnersim_dynet.config import AGE_GROUPS

        assert set(df["AgeGroup"]) >= set(AGE_GROUPS)

    def test_creates_parent_dir(self, tmp_path):
        cfg = PartnershipConfig()
        path = save_probability_table(cfg, str(tmp_path / "sub" / "probs.csv"))
        assert os.path.exists(path)


# Effective probability bounds (from real run)


class TestExportProbabilityBounds:
    def test_returns_dataframe_with_expected_columns(self):
        cfg, log = _real_agent_log()
        bounds = export_probability_bounds(cfg, log)
        expected = {
            "AgeGroup",
            "Sex",
            "Orientation",
            "AgentCount",
            "Formation_Base",
            "Formation_Effective_Min",
            "Formation_Effective_Max",
            "Breakage_Base",
            "Breakage_Effective_Min",
            "Breakage_Effective_Max",
        }
        assert set(bounds.columns) == expected

    def test_effective_max_above_base(self):
        """With NB heterogeneity, some agents should have effective probs
        ABOVE the base rate (NB multiplier > 1)."""
        cfg, log = _real_agent_log()
        bounds = export_probability_bounds(cfg, log)
        # At least one combo should have Effective_Max strictly > Base
        gap = bounds["Formation_Effective_Max"] - bounds["Formation_Base"]
        assert (gap > 0).any(), "no combo had effective max above base — NB multiplier broken?"

    def test_agent_count_sums_to_log_length(self):
        cfg, log = _real_agent_log()
        bounds = export_probability_bounds(cfg, log)
        assert bounds["AgentCount"].sum() == len(log)

    def test_missing_column_raises(self):
        cfg = PartnershipConfig()
        bad_log = pd.DataFrame({"Agent": [1]})  # missing required columns
        with pytest.raises(KeyError, match="missing columns"):
            export_probability_bounds(cfg, bad_log)


class TestExportProbabilityBoundsCsv:
    def test_writes_csv(self, tmp_path):
        cfg, log = _real_agent_log()
        path = export_probability_bounds_csv(cfg, log, str(tmp_path / "bounds.csv"))
        assert os.path.exists(path)
        # Read it back to confirm it's valid
        df = pd.read_csv(path)
        assert "AgentCount" in df.columns


# Agent probability distribution plots


class TestPlotAgentProbabilityDistributions:
    def test_writes_six_figures(self, tmp_path):
        cfg, log = _real_agent_log()
        written = plot_agent_probability_distributions(cfg, log, str(tmp_path))
        # 2 sexes × 3 orientations × 1 format = 6 files (PNG by default)
        assert len(written) == 6
        for p in written:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000

    def test_writes_all_formats(self, tmp_path):
        from partnersim_dynet.network.plots import OutputFormats

        cfg, log = _real_agent_log()
        written = plot_agent_probability_distributions(
            cfg,
            log,
            str(tmp_path),
            formats=OutputFormats.all_enabled(),
        )
        # 6 combinations × 3 formats = 18 files
        assert len(written) == 18

    def test_filename_prefix(self, tmp_path):
        cfg, log = _real_agent_log()
        written = plot_agent_probability_distributions(
            cfg, log, str(tmp_path), filename_prefix="custom_prefix"
        )
        for p in written:
            assert "custom_prefix_" in os.path.basename(p)

    def test_no_figure_leaks(self, tmp_path):
        cfg, log = _real_agent_log()
        before = len(plt.get_fignums())
        plot_agent_probability_distributions(cfg, log, str(tmp_path))
        after = len(plt.get_fignums())
        assert after == before

    def test_rcparams_not_mutated(self, tmp_path):
        cfg, log = _real_agent_log()
        original_dpi = plt.rcParams["savefig.dpi"]
        plot_agent_probability_distributions(cfg, log, str(tmp_path))
        assert plt.rcParams["savefig.dpi"] == original_dpi
