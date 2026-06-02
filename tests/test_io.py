"""Tests for partnersim_dynet.io utilities."""

from __future__ import annotations

import os

import pytest

from partnersim_dynet.io import make_experiment_dir


class TestMakeExperimentDir:
    def test_creates_run_1_when_no_existing(self, tmp_path):
        path = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="my_experiment",
            date_str="2026-06-15",
        )
        assert os.path.basename(path) == "my_experiment_2026-06-15_Run#1"
        assert os.path.isdir(path)

    def test_serial_increments_on_repeat(self, tmp_path):
        path1 = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
            date_str="2026-06-15",
        )
        path2 = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
            date_str="2026-06-15",
        )
        path3 = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
            date_str="2026-06-15",
        )
        assert os.path.basename(path1) == "exp_2026-06-15_Run#1"
        assert os.path.basename(path2) == "exp_2026-06-15_Run#2"
        assert os.path.basename(path3) == "exp_2026-06-15_Run#3"

    def test_different_prefix_independent_serial(self, tmp_path):
        """exp_A and exp_B keep separate serials."""
        a = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp_A",
            date_str="2026-06-15",
        )
        b = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp_B",
            date_str="2026-06-15",
        )
        # Both get Run#1 because the prefixes differ
        assert os.path.basename(a) == "exp_A_2026-06-15_Run#1"
        assert os.path.basename(b) == "exp_B_2026-06-15_Run#1"

    def test_different_date_independent_serial(self, tmp_path):
        a = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
            date_str="2026-06-14",
        )
        b = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
            date_str="2026-06-15",
        )
        # Both get Run#1 because the dates differ
        assert os.path.basename(a) == "exp_2026-06-14_Run#1"
        assert os.path.basename(b) == "exp_2026-06-15_Run#1"

    def test_serial_resumes_after_gap(self, tmp_path):
        """If only Run#1 and Run#3 exist (Run#2 manually deleted),
        the next directory should be Run#4 (max+1), not Run#2."""
        # Create Run#1 and Run#3 manually
        os.makedirs(tmp_path / "exp_2026-06-15_Run#1")
        os.makedirs(tmp_path / "exp_2026-06-15_Run#3")
        new = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
            date_str="2026-06-15",
        )
        assert os.path.basename(new) == "exp_2026-06-15_Run#4"

    def test_ignores_unrelated_directories(self, tmp_path):
        """Other directories in base_dir don't affect the serial."""
        os.makedirs(tmp_path / "unrelated_dir")
        os.makedirs(tmp_path / "exp_2025-12-31_Run#5")  # different date
        os.makedirs(tmp_path / "other_2026-06-15_Run#9")  # different prefix
        path = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
            date_str="2026-06-15",
        )
        assert os.path.basename(path) == "exp_2026-06-15_Run#1"

    def test_creates_base_dir_if_missing(self, tmp_path):
        nested = tmp_path / "nested" / "deeper"
        path = make_experiment_dir(
            base_dir=str(nested),
            prefix="exp",
            date_str="2026-06-15",
        )
        assert os.path.isdir(path)

    def test_uses_today_when_date_str_none(self, tmp_path):
        from datetime import date

        path = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
        )
        today = date.today().isoformat()
        assert today in path

    def test_empty_prefix_raises(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            make_experiment_dir(base_dir=str(tmp_path), prefix="")

    def test_prefix_with_whitespace_raises(self, tmp_path):
        with pytest.raises(ValueError, match="whitespace or slashes"):
            make_experiment_dir(base_dir=str(tmp_path), prefix="my exp")

    def test_prefix_with_slash_raises(self, tmp_path):
        with pytest.raises(ValueError, match="whitespace or slashes"):
            make_experiment_dir(base_dir=str(tmp_path), prefix="my/exp")

    def test_returns_absolute_path(self, tmp_path):
        path = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="exp",
            date_str="2026-06-15",
        )
        assert os.path.isabs(path)

    def test_handles_special_characters_in_prefix(self, tmp_path):
        """Underscores, hyphens, dots are fine."""
        path = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="female_hetero-baseline.v2",
            date_str="2026-06-15",
        )
        assert os.path.isdir(path)
        # And the serial should still increment correctly
        path2 = make_experiment_dir(
            base_dir=str(tmp_path),
            prefix="female_hetero-baseline.v2",
            date_str="2026-06-15",
        )
        assert os.path.basename(path2).endswith("Run#2")
