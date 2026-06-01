"""Tests for ProbabilityConfig — validation, construction, and reference behaviour."""

import pytest

from partnersim_dynet.config import AGE_GROUPS, ProbabilityConfig


class TestDefaultsReproduceOldUtils:
    def test_reference_cell_equals_base(self):
        """Female / Opposite-sex / 16-24 should approximately equal base × youth_boost.

        With default sex_mult=1.0 and orient_mult=1.0, the formation prob for
        the reference cell is base × age_multiplier["16-24"], where the age
        multiplier is the youth_boost.
        """
        cfg = ProbabilityConfig()
        table = cfg.build_formation_probs()
        expected = cfg.formation_base * cfg.formation_youth_boost
        assert table["Female"]["Opposite-sex"]["16-24"] == pytest.approx(expected)

    def test_baseline_age_group_has_multiplier_one(self):
        cfg = ProbabilityConfig()
        mults = cfg.formation_age_multipliers()
        assert mults[cfg.age_decay_baseline_group] == pytest.approx(1.0)

    def test_age_multipliers_decay_monotonically_after_baseline(self):
        cfg = ProbabilityConfig()
        mults = cfg.formation_age_multipliers()
        baseline_idx = AGE_GROUPS.index(cfg.age_decay_baseline_group)
        values_after_baseline = [mults[AGE_GROUPS[i]] for i in range(baseline_idx, len(AGE_GROUPS))]
        # strictly decreasing
        for a, b in zip(values_after_baseline, values_after_baseline[1:], strict=False):
            assert a > b


class TestTableStructure:
    def test_all_cells_populated(self):
        cfg = ProbabilityConfig()
        table = cfg.build_formation_probs()
        for sex in ("Male", "Female"):
            for ori in ("Opposite-sex", "Same-sex", "Bisexual"):
                for age_group in AGE_GROUPS:
                    assert age_group in table[sex][ori]
                    assert table[sex][ori][age_group] >= 0

    def test_formation_and_breakage_differ(self):
        """The orientation multipliers differ between formation and breakage,
        so the tables should not be identical."""
        cfg = ProbabilityConfig()
        f = cfg.build_formation_probs()
        b = cfg.build_breakage_probs()
        # at least one cell must differ
        any_differ = any(
            f[sex][ori][ag] != b[sex][ori][ag]
            for sex in ("Male", "Female")
            for ori in ("Opposite-sex", "Same-sex", "Bisexual")
            for ag in AGE_GROUPS
        )
        assert any_differ


class TestValidation:
    def test_negative_base_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ProbabilityConfig(formation_base=-0.001)

    def test_negative_decay_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ProbabilityConfig(formation_age_decay=-0.1)

    def test_unknown_baseline_group_rejected(self):
        with pytest.raises(ValueError, match="age_decay_baseline_group"):
            ProbabilityConfig(age_decay_baseline_group="999")

    def test_unknown_sex_key_rejected(self):
        with pytest.raises(ValueError, match="unknown sex key"):
            ProbabilityConfig(sex_multipliers={"X": 1.0})


class TestOverrides:
    """Custom configs (the LHS use case) should work without monkey-patching."""

    def test_override_base_changes_table(self):
        default = ProbabilityConfig().build_formation_probs()
        doubled = ProbabilityConfig(formation_base=0.004).build_formation_probs()
        # every cell should be exactly twice as large
        for sex in ("Male", "Female"):
            for ori in ("Opposite-sex", "Same-sex", "Bisexual"):
                for ag in AGE_GROUPS:
                    assert doubled[sex][ori][ag] == pytest.approx(2 * default[sex][ori][ag])

    def test_zero_youth_boost_zeroes_16_24_row(self):
        cfg = ProbabilityConfig(formation_youth_boost=0.0)
        table = cfg.build_formation_probs()
        for sex in ("Male", "Female"):
            for ori in ("Opposite-sex", "Same-sex", "Bisexual"):
                assert table[sex][ori]["16-24"] == 0.0
