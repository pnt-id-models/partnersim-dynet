"""Tests for config.constants"""

import pytest

from partnersim_dynet.config import (
    AGE_GROUP_BOUNDARIES,
    AGE_GROUPS,
    MAX_AGE,
    MIN_AGE,
    REPLENISHMENT_AGE,
    age_to_group,
)


class TestAgeGroups:
    """Age group boundaries are inclusive on both ends.
    The age_to_group() function is the source of truth for the age boundaries.
    The tests check the inclusive boundaries, the "Unknown" fallback for out-of-bounds ages, and that every valid age maps to a real group.
    """

    def test_lower_boundary_inclusive(self):
        assert age_to_group(16) == "16-24"
        assert age_to_group(25) == "25-34"
        assert age_to_group(35) == "35-44"
        assert age_to_group(45) == "45-54"
        assert age_to_group(55) == "55-64"
        assert age_to_group(65) == "65-74"

    def test_upper_boundary_inclusive(self):
        assert age_to_group(24) == "16-24"
        assert age_to_group(34) == "25-34"
        assert age_to_group(44) == "35-44"
        assert age_to_group(54) == "45-54"
        assert age_to_group(64) == "55-64"
        assert age_to_group(74) == "65-74"

    def test_below_min_age_is_unknown(self):
        assert age_to_group(15) == "Unknown"
        assert age_to_group(0) == "Unknown"
        assert age_to_group(-1) == "Unknown"

    def test_above_max_age_is_unknown(self):
        # Agents above MAX_AGE should be removed and not appear in the simulation, but if they do, the function must not crash.
        assert age_to_group(75) == "Unknown"
        assert age_to_group(100) == "Unknown"

    @pytest.mark.parametrize("age", range(16, 75))
    def test_every_valid_age_maps_to_a_group(self, age):
        """Every age from MIN_AGE through MAX_AGE must map to a group."""
        result = age_to_group(age)
        assert result != "Unknown", f"age {age} mapped to Unknown"
        assert result in AGE_GROUPS


class TestAgeGroupStructure:
    """Structural invariants of the age-group definitions."""

    def test_age_groups_match_boundaries(self):
        assert AGE_GROUPS == tuple(label for label, _, _ in AGE_GROUP_BOUNDARIES)

    def test_no_gaps_between_groups(self):
        for (_, _, prev_hi), (_, next_lo, _) in zip(
            AGE_GROUP_BOUNDARIES, AGE_GROUP_BOUNDARIES[1:], strict=False
        ):
            assert next_lo == prev_hi + 1, f"gap between {prev_hi} and {next_lo}"

    def test_min_age_matches_first_group(self):
        assert MIN_AGE == AGE_GROUP_BOUNDARIES[0][1]

    def test_max_age_matches_last_group(self):
        assert MAX_AGE == AGE_GROUP_BOUNDARIES[-1][2]

    def test_replenishment_age_is_min_age(self):
        # New agents enter at the start of the age range
        assert REPLENISHMENT_AGE == MIN_AGE


class TestSexAndOrientationMaps:
    """Code ↔ string mappings are consistent."""

    def test_sex_round_trip(self):
        from partnersim_dynet.config import SEX_CODE_TO_STR, SEX_STR_TO_CODE

        for code, label in SEX_CODE_TO_STR.items():
            assert SEX_STR_TO_CODE[label] == code

    def test_orientation_round_trip(self):
        from partnersim_dynet.config import ORI_CODE_TO_STR, ORI_STR_TO_CODE

        for code, label in ORI_CODE_TO_STR.items():
            assert ORI_STR_TO_CODE[label] == code
