"""Tests for the demographic constants added to constants.py.

These constants are not configurable but are fixed parameters of the
population structure. The tests check the expected values so they can't deviate.
"""

import pytest

from partnersim_dynet.config import (
    GUARANTEED_DEBUT_AGE,
    MIN_AGE,
    ORIENTATION_PRIORS_FEMALE,
    ORIENTATION_PRIORS_MALE,
    PROPORTION_MALE,
    SEXUAL_DEBUT_PROBABILITIES,
)


class TestSexAndOrientationPriors:
    def test_proportion_male_is_balanced(self):
        assert PROPORTION_MALE == 0.5

    def test_male_orientation_priors_sum_to_one(self):
        assert sum(ORIENTATION_PRIORS_MALE) == pytest.approx(1.0)

    def test_female_orientation_priors_sum_to_one(self):
        assert sum(ORIENTATION_PRIORS_FEMALE) == pytest.approx(1.0)

    def test_male_orientation_priors_match_calibration(self):
        # (Opposite-sex, Same-sex, Bisexual) — pin down the calibrated values
        assert ORIENTATION_PRIORS_MALE == (0.90, 0.05, 0.05)

    def test_female_orientation_priors_match_calibration(self):
        assert ORIENTATION_PRIORS_FEMALE == (0.80, 0.10, 0.10)


class TestSexualDebut:
    def test_debut_schedule_covers_relevant_ages(self):
        # The schedule should cover MIN_AGE up to (but not including) the
        # guaranteed-debut age.
        expected_ages = set(range(MIN_AGE, GUARANTEED_DEBUT_AGE))
        assert set(SEXUAL_DEBUT_PROBABILITIES.keys()) == expected_ages

    def test_debut_probabilities_are_valid(self):
        for age, p in SEXUAL_DEBUT_PROBABILITIES.items():
            assert 0.0 <= p <= 1.0, f"age {age}: prob {p} outside [0, 1]"

    def test_debut_probabilities_decrease_with_age(self):
        # The expected pattern: most people debut at the youngest ages,
        # so the per-year debut probability decreases.
        ages = sorted(SEXUAL_DEBUT_PROBABILITIES.keys())
        probs = [SEXUAL_DEBUT_PROBABILITIES[a] for a in ages]
        for p1, p2 in zip(probs, probs[1:], strict=False):
            assert p1 >= p2, f"debut probs increase: {probs}"

    def test_guaranteed_debut_age_is_after_schedule(self):
        last_scheduled_age = max(SEXUAL_DEBUT_PROBABILITIES.keys())
        assert GUARANTEED_DEBUT_AGE > last_scheduled_age
