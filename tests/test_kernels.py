"""Tests for kernels.py — numba JIT helpers and vectorized breakage math.

The kernels are numerical functions, so they should be tested seperately from the rest of the generator code.

"""

from __future__ import annotations

import numpy as np
import pytest

from partnersim_dynet.config import MIN_AGE, age_to_group
from partnersim_dynet.generator import (
    compute_breakage_events,
    compute_failure_rates,
    fast_digitise_age_group,
    fast_normal_pdf,
)


# ─── fast_normal_pdf ──────────────────────────────────────────────────────────


class TestFastNormalPdf:
    def test_peak_at_loc(self):
        """The PDF should be maximised at x = loc."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = fast_normal_pdf(x, loc=0.0, scale=1.0)
        assert np.argmax(y) == 2  # the x=0 entry

    def test_symmetric_around_loc(self):
        x = np.array([-3.0, -1.0, 1.0, 3.0])
        y = fast_normal_pdf(x, loc=0.0, scale=1.0)
        np.testing.assert_allclose(y[0], y[3], rtol=1e-10)
        np.testing.assert_allclose(y[1], y[2], rtol=1e-10)

    def test_integrates_to_approx_one(self):
        """Riemann-sum sanity check over a wide range."""
        x = np.linspace(-10, 10, 10001)
        y = fast_normal_pdf(x, loc=0.0, scale=1.0)
        approx_integral = np.trapezoid(y, x)
        np.testing.assert_allclose(approx_integral, 1.0, rtol=1e-3)

    def test_matches_analytic_at_known_points(self):
        """Spot-check against the standard normal PDF formula."""
        # PDF(0) = 1 / sqrt(2*pi) ≈ 0.3989
        y0 = fast_normal_pdf(np.array([0.0]), loc=0.0, scale=1.0)[0]
        np.testing.assert_allclose(y0, 1 / np.sqrt(2 * np.pi), rtol=1e-10)

    def test_scale_widens_distribution(self):
        """A larger scale should produce a lower peak."""
        peak_narrow = fast_normal_pdf(np.array([0.0]), loc=0.0, scale=1.0)[0]
        peak_wide = fast_normal_pdf(np.array([0.0]), loc=0.0, scale=4.0)[0]
        assert peak_narrow > peak_wide


# ─── fast_digitise_age_group ──────────────────────────────────────────────────


class TestFastDigitiseAgeGroup:
    """The numba age-group function must match the Python age_to_group
    function from constants.py exactly. Otherwise we have the same bug
    that plagued the old codebase: two implementations, subtly different."""

    @pytest.mark.parametrize("age", range(MIN_AGE, 100))
    def test_matches_python_age_to_group(self, age):
        ages_arr = np.array([age], dtype=np.int16)
        idx = int(fast_digitise_age_group(ages_arr)[0])

        # Map numba's int index back to a label using the same scheme
        # documented in the function docstring.
        labels = ["16-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75", "Unknown"]
        numba_label = labels[idx]

        python_label = age_to_group(age)

        # Special case: age 75 maps to "75" in the numba helper but to
        # "Unknown" in age_to_group (since 75 is above MAX_AGE = 74 and
        # agents are removed at MAX_AGE+1).
        if age == 75:
            assert numba_label == "75"
            assert python_label == "Unknown"
        else:
            assert numba_label == python_label, (
                f"age {age}: numba says {numba_label!r}, " f"python says {python_label!r}"
            )

    def test_below_min_age_maps_to_unknown(self):
        ages = np.array([0, 10, 15], dtype=np.int16)
        idx = fast_digitise_age_group(ages)
        assert np.all(idx == 7)  # the "Unknown" bucket

    def test_far_above_max_age_maps_to_unknown(self):
        ages = np.array([76, 100, 120], dtype=np.int16)
        idx = fast_digitise_age_group(ages)
        assert np.all(idx == 7)

    def test_handles_array_input(self):
        ages = np.array([16, 25, 50, 70, 16, 80], dtype=np.int16)
        idx = fast_digitise_age_group(ages)
        np.testing.assert_array_equal(idx, [0, 1, 3, 5, 0, 7])


# ─── compute_failure_rates ────────────────────────────────────────────────────


class TestComputeFailureRates:
    def test_zero_duration_returns_one(self):
        """A partnership of zero duration has the full hazard multiplier."""
        rates = compute_failure_rates(durations=np.array([0]), alpha=1500.0, gamma=2.0)
        np.testing.assert_allclose(rates, [1.0])

    def test_decreases_monotonically_with_duration(self):
        durations = np.array([0, 100, 500, 1500, 5000])
        rates = compute_failure_rates(durations, alpha=1500.0, gamma=2.0)
        # strictly decreasing
        assert np.all(np.diff(rates) < 0)

    def test_at_d_equals_alpha_rate_is_2_to_minus_gamma(self):
        """At d = alpha, (1 + d/alpha) = 2, so the rate is 2**(-gamma)."""
        rates = compute_failure_rates(durations=np.array([1500]), alpha=1500.0, gamma=2.0)
        np.testing.assert_allclose(rates, [0.25], rtol=1e-10)

    def test_default_parameters_match_legacy(self):
        """Pin down the expected rates at the calibrated alpha=1500, gamma=2.
        These numbers were quoted in the original code comments."""
        durations = np.array([0, 1500, 5000])
        rates = compute_failure_rates(durations, alpha=1500.0, gamma=2.0)
        # d=0: 1.0
        # d=1500: 0.25 (since 2**(-2) = 0.25)
        # d=5000: (1 + 5000/1500)^(-2) ≈ 0.0533
        np.testing.assert_allclose(rates, [1.0, 0.25, 0.0533], rtol=1e-2)

    def test_larger_alpha_means_slower_decay(self):
        d = np.array([2000])
        rate_short = compute_failure_rates(d, alpha=500.0, gamma=2.0)[0]
        rate_long = compute_failure_rates(d, alpha=5000.0, gamma=2.0)[0]
        assert rate_long > rate_short


# ─── compute_breakage_events ──────────────────────────────────────────────────


class TestComputeBreakageEvents:
    def test_low_uniforms_always_dissolve(self):
        """If uniform is well below the adjusted probability, it dissolves."""
        events = compute_breakage_events(
            durations=np.array([100, 100, 100]),
            base_breakage_probs=np.array([0.5, 0.5, 0.5]),
            alpha=1500.0,
            gamma=2.0,
            uniforms=np.array([0.0, 0.0, 0.0]),
        )
        assert events.tolist() == [True, True, True]

    def test_high_uniforms_never_dissolve(self):
        events = compute_breakage_events(
            durations=np.array([100, 100, 100]),
            base_breakage_probs=np.array([0.5, 0.5, 0.5]),
            alpha=1500.0,
            gamma=2.0,
            uniforms=np.array([0.99, 0.99, 0.99]),
        )
        assert events.tolist() == [False, False, False]

    def test_min_duration_guard(self):
        """Partnerships at or below min_duration never dissolve, regardless of
        the uniform draw — they're too young."""
        events = compute_breakage_events(
            durations=np.array([0, 1, 2]),
            base_breakage_probs=np.array([1.0, 1.0, 1.0]),
            alpha=1500.0,
            gamma=2.0,
            uniforms=np.array([0.0, 0.0, 0.0]),
            min_duration=1,
        )
        # only the partnership with duration=2 dissolves
        assert events.tolist() == [False, False, True]

    def test_long_partnerships_dissolve_less_often(self):
        """Same base prob and same uniform: a long partnership is less likely
        to dissolve than a short one."""
        durations = np.array([10, 5000])
        events = compute_breakage_events(
            durations=durations,
            base_breakage_probs=np.array([0.5, 0.5]),
            alpha=1500.0,
            gamma=2.0,
            uniforms=np.array([0.3, 0.3]),
        )
        # short one dissolves (0.3 < ~0.5*~1.0), long one doesn't (0.3 > ~0.5*~0.05)
        assert events[0]
        assert not events[1]

    def test_pure_function_deterministic(self):
        """Same inputs → same outputs, every call. This is the whole point
        of passing uniforms in rather than drawing internally."""
        inputs = dict(
            durations=np.array([10, 100, 1000]),
            base_breakage_probs=np.array([0.1, 0.3, 0.5]),
            alpha=1500.0,
            gamma=2.0,
            uniforms=np.array([0.05, 0.5, 0.2]),
        )
        a = compute_breakage_events(**inputs)
        b = compute_breakage_events(**inputs)
        np.testing.assert_array_equal(a, b)
