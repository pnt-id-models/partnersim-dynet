"""Numerical helpers for the performance of thepartnership simulator.


1. **Numba JIT helpers** (``fast_normal_pdf``, ``fast_digitise_age_group``).
   These are called in tight loops where Python overhead dominates;
   ``@njit(cache=True)`` compiles them on first call and
   caches the binary on disk for subsequent runs.

2. **Vectorized NumPy helpers** for the breakage phase
   (``compute_failure_rates``, ``draw_breakage_events``).
   These functions operate on whole arrays at once so the inner loop runs in C.

"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

# Numba JIT helpers


@njit(cache=True)
def fast_normal_pdf(
    x: NDArray[np.float64],
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Numba-compiled normal PDF for partner-age weighting.

    Used in the partner-selection step to weight candidate partners by
    age difference. Equivalent to ``scipy.stats.norm.pdf`` but ~100x
    faster in the tight loop because there's no Python overhead.
    """
    return np.exp(-0.5 * ((x - loc) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))


@njit(cache=True)
def fast_digitise_age_group(ages: NDArray[np.int16]) -> NDArray[np.int32]:
    """Map an array of ages to age-group indices.

    Returns
    -------
    NDArray[np.int32]
        Index into the AGE_GROUPS tuple from constants.py:
            0 = "16-24", 1 = "25-34", 2 = "35-44",
            3 = "45-54", 4 = "55-64", 5 = "65-74",
            6 = "75" (the removal-boundary group),
            7 = "Unknown" (out of range).

    Notes
    -----
    The boundary logic here MUST match ``age_to_group`` in
    ``config/constants.py``. Both use inclusive boundaries
    (``16 <= age <= 24``, etc.).
    """
    result = np.empty(len(ages), dtype=np.int32)
    for i in range(len(ages)):
        age = ages[i]
        if 16 <= age <= 24:
            result[i] = 0
        elif 25 <= age <= 34:
            result[i] = 1
        elif 35 <= age <= 44:
            result[i] = 2
        elif 45 <= age <= 54:
            result[i] = 3
        elif 55 <= age <= 64:
            result[i] = 4
        elif 65 <= age <= 74:
            result[i] = 5
        elif age == 75:
            result[i] = 6
        else:
            result[i] = 7
    return result


# Vectorised breakage helpers


def compute_failure_rates(
    durations: NDArray[np.int_],
    alpha: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Vectorised hazard multiplier.

    For an array of partnership durations, return the per-partnership
    multiplier that scales the agent's base breakage probability::

        failure_rate(d) = (1 + d / alpha) ** (-gamma)

    The multiplier starts at 1.0 (new partnership, full breakage rate)
    and decays toward 0 as duration grows — long-running partnerships
    are increasingly stable.

    Parameters
    ----------
    durations : ndarray of int
        Partnership durations in timesteps (typically ``t - start_time``).
    alpha : float
        Scale parameter. Larger values = slower decay.
    gamma : float
        Shape parameter. Larger values = sharper decay once duration
        exceeds alpha.
    """
    return (1.0 + durations / alpha) ** (-gamma)


def compute_breakage_events(
    durations: NDArray[np.int_],
    base_breakage_probs: NDArray[np.float64],
    alpha: float,
    gamma: float,
    uniforms: NDArray[np.float64],
    min_duration: int = 1,
) -> NDArray[np.bool_]:
    """Decide which partnerships dissolve this timestep, vectorised.

    For an array of (duration, base_prob) pairs, return a boolean mask
    where True means the partnership dissolves this step.

    Parameters
    ----------
    durations : ndarray of int
        Partnership durations in timesteps.
    base_breakage_probs : ndarray of float
        Per-partnership base breakage probability (already incorporating
        the focal agent's NB multiplier and high-activity boost).
    alpha, gamma : float
        Dissolution decay parameters.
    uniforms : ndarray of float
        Pre-drawn uniform random numbers in [0, 1), one per partnership.
        Passing these in (rather than drawing inside this function) keeps
        the function pure and deterministic for testing.
    min_duration : int
        Partnerships younger than this are never dissolved.

    Returns
    -------
    ndarray of bool
        True where the corresponding partnership dissolves.
    """
    failure_rates = compute_failure_rates(durations, alpha, gamma)
    adjusted = base_breakage_probs * failure_rates
    dissolves = uniforms < adjusted
    # Test the minimum-duration guard
    dissolves &= durations > min_duration
    return dissolves
