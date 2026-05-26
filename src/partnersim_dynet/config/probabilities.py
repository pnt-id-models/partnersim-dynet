"""Probability configuration for partnership formation and breakage.

The probability model is multiplicative:

    P(action | sex, orientation, age_group) = base
                                              × sex_multiplier[sex]
                                              × orientation_multiplier[sex][orientation]
                                              × age_multiplier[age_group]

Reference category (all multipliers = 1.0): Female, Opposite-sex, age 16-24.

In the simulation, the resulting probability is further modulated per-agent
by a negative-binomial multiplier (`nb_mult_form` / `nb_mult_break`), drawn
once at agent creation to introduce individual-level heterogeneity.
This happens inside the PartnershipGenerator and is not part of this config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from partnersim_dynet.config.constants import AGE_GROUPS

# Type alias for the nested probability tables produced by build_*_probs().
# Shape: {sex: {orientation: {age_group: probability}}}.
ProbabilityTable = dict[str, dict[str, dict[str, float]]]


def _default_sex_multipliers() -> dict[str, float]:
    return {"Female": 1.0, "Male": 1.0}


def _default_orient_multipliers_form() -> dict[str, dict[str, float]]:
    """Formation orientation multipliers, conditional on sex."""
    return {
        "Female": {"Opposite-sex": 1.0, "Same-sex": 4.604, "Bisexual": 2.565},
        "Male": {"Opposite-sex": 4.374, "Same-sex": 3.793, "Bisexual": 1.307},
    }


def _default_orient_multipliers_break() -> dict[str, dict[str, float]]:
    """Breakage orientation multipliers, conditional on sex."""
    return {
        "Female": {"Opposite-sex": 1.0, "Same-sex": 0.882, "Bisexual": 2.562},
        "Male": {"Opposite-sex": 3.774, "Same-sex": 4.792, "Bisexual": 4.096},
    }


@dataclass
class ProbabilityConfig:
    """Parameters controlling formation and breakage probabilities.

    Use :meth:`build_formation_probs` and
    :meth:`build_breakage_probs` to materialise the full probability tables
    consumed by ``PartnershipGenerator``.

    Parameters
    ----------
    formation_base, breakage_base : float
        Per-timestep partnership probability for the reference category (Female,
        Opposite-sex, 16-24). All other groups scale relative to this.
    sex_multipliers : dict
        Multiplier per sex, applied to both formation and breakage.
    orient_multipliers_form, orient_multipliers_break : dict
        Multiplier per (sex, orientation) pair. Formation and breakage have
        independent values because the underlying behavioural drivers
        differ.
    formation_youth_boost, breakage_youth_boost : float
        Multiplier specifically for the 16-24 age group, applied instead of
        the exponential decay below. Captures the empirically observed
        partnership turnover spike in early adulthood
    formation_age_decay, breakage_age_decay : float
        Exponential decay rate for ages 25+. The multiplier for the k-th
        age group above the decay baseline is exp(-decay * k).
    age_decay_baseline_group : str
        Which age group starts the exponential decay at 1.0. Defaults to
        "25-34" so the youth boost is kept separate.
    """

    formation_base: float = 0.002
    breakage_base: float = 0.002

    sex_multipliers: dict[str, float] = field(default_factory=_default_sex_multipliers)
    orient_multipliers_form: dict[str, dict[str, float]] = field(
        default_factory=_default_orient_multipliers_form
    )
    orient_multipliers_break: dict[str, dict[str, float]] = field(
        default_factory=_default_orient_multipliers_break
    )

    formation_youth_boost: float = 3.133
    formation_age_decay: float = 0.313
    breakage_youth_boost: float = 2.059
    breakage_age_decay: float = 0.537
    age_decay_baseline_group: str = "25-34"

    def __post_init__(self) -> None:
        self._validate()

    # ── validation ──────────────────────────────────────────────────────

    def _validate(self) -> None:
        if self.formation_base < 0 or self.breakage_base < 0:
            raise ValueError("base probabilities must be non-negative")
        if self.formation_age_decay < 0 or self.breakage_age_decay < 0:
            raise ValueError("age decay rates must be non-negative")
        if self.age_decay_baseline_group not in AGE_GROUPS:
            raise ValueError(
                f"age_decay_baseline_group must be one of {AGE_GROUPS}, "
                f"got {self.age_decay_baseline_group!r}"
            )
        for sex, mult in self.sex_multipliers.items():
            if sex not in ("Male", "Female"):
                raise ValueError(f"unknown sex key {sex!r}; expected 'Male' or 'Female'")
            if mult < 0:
                raise ValueError(f"sex multiplier for {sex} must be non-negative")

    # ── age multiplier construction ─────────────────────────────────────

    def _build_age_multipliers(self, youth_boost: float, decay: float) -> dict[str, float]:
        """Construct {age_group: multiplier} via youth boost + exp decay."""
        baseline_idx = AGE_GROUPS.index(self.age_decay_baseline_group)
        multipliers: dict[str, float] = {}
        for i, age_group in enumerate(AGE_GROUPS):
            if age_group == "16-24":
                multipliers[age_group] = float(youth_boost)
            else:
                k = i - baseline_idx
                multipliers[age_group] = float(np.exp(-decay * k))
        return multipliers

    def formation_age_multipliers(self) -> dict[str, float]:
        """Age multipliers for formation probabilities."""
        return self._build_age_multipliers(self.formation_youth_boost, self.formation_age_decay)

    def breakage_age_multipliers(self) -> dict[str, float]:
        """Age multipliers for breakage probabilities."""
        return self._build_age_multipliers(self.breakage_youth_boost, self.breakage_age_decay)

    # ── full probability table construction ─────────────────────────────

    def _build_table(
        self,
        base: float,
        orient_multipliers: dict[str, dict[str, float]],
        age_multipliers: dict[str, float],
    ) -> ProbabilityTable:
        """Apply the multiplicative model across all (sex, ori, age) cells."""
        sexes = ("Male", "Female")
        orientations = ("Opposite-sex", "Same-sex", "Bisexual")
        table: ProbabilityTable = {}
        for sex in sexes:
            table[sex] = {}
            for ori in orientations:
                table[sex][ori] = {}
                for age_group in AGE_GROUPS:
                    prob = (
                        base
                        * self.sex_multipliers.get(sex, 1.0)
                        * orient_multipliers.get(sex, {}).get(ori, 1.0)
                        * age_multipliers.get(age_group, 1.0)
                    )
                    table[sex][ori][age_group] = max(prob, 0.0)
        return table

    def build_formation_probs(self) -> ProbabilityTable:
        """Full formation probability table {sex: {ori: {age_group: p}}}."""
        return self._build_table(
            self.formation_base,
            self.orient_multipliers_form,
            self.formation_age_multipliers(),
        )

    def build_breakage_probs(self) -> ProbabilityTable:
        """Full breakage probability table {sex: {ori: {age_group: p}}}."""
        return self._build_table(
            self.breakage_base,
            self.orient_multipliers_break,
            self.breakage_age_multipliers(),
        )
