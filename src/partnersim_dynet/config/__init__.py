"""Configuration layer for partnersim_dynet.

All package-wide constants, probability tables, and simulation parameters
are defined here. This is the single source of truth for shared types and
default values — every other module imports from here rather than defining
its own copies.
"""

from partnersim_dynet.config.constants import (
    AGE_GROUP_BOUNDARIES,
    AGE_GROUPS,
    GUARANTEED_DEBUT_AGE,
    MAX_AGE,
    MIN_AGE,
    ORI_CODE_TO_STR,
    ORI_STR_TO_CODE,
    ORIENTATION_PRIORS_FEMALE,
    ORIENTATION_PRIORS_MALE,
    PROPORTION_MALE,
    REPLENISHMENT_AGE,
    SEX_CODE_TO_STR,
    SEX_STR_TO_CODE,
    SEXUAL_DEBUT_PROBABILITIES,
    age_to_group,
)
from partnersim_dynet.config.probabilities import ProbabilityConfig
from partnersim_dynet.config.simulation import (
    PartnershipConfig,
    SimulationConfig,
)

__all__ = [
    # constants
    "AGE_GROUPS",
    "AGE_GROUP_BOUNDARIES",
    "MAX_AGE",
    "MIN_AGE",
    "ORI_CODE_TO_STR",
    "ORI_STR_TO_CODE",
    "ORIENTATION_PRIORS_FEMALE",
    "ORIENTATION_PRIORS_MALE",
    "PROPORTION_MALE",
    "REPLENISHMENT_AGE",
    "SEX_CODE_TO_STR",
    "SEX_STR_TO_CODE",
    "age_to_group",
    "SEXUAL_DEBUT_PROBABILITIES",
    "GUARANTEED_DEBUT_AGE",
    # configs
    "ProbabilityConfig",
    "PartnershipConfig",
    "SimulationConfig",
]
