"""Package-wide constants.

The source file for age groups, sex codes, orientation codes, and
related mappings. Every other module imports from here.
"""

from __future__ import annotations

"""
Age groups are used for age-based decay in formation probabilities, and for demographic structure in the partnership network. 
The age-to-group mapping is defined by AGE_GROUP_BOUNDARIES and the age_to_group() function. 
Inclusive age boundaries: an agent of age `a` belongs to group `label` if lo <= a <= hi. The "75+" group is the removal boundary — agents are removed from the simulation when they turn MAX_AGE.
"""

AGE_GROUP_BOUNDARIES: tuple[tuple[str, int, int], ...] = (
    ("16-24", 16, 24),
    ("25-34", 25, 34),
    ("35-44", 35, 44),
    ("45-54", 45, 54),
    ("55-64", 55, 64),
    ("65-74", 65, 74),
)

AGE_GROUPS: tuple[str, ...] = tuple(label for label, _, _ in AGE_GROUP_BOUNDARIES)

MIN_AGE: int = 16
MAX_AGE: int = 74  # agents are removed when they exceed this
REPLENISHMENT_AGE: int = 16  # new agents enter the simulation at this age


def age_to_group(age: int) -> str:
    """Map a numeric age to its age group label.

    Returns "Unknown" for ages outside the defined boundaries (e.g. below
    MIN_AGE or above MAX_AGE).  There should be no such agents in the simulation but this is a fallback to prevent crashes if the function is assigns an invalid age.
    """
    for label, lo, hi in AGE_GROUP_BOUNDARIES:
        if lo <= age <= hi:
            return label
    return "Unknown"


# Sex codes

# Internal representation uses int8 for memory efficiency; string forms are used in output DataFrames and plot labels.
SEX_CODE_TO_STR: dict[int, str] = {0: "Male", 1: "Female"}
SEX_STR_TO_CODE: dict[str, int] = {v: k for k, v in SEX_CODE_TO_STR.items()}


# Orientation codes
# Internal representation uses int8 for memory efficiency; string forms are used in output DataFrames and plot labels.

ORI_CODE_TO_STR: dict[int, str] = {
    0: "Opposite-sex",
    1: "Same-sex",
    2: "Bisexual",
}
ORI_STR_TO_CODE: dict[str, int] = {v: k for k, v in ORI_CODE_TO_STR.items()}


# Demographic distributions
"""
These describe the demographic structure of the population: how many of each sex,
how orientations are distributed within each sex, and at what age people enter sexual activity. 

They are constants — not config — because they are not expected to vary betweeen experiments. 

If a future experiment needs to vary them to depict a particular population, then they can be promoted to config parameters.
"""

PROPORTION_MALE: float = 0.5

# Orientation priors are conditional on sex. The tuple order is:
# (Opposite-sex, Same-sex, Bisexual), matching ORI_CODE_TO_STR.
# Each tuple must sum to 1.0.
ORIENTATION_PRIORS_MALE: tuple[float, float, float] = (0.90, 0.05, 0.05)
ORIENTATION_PRIORS_FEMALE: tuple[float, float, float] = (0.80, 0.10, 0.10)

# Sexual debut probabilities by age. Agents below age 16 are not in the population; agents at age 21+ are guaranteed to be sexually active.

# Between 16 and 20 inclusive, debut is probabilistic per year.
SEXUAL_DEBUT_PROBABILITIES: dict[int, float] = {
    16: 0.50,
    17: 0.20,
    18: 0.15,
    19: 0.10,
    20: 0.05,
}

# Age at which sexual debut becomes certain (probability 1.0).
GUARANTEED_DEBUT_AGE: int = 21
