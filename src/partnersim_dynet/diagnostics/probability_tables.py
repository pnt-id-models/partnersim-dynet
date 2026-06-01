"""Probability table inspection — base rates and effective bounds.

1. ``print_probability_table`` / ``save_probability_table``: the
   calibrated base probabilities from a ProbabilityConfig, formatted as
   a human-readable table. These are population-average values,
   independent of any specific agent's NB multiplier.

2. ``export_probability_bounds`` / ``export_probability_bounds_csv``:
   the EFFECTIVE per-(AgeGroup, Sex, Orientation) bounds after
   accounting for agent-specific NB multipliers from a real simulation
   run. Tells you the actual range of probabilities agents experienced.

"""

from __future__ import annotations

import csv
import os

import pandas as pd

from partnersim_dynet.config import AGE_GROUPS, PartnershipConfig


# Base probability tables (from config)

def print_probability_table(
    cfg: PartnershipConfig,
    include_breakage: bool = True,
) -> None:
    """Print the calibrated formation (and optionally breakage) probabilities.

    Uses the base tables from ``cfg.probabilities`` 
    No per-agent heterogeneity is applied at this stage. Useful for sanity-checking that the
    multiplicative model produced the expected rates

    Parameters
    ----------
    cfg : PartnershipConfig
        Configuration whose ``probabilities`` field is used.
    include_breakage : bool
        If True, also print breakage probabilities alongside formation.
    """
    formation = cfg.probabilities.build_formation_probs()
    breakage = cfg.probabilities.build_breakage_probs() if include_breakage else None

    print("\n" + "=" * 64)
    print("Calibrated probabilities (base rates, no agent heterogeneity)")
    print("=" * 64)

    for sex in formation:
        print(f"\nSex: {sex}")
        for ori in formation[sex]:
            print(f"  Orientation: {ori}")
            if breakage is not None:
                print("    Age group     | Formation prob | Breakage prob")
                print("    -------------------------------------------------")
            else:
                print("    Age group     | Formation prob")
                print("    -------------------------------")
            for age in AGE_GROUPS:
                f = formation[sex][ori][age]
                if breakage is not None:
                    b = breakage[sex][ori][age]
                    print(f"    {age:<13} | {f:>14.5f} | {b:>13.5f}")
                else:
                    print(f"    {age:<13} | {f:>14.5f}")


def save_probability_table(cfg: PartnershipConfig, output_path: str) -> str:
    """Save the calibrated probability tables to CSV.

    One row per (Type, Sex, Orientation, AgeGroup, Probability) tuple,
    where Type is either "Formation" or "Breakage".

    Parameters
    ----------
    cfg : PartnershipConfig
    output_path : str
        Where to write the CSV file. Parent directory is created if missing.

    Returns
    -------
    str
        The output path, for convenience.
    """
    formation = cfg.probabilities.build_formation_probs()
    breakage = cfg.probabilities.build_breakage_probs()

    rows: list[tuple] = []
    for sex in formation:
        for ori in formation[sex]:
            for age in AGE_GROUPS:
                rows.append(
                    ("Formation", sex, ori, age, formation[sex][ori][age])
                )
    for sex in breakage:
        for ori in breakage[sex]:
            for age in AGE_GROUPS:
                rows.append(
                    ("Breakage", sex, ori, age, breakage[sex][ori][age])
                )

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Sex", "Orientation", "AgeGroup", "Probability"])
        writer.writerows(rows)

    return output_path

# Effective probability bounds (from real simulation)

def export_probability_bounds(
    cfg: PartnershipConfig,
    agent_log: pd.DataFrame,
) -> pd.DataFrame:
    """Compute effective per-group probability bounds from a real run.

    For each (AgeGroup, Sex, Orientation) combo, computes:
      - Base formation/breakage probability (from cfg, no heterogeneity)
      - Min/max effective formation/breakage probability across agents
        in that combo (base × NB multiplier × high-activity multiplier,
        clipped to [prob_floor, prob_ceiling])
      - Count of agents in the combo

    Useful for understanding the actual range of probabilities agents
    experienced, since the NB heterogeneity multiplier can substantially
    spread the per-agent rates.

    Parameters
    ----------
    cfg : PartnershipConfig
        Configuration the simulation used (for base probabilities and
        clipping bounds).
    agent_log : DataFrame
        From ``PartnershipGenerator.get_agent_log()``. Must contain
        columns: Agent, Sex, Orientation, EntryAge, NBMultiplierForm,
        NBMultiplierBreak, HighActive.

    Returns
    -------
    DataFrame
        One row per (AgeGroup, Sex, Orientation) combo with columns:
        AgentCount, Formation_Base, Formation_Effective_Min,
        Formation_Effective_Max, Breakage_Base, Breakage_Effective_Min,
        Breakage_Effective_Max.
    """
    from partnersim_dynet.config import age_to_group

    required = {
        "Agent", "Sex", "Orientation", "EntryAge",
        "NBMultiplierForm", "NBMultiplierBreak", "HighActive",
    }
    missing = required - set(agent_log.columns)
    if missing:
        raise KeyError(f"agent_log missing columns: {sorted(missing)}")

    formation = cfg.probabilities.build_formation_probs()
    breakage = cfg.probabilities.build_breakage_probs()

    df = agent_log.copy()
    df["AgeGroup"] = df["EntryAge"].apply(age_to_group)

    def _effective(row, base_table: dict, mult_col: str) -> float:
        base = base_table.get(row["Sex"], {}).get(
            row["Orientation"], {}
        ).get(row["AgeGroup"], 0.0)
        prob = base * row[mult_col]
        if row["HighActive"]:
            prob *= cfg.high_activity_multiplier
        return max(cfg.prob_floor, min(prob, cfg.prob_ceiling))

    df["Formation_Effective"] = df.apply(
        lambda r: _effective(r, formation, "NBMultiplierForm"), axis=1
    )
    df["Breakage_Effective"] = df.apply(
        lambda r: _effective(r, breakage, "NBMultiplierBreak"), axis=1
    )

    grouped = df.groupby(
        ["AgeGroup", "Sex", "Orientation"], as_index=False
    ).agg(
        AgentCount=("Agent", "size"),
        Formation_Effective_Min=("Formation_Effective", "min"),
        Formation_Effective_Max=("Formation_Effective", "max"),
        Breakage_Effective_Min=("Breakage_Effective", "min"),
        Breakage_Effective_Max=("Breakage_Effective", "max"),
    )

    # Attach base rates separately (constant per combo)
    grouped["Formation_Base"] = grouped.apply(
        lambda r: formation.get(r["Sex"], {}).get(
            r["Orientation"], {}
        ).get(r["AgeGroup"], 0.0),
        axis=1,
    )
    grouped["Breakage_Base"] = grouped.apply(
        lambda r: breakage.get(r["Sex"], {}).get(
            r["Orientation"], {}
        ).get(r["AgeGroup"], 0.0),
        axis=1,
    )

    # Order columns logically
    return grouped[[
        "AgeGroup", "Sex", "Orientation", "AgentCount",
        "Formation_Base", "Formation_Effective_Min", "Formation_Effective_Max",
        "Breakage_Base", "Breakage_Effective_Min", "Breakage_Effective_Max",
    ]]


def export_probability_bounds_csv(
    cfg: PartnershipConfig,
    agent_log: pd.DataFrame,
    output_path: str,
) -> str:
    """Convenience: compute bounds and save to CSV in one call."""
    bounds = export_probability_bounds(cfg, agent_log)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    bounds.to_csv(output_path, index=False)
    return output_path