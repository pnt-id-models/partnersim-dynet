"""Multi-replicate example: to run multiple replicates of the same simulation,
and randomly select one for full analysis and plotting.

Run with:
    poetry run python examples/sweep.py

To regenerate the full plot suite for a specific known replicate instead
of running a fresh sweep and randomly selecting one, set FORCE_SEED below.
"""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

from partnersim_dynet import run_replicates, run_single
from partnersim_dynet.config import PartnershipConfig, SimulationConfig

# If a particular replicate is of interest then copy the seed number from the folder name,
# set to a specific seed (e.g. 380863079) to skip the sweep/selection step
# and just regenerate plots for that known replicate.
# Leave as None for regular multi-replicate sweep and random selection of one replicate for plotting.

# FORCE_SEED: int | None = 380863079
FORCE_SEED: int | None = None


def main() -> None:
    sim_cfg = SimulationConfig(
        partnership=PartnershipConfig(
            num_agents=1500,
            total_timesteps=1875,
            concurrency_prop=0.00,
        ),
        n_partnership_replicates=2,  # Note - Even though we can generate multiple replicates, we will only select one for full analysis and plotting.
        base_partnership_seed=2026,
        n_workers=8,
        verbose=True,
        run_metrics=True,
        run_plots=False,
        run_summary_table=True,
        run_diagnostics=False,
    )

    # Generate a unique output directory based on the feature name, concurrency percentage, number of agents, and current date.
    # If a directory with the same name already exists, increment a serial number until a unique name is found.
    FEATURE_NAME = "full_sweep"

    concurrency_pct = round(sim_cfg.partnership.concurrency_prop * 100)
    num_agents = sim_cfg.partnership.num_agents
    date_str = date.today().strftime("%d%b%Y")  # e.g. 11Aug2026
    stem = f"{FEATURE_NAME}_{concurrency_pct}pcconcurrency_{num_agents}agents_{date_str}"

    base_dir = Path(__file__).parent / "output" / "sweep"

    serial = 1
    while (base_dir / f"{stem}_#{serial}").exists():
        serial += 1
    run_name = f"{stem}_#{serial}"

    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if FORCE_SEED is not None:
        # If a specific seed is forced, skip the sweep and just use that seed for the full analysis and plotting.
        selected_seed = FORCE_SEED
        # selected_n_partnerships = None  # unknown until we actually run it
        print(f"FORCE_SEED set: skipping sweep, using seed={selected_seed}")
    else:
        # Without a specific seed, run the sweep to generate multiple replicates and randomly select one for full analysis and plotting.
        print(
            f"Running {sim_cfg.n_partnership_replicates} replicates "
            f"with {sim_cfg.n_workers} workers"
        )
        print(f"Output: {output_dir}")
        print()

        results = run_replicates(sim_cfg, str(output_dir))

        print()
        print(f"Completed {len(results)} replicates:")
        for r in results:
            print(f"  seed={r.seed:>10d}  partnerships={r.n_partnerships:>5d}")

        rng = random.Random(sim_cfg.base_partnership_seed)
        selected = rng.choice(results)
        selected_seed = selected.seed
        # selected_n_partnerships = selected.n_partnerships

        print()
        print(f"Selected replicate for network plots: seed={selected_seed}")

    # Generate the full plot suite for the selected replicate.
    # This will create a subdirectory for the selected replicate and run the simulation again to generate all plots and diagnostics.
    replicate_dir = output_dir / f"replicate_{selected_seed}"
    print("  Generating full plot suite ...")

    result = run_single(
        sim_cfg.partnership,
        seed=selected_seed,
        output_dir=str(replicate_dir),
        verbose=True,
        run_metrics=True,
        run_degree_distributions=True,
        run_plots=True,  # implies run_structural_summary=True
        run_diagnostics=True,
    )

    # Write a marker file in the main output directory to record which replicate was selected for full analysis and plotting.
    marker = output_dir / "selected_replicate.txt"
    marker.write_text(
        f"seed={selected_seed}\n"
        f"partnerships={result.n_partnerships}\n"
        f"replicate_dir=replicate_{selected_seed}\n"
        f"forced={FORCE_SEED is not None}\n"
    )

    print(f"  Done. Plots written to: {replicate_dir}")
    print(f"  Selection recorded in:  {marker}")


if __name__ == "__main__":
    main()
