"""Multi-replicate example: parallel replicates of one config.

Run with:
    poetry run python examples/sweep.py

To regenerate the full plot suite for a specific known replicate instead
of running a fresh sweep and randomly selecting one, set FORCE_SEED below.
"""

from __future__ import annotations

import random
from pathlib import Path

from partnersim_dynet import run_replicates, run_single
from partnersim_dynet.config import PartnershipConfig, SimulationConfig

# Set to a specific seed (e.g. 380863079) to skip the sweep/selection step
# and just regenerate plots for that known replicate. Leave as None for
# normal sweep-and-randomly-select behaviour.

# FORCE_SEED: int | None = 380863079
FORCE_SEED: int | None = None


def main() -> None:
    sim_cfg = SimulationConfig(
        partnership=PartnershipConfig(
            num_agents=1500,
            total_timesteps=1875,
            concurrency_prop=0.00,
        ),
        n_partnership_replicates=2,
        base_partnership_seed=2026,
        n_workers=8,
        verbose=True,
        run_metrics=True,
        run_plots=False,
        run_summary_table=True,
        run_diagnostics=False,
    )

    output_dir = Path(__file__).parent / "output" / "sweep" / "0pc_concurrency_11thAug_15kagents/"
    output_dir.mkdir(parents=True, exist_ok=True)

    if FORCE_SEED is not None:
        # ── Skip the sweep entirely; regenerate plots for a known seed ──
        selected_seed = FORCE_SEED
        # selected_n_partnerships = None  # unknown until we actually run it
        print(f"FORCE_SEED set: skipping sweep, using seed={selected_seed}")
    else:
        # ── Normal sweep: run N replicates, then randomly select one ──
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

    # ── Generate full plot suite for selected_seed ─────────────────────
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
