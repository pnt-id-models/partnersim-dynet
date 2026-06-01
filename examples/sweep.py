"""Multi-replicate example: parallel replicates of one config.

Demonstrates how to vary the seed across N replicates and run them in
parallel. Outputs go to one subdirectory per replicate.

Run with:
    poetry run python examples/sweep.py
"""

from __future__ import annotations

from pathlib import Path

from partnersim_dynet import run_replicates
from partnersim_dynet.config import PartnershipConfig, SimulationConfig


def main() -> None:
    sim_cfg = SimulationConfig(
        partnership=PartnershipConfig(
            num_agents=300,
            total_timesteps=300,
        ),
        n_partnership_replicates=4,
        base_partnership_seed=2026,
        n_workers=2,
        verbose=True,
        run_metrics=True,
        run_plots=False,       # too heavy per-replicate
        run_diagnostics=False,
    )

    output_dir = Path(__file__).parent / "output" / "sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {sim_cfg.n_partnership_replicates} replicates "
          f"with {sim_cfg.n_workers} workers")
    print(f"Output: {output_dir}")
    print()

    results = run_replicates(sim_cfg, str(output_dir))

    print()
    print(f"Completed {len(results)} replicates:")
    for r in results:
        print(f"  seed={r.seed:>10d}  partnerships={r.n_partnerships:>5d}")


if __name__ == "__main__":
    main()