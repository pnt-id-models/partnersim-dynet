"""Quickstart example: one simulation, all analysis outputs.

Run with:
    poetry run python examples/quickstart.py

Produces output in examples/output/, including:
- partnerships.parquet, agent_log.parquet (raw simulation)
- metrics.parquet, degree_*.parquet (network analysis)
- plots/ (8+ PNG/PDF figures)
- diagnostics/ (for inpecting probability distributions)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from partnersim_dynet import run_single
from partnersim_dynet.config import PartnershipConfig


def main() -> None:
    cfg = PartnershipConfig(
        num_agents=1500,
        total_timesteps=1875,
        concurrency_prop=0.00,
        concurrency_model=1,
    )

    # Generate a unique output directory based on the feature name, concurrency percentage, number of agents, and current date.
    # If a directory with the same name already exists, increment a serial number until a unique name is found.
    FEATURE_NAME = "quick_test"
    concurrency_pct = round(cfg.concurrency_prop * 100)
    date_str = date.today().strftime("%d%b%Y")  # e.g. 11Aug2026
    stem = f"{FEATURE_NAME}_{concurrency_pct}pcconcurrency_{cfg.num_agents}agents_{date_str}"

    base_dir = Path(__file__).parent / "output" / "quickstart"
    serial = 1
    while (base_dir / f"{stem}_#{serial}").exists():
        serial += 1
    run_name = f"{stem}_#{serial}"

    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running simulation: {cfg.num_agents} agents × {cfg.total_timesteps} timesteps")
    print(f"Output: {output_dir}")
    print()

    result = run_single(
        cfg=cfg,
        seed=42,
        output_dir=str(output_dir),
        verbose=True,
        run_metrics=True,
        run_degree_distributions=True,
        run_plots=True,
        run_diagnostics=True,
    )

    print()
    print(f"Wrote {len(result.files_written)} files")
    print(f"Partnerships: {result.n_partnerships}")
    print(f"Agents ever: {result.n_agents}")


if __name__ == "__main__":
    main()
