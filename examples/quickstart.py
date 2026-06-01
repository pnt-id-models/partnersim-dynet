"""Quickstart example: one simulation, all analysis outputs.

Run with:
    poetry run python examples/quickstart.py

Produces output in examples/output/, including:
- partnerships.parquet, agent_log.parquet (raw simulation)
- metrics.parquet, degree_*.parquet (network analysis)
- plots/ (8+ PNG/PDF figures)
- diagnostics/ (probability inspection)
"""

from __future__ import annotations

from pathlib import Path

from partnersim_dynet import run_single
from partnersim_dynet.config import PartnershipConfig


def main() -> None:
    cfg = PartnershipConfig(
        num_agents=500,
        total_timesteps=500,
        concurrency_prop=0.10,
        concurrency_model=1,
    )

    output_dir = Path(__file__).parent / "output" / "quickstart"
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
