# partnersim-dynet

Agent-based dynamic partnership network simulation.

This package generates synthetic populations and tracks who is partnered
with whom, over time. It's designed as the partnership-network foundation
for sexually-transmitted infection (STI) modelling — disease dynamics
build on top of partnership dynamics. The package only produces 
partnership data which can be used to investigate STI transmission.

## Features

- Agent population with demographics (sex, age, orientation) and
  heterogeneity (per-agent NB multipliers on formation and breakage rates)
- Discrete-time partnership simulation: formation, dissolution,
  concurrency, ageing, and population replenishment
- Concurrency in partnerships
- Vectorised hazard-based partnership dissolution 
- Reproducible: every run is determined by a seed
- Network analysis: temporal network metrics, degree distributions
  stratified by demographics, snapshot and window-aggregated views
- Plots: timeseries, ego networks
- Diagnostic tools for inspecting probability calibration
- Multi-replicate execution with parallelisation

## Installation

This package uses [Poetry](https://python-poetry.org/) for dependency
management.

```bash
git clone https://github.com/pnt-id-models/partnersim-dynet.git
cd partnersim-dynet
poetry install
```
## Package structure
partnersim_dynet/
├── config/             — Calibration constants and dataclasses
├── generator/          — The partnership simulation engine
├── network/            — Temporal network analysis
│   └── plots/          — Figures
├── diagnostics/        — Probability calibration check
└── partnership_dynet_simulator.py        — Top-level run_single and run_replicates

## Quick start

Modify the parameters in src/partnersim-dynet/probabilities.py from the parameter 
sweeps produced by partnersim-dynet-lhs

```python

def _default_sex_multipliers() -> dict[str, float]:
    return {"Female": 1.0, "Male": 1.0}


def _default_orient_multipliers_form() -> dict[str, dict[str, float]]:
    """Formation orientation multipliers, conditional on sex."""
    return {
        "Female": {"Opposite-sex": 1.0, "Same-sex": 1.804, "Bisexual": 2.155},
        "Male": {"Opposite-sex": 4.778, "Same-sex": 1.569, "Bisexual": 1.197},
    }


def _default_orient_multipliers_break() -> dict[str, dict[str, float]]:
    """Breakage orientation multipliers, conditional on sex."""
    return {
        "Female": {"Opposite-sex": 1.0, "Same-sex": 0.964, "Bisexual": 1.024},
        "Male": {"Opposite-sex": 1.934, "Same-sex": 5.261, "Bisexual": 1.559},
    }

sex_multipliers: dict[str, float] = field(default_factory=_default_sex_multipliers)
orient_multipliers_form: dict[str, dict[str, float]] = field(
    default_factory=_default_orient_multipliers_form
)
orient_multipliers_break: dict[str, dict[str, float]] = field(
    default_factory=_default_orient_multipliers_break
)

# Age-related multipliers: youth boost + exponential decay
formation_youth_boost: float = 2.159
formation_age_decay: float = 0.300
breakage_youth_boost: float = 2.083
breakage_age_decay: float = 0.286
age_decay_baseline_group: str = "25-34"
```

A complete simulation with all outputs:

```python
from partnersim_dynet import run_single
from partnersim_dynet.config import PartnershipConfig

cfg = PartnershipConfig(
    num_agents=1500,
    total_timesteps=1875,
    concurrency_prop=0.15,
    concurrency_model=1,
)

result = run_single(
    cfg=cfg,
    seed=42,
    output_dir="results/single_run",
    run_metrics=True,
    run_degree_distributions=True,
    run_plots=True,
    run_diagnostics=True,
)

print(f"Wrote {len(result.files_written)} files to {result.output_dir}")
```

This produces:
- `partnerships.parquet` and `agent_log.parquet` — raw simulation output
- `metrics.parquet` — per-timestep network metrics
- `degree_by_demographic.parquet`, `degree_at_snapshots.parquet`,
  `degree_in_window.parquet` — degree distribution views
- `plots/` — timeseries and ego network figures
- `diagnostics/` — probability calibration tables and boxplots

## Multi-replicate experiments

```python
from partnersim_dynet import run_replicates
from partnersim_dynet.config import PartnershipConfig, SimulationConfig

sim_cfg = SimulationConfig(
    partnership=PartnershipConfig(num_agents=1500, total_timesteps=1875),
    n_partnership_replicates=20,
    base_partnership_seed=2026,
    n_workers=4,
    run_metrics=True,
    run_plots=False,           # heavy; skip per-replicate plots
    run_diagnostics=False,
)

results = run_replicates(sim_cfg, base_output_dir="results/sweep")
```

Each replicate writes to `results/sweep/partnership_seed_<N>/`. Seeds are
derived deterministically from `base_partnership_seed`, so the same
config always produces the same set of runs.
Each module is independent: the network module reads partnership +
agent_log DataFrames (no dependency on the generator's internals); the
diagnostics module reads agent_log + PartnershipConfig (no dependency on
generator state). This means analysis can run on saved outputs from any
source, not just live generator instances.

## Running tests (Work in progress)

Fast unit and integration tests (default):

```bash
poetry run pytest
```

Include slow benchmarks:

```bash
poetry run pytest -m "slow or not slow"
```

Only benchmarks (without coverage instrumentation):

```bash
poetry run pytest -m benchmark --no-cov
```

## Configuration

The main configuration class is `PartnershipConfig`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_agents` | 15000 | Population size (held constant via replenishment) |
| `total_timesteps` | 1875 | Simulation length |
| `concurrency_prop` | 0.15 | Fraction of population flagged for concurrency |
| `concurrency_model` | 1 | Concurrency replenishment model (1, 2, or 3) |
| `nb_r`, `nb_p` | 5, 0.8 | Negative-binomial heterogeneity params |
| `dissolution_alpha` | 1500 | Log-logistic scale parameter |
| `dissolution_gamma` | 2 | Log-logistic shape parameter |
| `age_difference_scale` | 4.0 | SD of age-difference partner kernel |

See `src/partnersim_dynet/config/simulation.py` for the full list.

Calibrated probability tables (formation/breakage by sex × orientation
× age group) are in `src/partnersim_dynet/config/probabilities.py`.


## Citation

If you use this package in research, please cite [your paper/preprint
when available].

## Companion packages

- [`partnersim-dynet-sti`](https://github.com/pnt-id-models/partnersim-dynet-sti) — sexually transmitted infection models that run on partnerships generated by this package.
- [`partnerim-dynet-lhs`](https://github.com/pnt-id-models/partnersim-dynet-lhs) — Latin Hypercube Sampling experiment for parameter sweeps.

## License

MIT
