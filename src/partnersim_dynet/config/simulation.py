"""Top-level simulation configuration.

`PartnershipConfig` controls a single simulation run which includes the
population size, duration, concurrency proportions.
`SimulationConfig` wraps it for multi-replicate experiments with deterministic seed.

All flags default to values that prioritises performance: partnership simulation always
runs, optional analyses default off. Enable per-experiment via the boolean flags below.

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from partnersim_dynet.config.probabilities import ProbabilityConfig


@dataclass
class PartnershipConfig:
    """Parameters for one partnership-network simulation run.

    Population & duration
    ─────────────────────
    num_agents : int
        Initial population size. Population stays approximately constant —
        agents removed at age MAX_AGE are immediately replaced by new
        agents at REPLENISHMENT_AGE.
    total_timesteps : int
        Number of discrete timesteps to simulate.

    Agent heterogeneity
    ───────────────────
    nb_r, nb_p : float
        Negative-binomial parameters for per-agent formation and breakage
        heterogeneity multipliers. Drawn once at agent creation.

    Concurrency
    ───────────
    concurrency_prop : float
        Proportion of agents flagged as allOwed to form concurrent partnerships.
        These agents can hold multiple partnerships simultaneously; non-concurrent
        agents are limited to one partnership at a time.
    lambda_concurrency : int
        Poisson mean for the per-agent concurrency cap.
    concurrency_min_partner_cap : int
        Floor for the per-agent concurrency cap, regardless of the Poisson
        draw. The effective cap is max(this, Poisson(lambda_concurrency)).
    concurrency_model : int
        Which concurrency-assignment scheme to use (1, 2, or 3). See
        ``partnersim_dynet.generator.concurrency`` for details.
        Default is 1, which is the simplest and most efficient.
    concurrency_model_3_nb_threshold : float
        For Model 3 only: only agents whose ``nb_mult_form`` exceeds this
        threshold are eligible for concurrency.

    Partnership formation dynamics
    ──────────────────────────────
    age_difference_scale : float
        Standard deviation of the Gaussian kernel weighting age differences
        when selecting a partner. Smaller values produce more age-assortative
        matching.

    Partnership dissolution dynamics
    ────────────────────────────────
    Partnerships dissolve with a duration-dependent hazard that decays as
    partnerships persist (long-running partnerships are increasingly
    stable). The hazard multiplier is:

        failure_rate(d) = (1 + d / alpha) ** (-gamma)

    With the defaults (alpha=1500, gamma=2), the multiplier is 1.0 at d=0,
    ~0.44 at d=1500, and ~0.06 at d=5000.

    dissolution_alpha : float
        Scale parameter (in timesteps). Larger = slower decay.
    dissolution_gamma : float
        Shape parameter. Larger = sharper decay once duration exceeds alpha.

    High-activity boost (Optional feature not yet active)
    ───────────────────
    high_activity_proportion : float
        Fraction of the population flagged as high-activity. Currently
        defaults to 0 (feature disabled). When > 0, the flagged agents'
        formation and breakage probabilities are scaled by
        ``high_activity_multiplier``.
    high_activity_multiplier : float
        Multiplier applied to both formation and breakage probabilities for
        high-activity agents.

    Probability clipping
    ────────────────────
    prob_floor, prob_ceiling : float
        Effective probabilities are clipped to [prob_floor, prob_ceiling]
        to prevent unrealistic extreme values.

    Probabilities & I/O
    ───────────────────
    probabilities : ProbabilityConfig
        Probability calibration. Override to run sensitivity sweeps.
    record_population_history : bool
        If True, write a per-agent-per-timestep population history file.
    """

    # population size & duration
    num_agents: int = 15000
    total_timesteps: int = 1875

    # individual-level agent heterogeneity
    nb_r: float = 0.5
    nb_p: float = 0.5

    # concurrency parameters
    concurrency_prop: float = 0.0
    lambda_concurrency: int = 4
    concurrency_min_partner_cap: int = 2
    concurrency_model: int = 1
    concurrency_model_3_nb_threshold: float = 10.0

    # formation dynamics
    age_difference_scale: float = 4.0

    # dissolution dynamics (log-logistic hazard)
    dissolution_alpha: float = 1500.0
    dissolution_gamma: float = 2.0

    # high-activity boost
    high_activity_proportion: float = 0.0
    high_activity_multiplier: float = 10.0

    # probability clipping
    prob_floor: float = 0.0001
    prob_ceiling: float = 0.99

    # probabilities & I/O
    probabilities: ProbabilityConfig = field(default_factory=ProbabilityConfig)
    record_population_history: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        # population & duration
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")

        # heterogeneity
        if not 0.0 < self.nb_p <= 1.0:
            raise ValueError("nb_p must be in (0, 1]")
        if self.nb_r <= 0:
            raise ValueError("nb_r must be positive")

        # concurrency
        if not 0.0 <= self.concurrency_prop <= 1.0:
            raise ValueError("concurrency_prop must be in [0, 1]")
        if self.lambda_concurrency < 0:
            raise ValueError("lambda_concurrency must be non-negative")
        if self.concurrency_min_partner_cap < 2:
            raise ValueError("concurrency_min_partner_cap must be at least 2")
        if self.concurrency_model not in (1, 2, 3):
            raise ValueError("concurrency_model must be 1, 2, or 3")
        if self.concurrency_model_3_nb_threshold < 0:
            raise ValueError("concurrency_model_3_nb_threshold must be non-negative")

        # formation
        if self.age_difference_scale <= 0:
            raise ValueError("age_difference_scale must be positive")

        # dissolution
        if self.dissolution_alpha <= 0:
            raise ValueError("dissolution_alpha must be positive")
        if self.dissolution_gamma <= 0:
            raise ValueError("dissolution_gamma must be positive")

        # high activity
        if not 0.0 <= self.high_activity_proportion <= 1.0:
            raise ValueError("high_activity_proportion must be in [0, 1]")
        if self.high_activity_multiplier <= 0:
            raise ValueError("high_activity_multiplier must be positive")

        # clipping
        if not 0.0 < self.prob_floor < self.prob_ceiling <= 1.0:
            raise ValueError(
                "prob_floor and prob_ceiling must satisfy 0 < prob_floor < prob_ceiling <= 1"
            )


@dataclass
class SimulationConfig:
    """Top-level config for orchestrated multi-replicate experiments.

    Generates deterministic seeds for each replicate from a base seed,
    so the same SimulationConfig always produces the same set of runs.
    Change ``base_partnership_seed`` to get a different but still
    reproducible batch.
    """

    partnership: PartnershipConfig = field(default_factory=PartnershipConfig)

    n_partnership_replicates: int = 1
    base_partnership_seed: int = 1000

    output_format: str = "parquet"  # "parquet" or "csv"
    verbose: bool = False

    n_workers: int = 1

    # Flags for analysis and output. Default to False for performance; enable per-experiment.
    run_metrics: bool = False
    run_degree_distributions: bool = False
    run_plots: bool = False
    run_diagnostics: bool = False
    run_structural_summary: bool = False
    run_summary_table: bool = False

    # Validation of the config parameters. Raises ValueError if any parameter is invalid.
    def __post_init__(self) -> None:
        if self.n_partnership_replicates <= 0:
            raise ValueError("n_partnership_replicates must be positive")
        if self.output_format not in ("parquet", "csv"):
            raise ValueError("output_format must be 'parquet' or 'csv'")
        if self.n_workers <= 0:
            raise ValueError("n_workers must be positive")

    # Generates a list of deterministic seeds for each replicate based on the base seed.
    def partnership_seeds(self) -> list[int]:
        """Deterministically derive per-replicate seeds from the base seed."""
        rng = np.random.default_rng(self.base_partnership_seed)
        return rng.integers(0, 2**31, size=self.n_partnership_replicates).tolist()
