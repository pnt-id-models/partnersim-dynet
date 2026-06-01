"""Tests for SimulationConfig and PartnershipConfig validation + seed derivation."""

import pytest

from partnersim_dynet.config import (
    PartnershipConfig,
    ProbabilityConfig,
    SimulationConfig,
)


class TestPartnershipConfigDefaults:
    def test_defaults_are_valid(self):
        cfg = PartnershipConfig()
        assert cfg.num_agents > 0
        assert cfg.total_timesteps > 0
        assert isinstance(cfg.probabilities, ProbabilityConfig)

    def test_dissolution_defaults_match_legacy(self):
        cfg = PartnershipConfig()
        assert cfg.dissolution_alpha == 1500.0
        assert cfg.dissolution_gamma == 2.0

    def test_age_difference_scale_default(self):
        assert PartnershipConfig().age_difference_scale == 4.0

    def test_concurrency_model_3_threshold_standardised_at_10(self):
        # The old code had an inconsistency: >10 at init, >1.0 at replenishment.
        # We standardise on 10.0.
        assert PartnershipConfig().concurrency_model_3_nb_threshold == 10.0

    def test_high_activity_disabled_by_default(self):
        cfg = PartnershipConfig()
        assert cfg.high_activity_proportion == 0.0
        assert cfg.high_activity_multiplier == 10.0  # ready to use if enabled

    def test_probability_clipping_defaults(self):
        cfg = PartnershipConfig()
        assert cfg.prob_floor == 0.0001
        assert cfg.prob_ceiling == 0.99

    def test_concurrency_min_partner_cap_default(self):
        assert PartnershipConfig().concurrency_min_partner_cap == 2


class TestPartnershipConfigValidation:
    @pytest.mark.parametrize(
        "kwargs, msg",
        [
            # population & duration
            ({"num_agents": 0}, "num_agents"),
            ({"num_agents": -1}, "num_agents"),
            ({"total_timesteps": 0}, "total_timesteps"),
            # concurrency
            ({"concurrency_prop": -0.1}, "concurrency_prop"),
            ({"concurrency_prop": 1.5}, "concurrency_prop"),
            ({"concurrency_model": 0}, "concurrency_model"),
            ({"concurrency_model": 4}, "concurrency_model"),
            # heterogeneity
            ({"nb_p": 0.0}, "nb_p"),
            ({"nb_p": 1.5}, "nb_p"),
            ({"nb_r": 0.0}, "nb_r"),
            # dissolution
            ({"dissolution_alpha": 0}, "dissolution_alpha"),
            ({"dissolution_alpha": -1}, "dissolution_alpha"),
            ({"dissolution_gamma": 0}, "dissolution_gamma"),
            ({"dissolution_gamma": -0.5}, "dissolution_gamma"),
            # age-difference scale
            ({"age_difference_scale": 0}, "age_difference_scale"),
            ({"age_difference_scale": -1.0}, "age_difference_scale"),
            # concurrency
            ({"concurrency_min_partner_cap": 1}, "concurrency_min_partner_cap"),
            ({"concurrency_min_partner_cap": 0}, "concurrency_min_partner_cap"),
            ({"concurrency_model_3_nb_threshold": -0.1}, "concurrency_model_3_nb_threshold"),
            # high activity
            ({"high_activity_proportion": -0.1}, "high_activity_proportion"),
            ({"high_activity_proportion": 1.5}, "high_activity_proportion"),
            ({"high_activity_multiplier": 0}, "high_activity_multiplier"),
            ({"high_activity_multiplier": -1}, "high_activity_multiplier"),
            # clipping
            ({"prob_floor": 0}, "prob_floor"),  # must be > 0
            ({"prob_floor": -0.001}, "prob_floor"),
            ({"prob_ceiling": 1.01}, "prob_ceiling"),  # > 1
            ({"prob_floor": 0.5, "prob_ceiling": 0.3}, "prob_floor"),  # floor > ceiling
        ],
    )
    def test_invalid_values_rejected(self, kwargs, msg):
        with pytest.raises(ValueError, match=msg):
            PartnershipConfig(**kwargs)

    def test_concurrency_min_2_is_accepted(self):
        # Boundary: 2 is allowed, 1 is not.
        PartnershipConfig(concurrency_min_partner_cap=2)  # no raise

    def test_high_activity_proportion_boundaries(self):
        PartnershipConfig(high_activity_proportion=0.0)  # ok
        PartnershipConfig(high_activity_proportion=1.0)  # ok


class TestSeedDerivation:
    def test_seeds_are_deterministic(self):
        cfg1 = SimulationConfig(n_partnership_replicates=5, base_partnership_seed=42)
        cfg2 = SimulationConfig(n_partnership_replicates=5, base_partnership_seed=42)
        assert cfg1.partnership_seeds() == cfg2.partnership_seeds()

    def test_seeds_differ_with_different_bases(self):
        a = SimulationConfig(base_partnership_seed=1).partnership_seeds()
        b = SimulationConfig(base_partnership_seed=2).partnership_seeds()
        assert a != b

    def test_seed_count_matches_replicate_count(self):
        cfg = SimulationConfig(n_partnership_replicates=10)
        assert len(cfg.partnership_seeds()) == 10

    def test_seeds_are_unique(self):
        cfg = SimulationConfig(n_partnership_replicates=100, base_partnership_seed=7)
        seeds = cfg.partnership_seeds()
        assert len(set(seeds)) == len(seeds)


class TestSimulationConfigValidation:
    def test_invalid_output_format(self):
        with pytest.raises(ValueError, match="output_format"):
            SimulationConfig(output_format="hdf5")

    def test_zero_replicates_rejected(self):
        with pytest.raises(ValueError, match="n_partnership_replicates"):
            SimulationConfig(n_partnership_replicates=0)

    def test_zero_workers_rejected(self):
        with pytest.raises(ValueError, match="n_workers"):
            SimulationConfig(n_workers=0)


class TestSimulationConfigFlags:
    """The new analysis flags."""

    def test_all_flags_default_false(self):
        cfg = SimulationConfig()
        assert cfg.run_metrics is False
        assert cfg.run_degree_distributions is False
        assert cfg.run_plots is False
        assert cfg.run_diagnostics is False

    def test_flags_are_independently_settable(self):
        cfg = SimulationConfig(run_metrics=True, run_plots=True)
        assert cfg.run_metrics is True
        assert cfg.run_plots is True
        assert cfg.run_diagnostics is False
