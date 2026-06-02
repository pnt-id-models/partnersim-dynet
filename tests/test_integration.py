"""Integration tests: medium-scale end-to-end simulation runs.

These tests run 500-agent, 500-timestep simulations to verify the
simulation produces structurally and statistically sensible output across
many timesteps. They're slower than unit tests (~10s each) but still fast
enough to run by default.

What's NOT tested here:
- Full-scale runs (1500+ agents, 1875 timesteps) — see test_benchmark.py
- Concurrency Models 2 and 3 — only Model 1 is exercised in integration
- Disease-model interactions — those live in the sti-dynet package
- Epidemiological calibration — that's a longer-run validation exercise

The tests assert *invariants* rather than specific numeric outputs, so
they tolerate small RNG-driven variation across NumPy versions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from partnersim_dynet.config import (
    MAX_AGE,
    MIN_AGE,
    PartnershipConfig,
)
from partnersim_dynet.generator import PartnershipGenerator

# Testing a 500-agent, 500-timestep simulation


@pytest.fixture(scope="module")
def integration_run() -> tuple[PartnershipGenerator, pd.DataFrame, pd.DataFrame]:
    """A single 500-agent, 500-timestep simulation, shared across tests.

    Using scope="module" means the simulation runs once per test file load, not once per test.
    """
    cfg = PartnershipConfig(
        num_agents=500,
        total_timesteps=500,
        concurrency_prop=0.10,
        concurrency_model=1,
    )
    gen = PartnershipGenerator(cfg, seed=42)
    partnerships = gen.simulate_partnerships()
    agent_log = gen.get_agent_log()
    return gen, partnerships, agent_log


# Population invariants over time


class TestPopulationInvariants:
    """The population should stay structurally consistent across the run."""

    def test_active_population_matches_config(self, integration_run):
        gen, _, _ = integration_run
        assert int(gen.active.sum()) == gen.cfg.num_agents

    def test_agents_added_equals_agents_removed(self, integration_run):
        """For a steady-state population, every removal triggers exactly
        one replenishment."""
        gen, _, log = integration_run
        n_initial = gen.cfg.num_agents
        n_total = gen.next_agent_id - 1
        n_replenishments = n_total - n_initial
        n_removed = log["ExitTimestep"].notna().sum()
        assert n_replenishments == n_removed

    def test_no_active_agent_above_max_age(self, integration_run):
        gen, _, _ = integration_run
        active_ages = gen.age_arr[gen.active]
        assert active_ages.max() <= MAX_AGE

    def test_all_active_agents_at_or_above_min_age(self, integration_run):
        gen, _, _ = integration_run
        active_ages = gen.age_arr[gen.active]
        assert active_ages.min() >= MIN_AGE


# Agent log


class TestAgentLogIntegrity:
    def test_every_agent_in_partnership_df_is_in_log(self, integration_run):
        gen, partnerships, log = integration_run
        # Every Agent in the partnership df must also appear in the log
        partnership_agents = set(partnerships["Agent"].dropna().astype(int))
        log_agents = set(log["Agent"].astype(int))
        assert partnership_agents.issubset(log_agents)

    def test_log_row_count_matches_next_agent_id(self, integration_run):
        gen, _, log = integration_run
        # n_initial + n_replenishments = next_agent_id - 1
        assert len(log) == gen.next_agent_id - 1

    def test_initial_cohort_entry_timestep_is_one(self, integration_run):
        gen, _, log = integration_run
        initial = log.iloc[: gen.cfg.num_agents]
        assert (initial["EntryTimestep"] == 1).all()

    def test_replenishments_have_entry_age_equal_to_replenishment_age(self, integration_run):
        gen, _, log = integration_run
        replenishments = log.iloc[gen.cfg.num_agents :]
        if len(replenishments) > 0:
            assert (replenishments["EntryAge"] == MIN_AGE).all()

    def test_exited_agents_have_consistent_exit_age(self, integration_run):
        """Exit age should equal the agent's age at the moment of removal,
        which is MAX_AGE + 1 (the trigger condition for removal)."""
        _, _, log = integration_run
        exited = log[log["ExitTimestep"].notna()]
        # Removal triggers when age > MAX_AGE, so exit age should be MAX_AGE + 1
        assert (exited["ExitAge"] == MAX_AGE + 1).all()


# Partnership dynamics


class TestPartnershipDynamics:
    def test_partnerships_are_formed(self, integration_run):
        """Over 500 timesteps, some partnerships must form."""
        _, partnerships, _ = integration_run
        real = partnerships[partnerships["RelationshipType"] != "None"]
        assert len(real) > 0

    def test_partnerships_have_non_negative_duration(self, integration_run):
        # Partnerships that form in the final timestep have Duration == 0
        # (censored before any time elapses).
        _, partnerships, _ = integration_run
        real = partnerships[partnerships["RelationshipType"] != "None"]
        assert (real["Duration"] >= 0).all()

    def test_partnerships_have_consistent_start_end(self, integration_run):
        _, partnerships, _ = integration_run
        real = partnerships[partnerships["RelationshipType"] != "None"]
        assert (real["EndTime"] >= real["StartTime"]).all()
        assert (real["Duration"] == real["EndTime"] - real["StartTime"]).all()

    def test_some_partnerships_dissolve(self, integration_run):
        """Not every partnership should be censored at end-of-simulation."""
        _, partnerships, _ = integration_run
        real = partnerships[partnerships["RelationshipType"] != "None"]
        n_dissolved = (~real["Censored"]).sum()
        assert n_dissolved > 0

    def test_no_self_partnerships(self, integration_run):
        _, partnerships, _ = integration_run
        real = partnerships[partnerships["PartnerAgent"].notna()]
        # Cast partner agent to int for comparison
        assert (real["Agent"].astype(int) != real["PartnerAgent"].astype(int)).all()


# Demographic compatibility


class TestPartnershipCompatibility:
    """Partnerships should respect orientation rules: opposite-sex agents
    only partner the opposite sex, same-sex only partner the same sex, and
    bisexuals may partner either."""

    def test_no_opposite_sex_agent_in_same_sex_partnership(self, integration_run):
        _, partnerships, _ = integration_run
        real = partnerships[partnerships["RelationshipType"] != "None"].copy()

        # Same-sex partnerships: both agents must be Same-sex OR Bisexual
        same_sex = real[real["RelationshipType"].isin(["M-M", "F-F"])]
        valid_orientations = {"Same-sex", "Bisexual"}
        assert same_sex["AgentOrientation"].isin(valid_orientations).all()

    def test_no_same_sex_agent_in_opposite_sex_partnership(self, integration_run):
        _, partnerships, _ = integration_run
        real = partnerships[partnerships["RelationshipType"] != "None"].copy()

        opposite_sex = real[real["RelationshipType"] == "M-F"]
        valid_orientations = {"Opposite-sex", "Bisexual"}
        assert opposite_sex["AgentOrientation"].isin(valid_orientations).all()


# Reproducibility
class TestIntegrationReproducibility:
    def test_full_simulation_is_deterministic(self):
        """A full simulation should produce bit-identical output across runs
        with the same seed. This is the strongest reproducibility guarantee."""
        cfg = PartnershipConfig(
            num_agents=300,
            total_timesteps=300,
            concurrency_prop=0.1,
            concurrency_model=1,
        )

        gen_a = PartnershipGenerator(cfg, seed=2026)
        df_a = gen_a.simulate_partnerships()
        log_a = gen_a.get_agent_log()

        gen_b = PartnershipGenerator(cfg, seed=2026)
        df_b = gen_b.simulate_partnerships()
        log_b = gen_b.get_agent_log()

        pd.testing.assert_frame_equal(df_a, df_b)
        pd.testing.assert_frame_equal(log_a, log_b)
