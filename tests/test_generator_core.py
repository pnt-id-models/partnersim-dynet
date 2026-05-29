"""End-to-end smoke tests for PartnershipGenerator.

These tests run small, fast simulations and check structural properties of
the output. They don't validate epidemiological correctness — that's the
job of long simulation runs with statistical comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partnersim_dynet.config import (
    GUARANTEED_DEBUT_AGE,
    MAX_AGE,
    PartnershipConfig,
)
from partnersim_dynet.generator.core import PartnershipGenerator


# Tiny-simulation fixture


@pytest.fixture
def small_sim() -> tuple[PartnershipGenerator, pd.DataFrame]:
    """A small simulation: 200 agents, 100 timesteps. Fast enough for unit tests."""
    cfg = PartnershipConfig(num_agents=200, total_timesteps=100)
    gen = PartnershipGenerator(cfg, seed=42)
    df = gen.simulate_partnerships()
    return gen, df


# Construction

class TestConstruction:
    def test_default_construction(self):
        cfg = PartnershipConfig(num_agents=100, total_timesteps=10)
        gen = PartnershipGenerator(cfg, seed=42)
        assert gen.cfg.num_agents == 100
        assert int(gen.active.sum()) == 100

    def test_initial_agent_count_matches_config(self):
        cfg = PartnershipConfig(num_agents=137, total_timesteps=10)
        gen = PartnershipGenerator(cfg, seed=0)
        assert int(gen.active.sum()) == 137

    def test_agent_log_populated_at_construction(self):
        cfg = PartnershipConfig(num_agents=50, total_timesteps=10)
        gen = PartnershipGenerator(cfg, seed=0)
        log = gen.get_agent_log()
        assert len(log) == 50
        # All initial-cohort agents have EntryTimestep == 1
        assert (log["EntryTimestep"] == 1).all()

    def test_concurrent_agents_assigned(self):
        cfg = PartnershipConfig(
            num_agents=200,
            total_timesteps=10,
            concurrency_prop=0.10,
            concurrency_model=1,
        )
        gen = PartnershipGenerator(cfg, seed=0)
        # ~10% of 200 = 20 (allowing for rounding)
        assert 15 <= len(gen.active_concurrent_ids) <= 25



# Output shape

class TestOutputStructure:
    def test_partnership_df_columns(self, small_sim):
        _, df = small_sim
        expected = {
            "Agent",
            "AgentSex",
            "AgentOrientation",
            "AgentAge",
            "PartnerAgent",
            "PartnerSex",
            "PartnerOrientation",
            "PartnerAge",
            "StartTime",
            "EndTime",
            "Duration",
            "RelationshipType",
            "Censored",
            "ExternalPartner",
        }
        assert set(df.columns) == expected

    def test_every_agent_appears_in_partnership_df(self, small_sim):
        gen, df = small_sim
        # Every agent ever in the simulation must have at least one row (real partnership, censored, or singleton).
        total_agents = gen.next_agent_id - 1
        unique_in_df = df["Agent"].nunique()
        assert unique_in_df == total_agents

    def test_agent_log_columns(self, small_sim):
        gen, _ = small_sim
        log = gen.get_agent_log()
        expected = {
            "Agent",
            "Sex",
            "Orientation",
            "EntryAge",
            "EntryTimestep",
            "ExitTimestep",
            "ExitAge",
            "NBMultiplierForm",
            "NBMultiplierBreak",
            "HighActive",
            "ConcurrencyAllowed",
            "ConcurrencyCap",
        }
        assert set(log.columns) == expected

    def test_agent_log_has_one_row_per_unique_agent(self, small_sim):
        gen, _ = small_sim
        log = gen.get_agent_log()
        assert log["Agent"].nunique() == len(log)

    def test_singletons_have_none_relationship_type(self, small_sim):
        gen, df = small_sim
        singletons = df[df["RelationshipType"] == "None"]
        # Singletons should have NaN partner fields
        assert singletons["PartnerAgent"].isna().all()
        assert singletons["StartTime"].isna().all()

    def test_agent_log_concurrency_matches_generator_state(self):
            """Initial-cohort concurrency status in the log must match the
            generator's runtime state. Catches a class of ordering bugs where
            concurrency is assigned after the log entries are written."""
            cfg = PartnershipConfig(
                num_agents=200, total_timesteps=10,
                concurrency_prop=0.15, concurrency_model=1,
            )
            gen = PartnershipGenerator(cfg, seed=42)
            log = gen.get_agent_log()

            # State and log must agree on the count.
            state_count = len(gen.active_concurrent_ids)
            log_count = int(log["ConcurrencyAllowed"].sum())
            assert state_count == log_count, (
                f"State has {state_count} concurrent agents but log has {log_count}"
            )

            # Every concurrent agent in the state must have ConcurrencyCap set.
            log_concurrent = log[log["ConcurrencyAllowed"]]
            assert log_concurrent["ConcurrencyCap"].notna().all()


# Invariants the simulation should maintain

class TestInvariants:
    def test_population_stays_constant(self, small_sim):
        gen, _ = small_sim
        assert int(gen.active.sum()) == gen.cfg.num_agents

    def test_no_agent_above_max_age(self, small_sim):
        gen, _ = small_sim
        for idx in np.where(gen.active)[0]:
            assert gen.age_arr[idx] <= MAX_AGE

    def test_agent_log_exit_implies_inactive(self, small_sim):
        gen, _ = small_sim
        log = gen.get_agent_log()
        exited = log[log["ExitTimestep"].notna()]
        for aid in exited["Agent"]:
            # Removed agents shouldn't be in id2idx anymore
            assert aid not in gen.id2idx

    def test_agent_log_active_implies_no_exit(self, small_sim):
        gen, _ = small_sim
        log = gen.get_agent_log()
        for idx in np.where(gen.active)[0]:
            aid = int(gen.idx2id[idx])
            row = log[log["Agent"] == aid].iloc[0]
            assert pd.isna(row["ExitTimestep"])
            assert pd.isna(row["ExitAge"])

    def test_relationship_types_are_valid(self, small_sim):
        _, df = small_sim
        valid = {"M-M", "F-F", "M-F", "None"}
        assert set(df["RelationshipType"].unique()).issubset(valid)



# Reproducibility

class TestReproducibility:
    def test_same_seed_produces_same_partnership_df(self):
        cfg = PartnershipConfig(num_agents=100, total_timesteps=50)
        gen_a = PartnershipGenerator(cfg, seed=123)
        gen_b = PartnershipGenerator(cfg, seed=123)
        df_a = gen_a.simulate_partnerships()
        df_b = gen_b.simulate_partnerships()
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_same_seed_produces_same_agent_log(self):
        cfg = PartnershipConfig(num_agents=100, total_timesteps=50)
        gen_a = PartnershipGenerator(cfg, seed=123)
        gen_b = PartnershipGenerator(cfg, seed=123)
        gen_a.simulate_partnerships()
        gen_b.simulate_partnerships()
        pd.testing.assert_frame_equal(gen_a.get_agent_log(), gen_b.get_agent_log())

    def test_different_seeds_produce_different_output(self):
        cfg = PartnershipConfig(num_agents=100, total_timesteps=50)
        gen_a = PartnershipGenerator(cfg, seed=1)
        gen_b = PartnershipGenerator(cfg, seed=2)
        df_a = gen_a.simulate_partnerships()
        df_b = gen_b.simulate_partnerships()
        # They should differ in at least one cell
        assert not df_a.equals(df_b)


# Sexual debut behaviour

class TestSexualDebut:
    def test_agents_above_guaranteed_age_are_sexually_active(self):
        cfg = PartnershipConfig(num_agents=500, total_timesteps=10)
        gen = PartnershipGenerator(cfg, seed=0)
        # Every agent above the guaranteed age should be sexually active
        for idx in np.where(gen.active)[0]:
            if gen.age_arr[idx] >= GUARANTEED_DEBUT_AGE:
                assert gen.sexually_active_arr[idx], (
                    f"agent at idx {idx} aged {gen.age_arr[idx]} " "is not sexually active"
                )
