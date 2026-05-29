"""Tests for AliveIntervals — alive-status lookups from the agent log."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partnersim_dynet.network import AliveIntervals

# Helpers
def _make_log(rows: list[dict]) -> pd.DataFrame:
    """Make a minimal agent log DataFrame for tests."""
    df = pd.DataFrame(rows)
    # Ensure column dtypes match what the generator emits
    df["Agent"] = df["Agent"].astype("int64")
    df["EntryTimestep"] = df["EntryTimestep"].astype("int32")
    # ExitTimestep can be float (NaN allowed)
    df["ExitTimestep"] = df["ExitTimestep"].astype("float64")
    return df

# Construction
class TestConstruction:
    def test_minimal_log(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1, "ExitTimestep": 500.0},
        ])
        alive = AliveIntervals.from_agent_log(log, total_timesteps=1000)
        assert len(alive) == 1
        assert alive.total_timesteps == 1000

    def test_nan_exit_becomes_sentinel(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1, "ExitTimestep": float("nan")},
        ])
        alive = AliveIntervals.from_agent_log(log, total_timesteps=1000)
        # total_timesteps + 1
        assert alive.exit_t[0] == 1001

    def test_missing_column_raises(self):
        log = pd.DataFrame({
            "Agent": [1, 2],
            "EntryTimestep": [1, 2],
            # ExitTimestep missing
        })
        with pytest.raises(ValueError, match="missing required columns"):
            AliveIntervals.from_agent_log(log, total_timesteps=100)

    def test_zero_total_timesteps_raises(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1, "ExitTimestep": 100.0},
        ])
        with pytest.raises(ValueError, match="total_timesteps"):
            AliveIntervals.from_agent_log(log, total_timesteps=0)

    def test_zero_entry_timestep_raises(self):
        # EntryTimestep should be 1-indexed per the generator's convention
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 0, "ExitTimestep": 100.0},
        ])
        with pytest.raises(ValueError, match="EntryTimestep"):
            AliveIntervals.from_agent_log(log, total_timesteps=100)

# Query semantics
class TestAliveQueries:
    @pytest.fixture
    def alive(self) -> AliveIntervals:
        # Three agents:
        #   1: alive throughout [1, 1000]   (active at end)
        #   2: alive during    [1, 500)     (exited at t=500)
        #   3: alive during    [200, 800)   (joined later, exited mid-run)
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1,   "ExitTimestep": float("nan")},
            {"Agent": 2, "EntryTimestep": 1,   "ExitTimestep": 500.0},
            {"Agent": 3, "EntryTimestep": 200, "ExitTimestep": 800.0},
        ])
        return AliveIntervals.from_agent_log(log, total_timesteps=1000)

    def test_alive_at_start(self, alive):
        assert alive.alive_at(1) == {1, 2}

    def test_alive_at_mid_run(self, alive):
        # At t=300, agents 1, 2 (still alive), and 3 (joined at 200)
        assert alive.alive_at(300) == {1, 2, 3}

    def test_exit_is_exclusive(self, alive):
        # Agent 2 exits at t=500. Alive at 499, not at 500.
        assert 2 in alive.alive_at(499)
        assert 2 not in alive.alive_at(500)

    def test_entry_is_inclusive(self, alive):
        # Agent 3 enters at t=200. Alive at 200, not at 199.
        assert 3 not in alive.alive_at(199)
        assert 3 in alive.alive_at(200)

    def test_alive_at_end(self, alive):
        # At t=1000, only agent 1 (the sentinel-exit one)
        assert alive.alive_at(1000) == {1}

    def test_is_alive_matches_alive_at(self, alive):
        for t in (1, 100, 199, 200, 499, 500, 800, 1000):
            for aid in (1, 2, 3):
                assert alive.is_alive(aid, t) == (aid in alive.alive_at(t))

    def test_is_alive_unknown_agent_returns_false(self, alive):
        assert alive.is_alive(999, t=50) is False

    def test_bounds_returns_entry_exit(self, alive):
        assert alive.bounds(1) == (1, 1001)   # sentinel for active-at-end
        assert alive.bounds(2) == (1, 500)
        assert alive.bounds(3) == (200, 800)

    def test_bounds_unknown_agent_returns_none(self, alive):
        assert alive.bounds(999) is None

# Array-form query

class TestAliveAtArray:
    def test_returns_ndarray(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1, "ExitTimestep": 100.0},
            {"Agent": 2, "EntryTimestep": 1, "ExitTimestep": 100.0},
        ])
        alive = AliveIntervals.from_agent_log(log, total_timesteps=200)
        arr = alive.alive_at_array(50)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(sorted(arr), [1, 2])

    def test_empty_when_nobody_alive(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 100, "ExitTimestep": 200.0},
        ])
        alive = AliveIntervals.from_agent_log(log, total_timesteps=500)
        # At t=50, agent 1 hasn't joined yet
        arr = alive.alive_at_array(50)
        assert len(arr) == 0

# Integration with generator

class TestIntegrationWithGenerator:
    """Spot-check that alive intervals work on a real generator output."""

    def test_alive_count_matches_active_population(self):
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator

        cfg = PartnershipConfig(num_agents=200, total_timesteps=200)
        gen = PartnershipGenerator(cfg, seed=0)
        gen.simulate_partnerships()

        alive = AliveIntervals.from_agent_log(
            gen.get_agent_log(), total_timesteps=cfg.total_timesteps
        )

        # At t=1 (first timestep), every initial-cohort agent should be alive
        assert len(alive.alive_at(1)) == cfg.num_agents

        # At t=total_timesteps, the number of alive agents should equal the active population (since population is steady-state
        # replenishments bring it back to num_agents after each removal)
        n_alive_at_end = len(alive.alive_at(cfg.total_timesteps))
        assert n_alive_at_end == int(gen.active.sum())