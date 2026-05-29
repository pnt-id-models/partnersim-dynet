"""Tests for ActiveIntervals - Active-statuslookups from the agent log."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partnersim_dynet.network import ActiveIntervals

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
        active = ActiveIntervals.from_agent_log(log, total_timesteps=1000)
        assert len(active) == 1
        assert active.total_timesteps == 1000

    def test_nan_exit_becomes_sentinel(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1, "ExitTimestep": float("nan")},
        ])
        active = ActiveIntervals.from_agent_log(log, total_timesteps=1000)
        # total_timesteps + 1
        assert active.exit_t[0] == 1001

    def test_missing_column_raises(self):
        log = pd.DataFrame({
            "Agent": [1, 2],
            "EntryTimestep": [1, 2],
            # ExitTimestep missing
        })
        with pytest.raises(ValueError, match="missing required columns"):
            ActiveIntervals.from_agent_log(log, total_timesteps=100)

    def test_zero_total_timesteps_raises(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1, "ExitTimestep": 100.0},
        ])
        with pytest.raises(ValueError, match="total_timesteps"):
            ActiveIntervals.from_agent_log(log, total_timesteps=0)

    def test_zero_entry_timestep_raises(self):
        # EntryTimestep should be 1-indexed per the generator's convention
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 0, "ExitTimestep": 100.0},
        ])
        with pytest.raises(ValueError, match="EntryTimestep"):
            ActiveIntervals.from_agent_log(log, total_timesteps=100)

# Query semantics
class TestActiveQueries:
    @pytest.fixture
    def active(self) -> ActiveIntervals:
        # Three agents:
        #   1: active throughout [1, 1000]   (active at end)
        #   2: active during    [1, 500)     (exited at t=500)
        #   3: active during    [200, 800)   (joined later, exited mid-run)
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1,   "ExitTimestep": float("nan")},
            {"Agent": 2, "EntryTimestep": 1,   "ExitTimestep": 500.0},
            {"Agent": 3, "EntryTimestep": 200, "ExitTimestep": 800.0},
        ])
        return ActiveIntervals.from_agent_log(log, total_timesteps=1000)

    def test_active_at_start(self, active):
        assert active.active_at(1) == {1, 2}

    def test_active_at_mid_run(self, active):
        # At t=300, agents 1, 2 (still active), and 3 (joined at 200)
        assert active.active_at(300) == {1, 2, 3}

    def test_exit_is_exclusive(self, active):
        # Agent 2 exits at t=500. Active at 499, not at 500.
        assert 2 in active.active_at(499)
        assert 2 not in active.active_at(500)

    def test_entry_is_inclusive(self, active):
        # Agent 3 enters at t=200. Active at 200, not at 199.
        assert 3 not in active.active_at(199)
        assert 3 in active.active_at(200)

    def test_active_at_end(self, active):
        # At t=1000, only agent 1 (the sentinel-exit one)
        assert active.active_at(1000) == {1}

    def test_is_active_matches_active_at(self, active):
        for t in (1, 100, 199, 200, 499, 500, 800, 1000):
            for aid in (1, 2, 3):
                assert active.is_active(aid, t) == (aid in active.active_at(t))

    def test_is_active_unknown_agent_returns_false(self, active):
        assert active.is_active(999, t=50) is False

    def test_bounds_returns_entry_exit(self, active):
        assert active.bounds(1) == (1, 1001)   # sentinel for active-at-end
        assert active.bounds(2) == (1, 500)
        assert active.bounds(3) == (200, 800)

    def test_bounds_unknown_agent_returns_none(self, active):
        assert active.bounds(999) is None

# Array-form query

class TestActiveAtArray:
    def test_returns_ndarray(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 1, "ExitTimestep": 100.0},
            {"Agent": 2, "EntryTimestep": 1, "ExitTimestep": 100.0},
        ])
        active = ActiveIntervals.from_agent_log(log, total_timesteps=200)
        arr = active.active_at_array(50)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(sorted(arr), [1, 2])

    def test_empty_when_nobody_active(self):
        log = _make_log([
            {"Agent": 1, "EntryTimestep": 100, "ExitTimestep": 200.0},
        ])
        active = ActiveIntervals.from_agent_log(log, total_timesteps=500)
        # At t=50, agent 1 hasn't joined yet
        arr = active.active_at_array(50)
        assert len(arr) == 0

# Integration with generator

class TestIntegrationWithGenerator:
    """Spot-check that active intervals work on a real generator output."""

    def test_active_count_matches_active_population(self):
        from partnersim_dynet.config import PartnershipConfig
        from partnersim_dynet.generator import PartnershipGenerator

        cfg = PartnershipConfig(num_agents=200, total_timesteps=200)
        gen = PartnershipGenerator(cfg, seed=0)
        gen.simulate_partnerships()

        active = ActiveIntervals.from_agent_log(
            gen.get_agent_log(), total_timesteps=cfg.total_timesteps
        )

        # At t=1 (first timestep), every initial-cohort agent should be active
        assert len(active.active_at(1)) == cfg.num_agents

        # At t=total_timesteps, the number of active agents should equal the active population (since population is steady-state
        # replenishments bring it back to num_agents after each removal)
        n_active_at_end = len(active.active_at(cfg.total_timesteps))
        assert n_active_at_end == int(gen.active.sum())