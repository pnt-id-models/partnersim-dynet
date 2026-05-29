"""Active-interval lookups from the agent log.

The agent log gives us ground truth for each agent's lifetime in the
simulation:
    EntryTimestep <= t < ExitTimestep  which tells us whether an agent is active at t

(ExitTimestep is exclusive: an agent removed at t=500 was active at t=499
but not at t=500. Active-at-end agents have ExitTimestep = NaN, which we
treat as "active forever" by using total_timesteps + 1 as a sentinel.)

This module provides three views:

1. `ActiveIntervals.active_at(t)` -> set of agent IDs active at t
2. `ActiveIntervals.is_active(aid, t)` -> bool
3. `ActiveIntervals.bounds(aid)` -> (entry_t, exit_t) tuple

The class is built once from an agent log DataFrame and then queried
many times during network construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ActiveIntervals:
    """Efficient active-status lookups derived from the agent log.

    Use `ActiveIntervals.from_agent_log(agent_log, total_timesteps)` to
    construct. The internal representation uses NumPy arrays for vectorised queries.

    Attributes
    ----------
    agent_ids : ndarray of int64, shape (n_agents,)
        Stable agent IDs, in insertion order from the agent log.
    entry_t : ndarray of int32, shape (n_agents,)
        EntryTimestep per agent.
    exit_t : ndarray of int32, shape (n_agents,)
        ExitTimestep per agent. Active-at-end agents have
        total_timesteps + 1 as the sentinel value.
    total_timesteps : int
        The simulation's total_timesteps, used to interpret the sentinel.
    """

    agent_ids: np.ndarray
    entry_t: np.ndarray
    exit_t: np.ndarray
    total_timesteps: int

    # ── construction ──────────────────────────────────────────────────

    @classmethod
    def from_agent_log(
        cls, agent_log: pd.DataFrame, total_timesteps: int
    ) -> "ActiveIntervals":
        """Build active intervals from a generator agent log.

        Parameters
        ----------
        agent_log : DataFrame
            Output of `PartnershipGenerator.get_agent_log()`. Must contain
            columns: Agent, EntryTimestep, ExitTimestep.
        total_timesteps : int
            The simulation's total_timesteps. Active-at-end agents
            (ExitTimestep == NaN) are treated as active at every t in
            [EntryTimestep, total_timesteps], so this method assigns them
            ExitTimestep = total_timesteps + 1 internally.

        Raises
        ------
        ValueError
            If required columns are missing or if EntryTimestep contains
            non-positive values.
        """
        required = {"Agent", "EntryTimestep", "ExitTimestep"}
        missing = required - set(agent_log.columns)
        if missing:
            raise ValueError(
                f"agent_log is missing required columns: {sorted(missing)}"
            )

        if total_timesteps <= 0:
            raise ValueError(
                f"total_timesteps must be positive, got {total_timesteps}"
            )

        agent_ids = agent_log["Agent"].to_numpy(dtype=np.int64)
        entry_t = agent_log["EntryTimestep"].to_numpy(dtype=np.int32)

        # NaN ExitTimestep means "still active at end of sim"; sentinel
        # is total_timesteps + 1 so active-at-t comparisons work uniformly.
        exit_raw = agent_log["ExitTimestep"]
        sentinel = total_timesteps + 1
        exit_t = exit_raw.fillna(sentinel).to_numpy(dtype=np.int32)

        if (entry_t <= 0).any():
            bad = agent_log.loc[entry_t <= 0, "Agent"].tolist()[:5]
            raise ValueError(
                f"EntryTimestep must be positive; bad agents include {bad}"
            )

        return cls(
            agent_ids=agent_ids,
            entry_t=entry_t,
            exit_t=exit_t,
            total_timesteps=total_timesteps,
        )

    # ── queries ───────────────────────────────────────────────────────

    def active_at(self, t: int) -> set[int]:
        """Return the set of agent IDs active at timestep t.

        An agent is active iff entry_t <= t < exit_t.
        """
        mask = (self.entry_t <= t) & (t < self.exit_t)
        return set(self.agent_ids[mask].tolist())

    def active_at_array(self, t: int) -> np.ndarray:
        """Return active agent IDs at t as a NumPy array.

        Use this when you'll be doing further NumPy work; `active_at`
        returns a set which is better for membership tests.
        """
        mask = (self.entry_t <= t) & (t < self.exit_t)
        return self.agent_ids[mask]

    def is_active(self, agent_id: int, t: int) -> bool:
        """Whether `agent_id` is active at timestep `t`."""
        # Linear scan with NumPy — fine for thousands of agents because
        # NumPy comparison is vectorised. If this ever becomes a hot path
        mask = self.agent_ids == agent_id
        if not mask.any():
            return False
        idx = int(np.argmax(mask))
        return bool(self.entry_t[idx] <= t < self.exit_t[idx])

    def bounds(self, agent_id: int) -> tuple[int, int] | None:
        """Return (entry_t, exit_t) for `agent_id`, or None if absent.

        Note: exit_t may be the sentinel (total_timesteps + 1) for
        active-at-end agents.
        """
        mask = self.agent_ids == agent_id
        if not mask.any():
            return None
        idx = int(np.argmax(mask))
        return (int(self.entry_t[idx]), int(self.exit_t[idx]))

    def __len__(self) -> int:
        """Total number of agents ever in the simulation."""
        return len(self.agent_ids)

    def __repr__(self) -> str:
        return (
            f"ActiveIntervals(n_agents={len(self)}, "
            f"total_timesteps={self.total_timesteps})"
        )