"""The PartnershipGenerator simulation engine.

Generates a dynamic partnership network over discrete timesteps. The
simulation maintains a fixed-capacity population (agents removed at
MAX_AGE are immediately replaced by new agents at REPLENISHMENT_AGE) and
records every dissolved or censored partnership as a `PartnershipRecord`.

The output is a pandas DataFrame with one row per partnership and one row
per agent who never partnered.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from partnersim_dynet.config import (
    AGE_GROUPS,
    GUARANTEED_DEBUT_AGE,
    MAX_AGE,
    MIN_AGE,
    ORI_CODE_TO_STR,
    ORIENTATION_PRIORS_FEMALE,
    ORIENTATION_PRIORS_MALE,
    PROPORTION_MALE,
    REPLENISHMENT_AGE,
    SEX_CODE_TO_STR,
    SEXUAL_DEBUT_PROBABILITIES,
    PartnershipConfig,
)
from partnersim_dynet.generator.concurrency import select_concurrent_indices
from partnersim_dynet.generator.kernels import (
    compute_breakage_events,
    fast_digitise_age_group,
    fast_normal_pdf,
)
from partnersim_dynet.generator.records import PartnershipRecord

logger = logging.getLogger(__name__)

# Numba's age-group indices map to these labels; index 6 = "75" (transient
# state before removal), index 7 = "Unknown" (defensive fallback).
_AGE_GROUP_LABELS_FOR_NUMBA = np.array([*AGE_GROUPS, "75", "Unknown"])


class PartnershipGenerator:
    """Simulates a dynamic population of sexual partnerships from a PartnershipConfig.

    Usage
    -----
    >>> from partnersim_dynet.config import PartnershipConfig
    >>> from partnersim_dynet.generator import PartnershipGenerator
    >>> cfg = PartnershipConfig(num_agents=100, total_timesteps=100)
    >>> gen = PartnershipGenerator(cfg, seed=42)
    >>> df = gen.simulate_partnerships()
    >>> df.head()

    Architecture
    ------------
    Agents are stored as structure-of-arrays (`sex_arr`, `ori_arr`,
    `age_arr`, etc.), indexed by an internal slot. An `idx2id`/`id2idx`
    mapping converts between internal slots and stable agent IDs.
    Partnerships are sparse dicts indexed by slot.

    The simulation maintains exactly `cfg.num_agents` active agents at all
    times. When an agent reaches MAX_AGE+1, they are removed and a new
    agent at REPLENISHMENT_AGE is created in the freed slot.

    Reproducibility
    ---------------
    The constructor accepts a `seed` parameter. All random draws inside
    the simulation go through `self._rng`, never through the NumPy global
    state. Two generators with the same config and seed will produce
    an identical output.
    """

    def __init__(self, cfg: PartnershipConfig, seed: int | None = None):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

        # Agent log is maintained as a list of dicts, with one entry per agent. Each entry contains demographic info plus entry/exit metadata.
        # When an agent is removed, their exit info is filled in but the record remains in the log.
        # This allows external partnerships to still produce records with the removed agent's demographics.
        self._agent_log: list[dict] = []
        self._agent_log_idx: dict[int, int] = {}

        # Capacity equals initial population size. Removed agents' slots are immediately reused for new agents, so capacity stays constant.
        self.capacity: int = cfg.num_agents

        # Internal sex code: 0=Males, 1=Females (see SEX_CODE_TO_STR).
        # Internal ori code: 0=Opposite-sex, 1=Same-sex, 2=Bisexual.
        # Internal age is an integer in [MIN_AGE, MAX_AGE]. These arrays are indexed by internal slot.
        self.sex_arr = np.full(self.capacity, -1, dtype=np.int8)
        self.ori_arr = np.full(self.capacity, -1, dtype=np.int8)
        self.age_arr = np.full(self.capacity, -1, dtype=np.int16)

        # Days since last birthday, for accurate ageing at each timestep. Agents age up when this hits 365; new agents start at a random point in the year.
        self.days_since_last_bday = np.zeros(self.capacity, dtype=np.int16)

        # Active flag: False means the slot is empty and should be ignored. True means the slot is occupied by an agent who hasn't yet reached MAX_AGE.
        # Sexually active flag: True means the agent is eligible to form partnerships.
        # Agents start sexually active if they've hit GUARANTEED_DEBUT_AGE, else they have a debut probability based on their current age.
        self.active = np.zeros(self.capacity, dtype=bool)
        self.sexually_active_arr = np.zeros(self.capacity, dtype=bool)

        # Per-agent heterogeneity multipliers, drawn once at initialisation.
        self.nb_mult_form = np.ones(self.capacity, dtype=np.float64)
        self.nb_mult_break = np.ones(self.capacity, dtype=np.float64)

        # High-activity flag (currently disabled via config default).
        self.high_active_arr = np.zeros(self.capacity, dtype=bool)

        # Partnership counts (active + external with removed partners). Maintained for quick access to single status and concurrency cap checks.
        self.partner_count_arr = np.zeros(self.capacity, dtype=np.int16)
        self.external_count_arr = np.zeros(self.capacity, dtype=np.int16)

        # Internal slot is for stable agent ID
        self.idx2id = np.zeros(self.capacity, dtype=np.int64)
        # Stable ID is for internal slot
        self.id2idx: dict[int, int] = {}
        # IDs 1..num_agents are the initial cohort; new agents get next_agent_id++
        self.next_agent_id: int = cfg.num_agents + 1
        # When no slots are freed by removal, next_free_idx advances.
        self.next_free_idx: int = cfg.num_agents

        # External partnerships are stored separately from active partnerships
        # because the partner may have been removed from the simulation (e.g. aged out at MAX_AGE) before the record was created.
        self.partnerships: list[dict[int, int]] = [{} for _ in range(self.capacity)]
        self.external_partner: list[dict[int, dict]] = [{} for _ in range(self.capacity)]

        # Metadata about removed agents, keyed by stable ID. This generates partnership records for external partnerships even after the agent has been removed.
        self.removed_agents_info: dict[int, dict] = {}

        # Agents flagged for concurrency: can hold multiple partnerships. Active_concurrent_ids tracks the currently active concurrent set
        # concurrent_agents_ids is the flagged as concurrent agents set. Max partners is the per-agent concurrency cap, drawn from a Poisson distribution..
        self.concurrent_agents_ids: set[int] = set()
        self.active_concurrent_ids: set[int] = set()
        self.concurrent_agent_max_partners: dict[int, int] = {}

        # Per-combo concurrent target counts for stratified replenishment
        # (Models 2 & 3). Maintains the demographic distribution of the initial concurrent cohort as agents age out and new ones arrive.
        self.concurrent_combo_targets: dict[tuple, int] = {}

        # Set of agent IDs currently unpartnered and sexually active. Used as the initiator pool in the formation phase.
        self.single_agents: set[int] = set()

        # Avoid recomputing base probabilities for the same (sex, ori, age_group) tuple every time.
        self._formation_base_cache: dict[tuple, float] = {}
        self._breakage_base_cache: dict[tuple, float] = {}

        # Populate the probability tables from the config once.
        self._formation_probs = cfg.probabilities.build_formation_probs()
        self._breakage_probs = cfg.probabilities.build_breakage_probs()

        # Initialise the population, concurrency flags, and agent log. This is done in a separate method to keep the constructor clean.
        self._initialise_agents()
        self._initialise_concurrency()
        self._log_initial_cohort()

    def _initialise_agents(self) -> None:
        """Populate the initial cohort with demographic priors.

        Initial age distribution is uniform across the 6 age groups
        (16-24 ... 65-74). NB heterogeneity multipliers drawn from
        NegativeBinomial(nb_r, nb_p). Sexual debut status drawn from
        the debut schedule.
        """
        # Uniform distribution across the 6 sexually active age groups. Zip the labels with the lower/upper bounds to get a list of (lo, hi) tuples for sampling.
        # The "75" and "Unknown" buckets are simulation-internal and are not used for initialisation, so we do not include them here.
        active_groups = [
            (lo, hi)
            for label, lo, hi in zip(
                AGE_GROUPS, [16, 25, 35, 45, 55, 65], [24, 34, 44, 54, 64, 74], strict=False
            )
        ]

        # Each sexually active age group gets an equal share of the initial population, with any remainder distributed one-per-group until it runs out.
        n_per_group = self.cfg.num_agents // len(active_groups)
        remainder = self.cfg.num_agents % len(active_groups)

        # Iterate through the groups in order, filling slots with agents drawn from the age range of the group until we hit num_agents.
        # This ensures the initial population is exactly num_agents even if the age groups do not divide it evenly.
        idx = 0
        for group_idx, (lo, hi) in enumerate(active_groups):
            count = n_per_group + (1 if group_idx < remainder else 0)
            for _ in range(count):
                if idx >= self.cfg.num_agents:
                    break

                # Demographics and ID mapping. Sex and orientation drawn from distributions given in probabilities.py.
                # Age uniform within the group's range. NB multipliers and concurrency flags are filled in later.
                self.age_arr[idx] = self._rng.integers(lo, hi + 1)
                self.sex_arr[idx] = 0 if self._rng.random() < PROPORTION_MALE else 1
                priors = (
                    ORIENTATION_PRIORS_MALE if self.sex_arr[idx] == 0 else ORIENTATION_PRIORS_FEMALE
                )
                self.ori_arr[idx] = self._rng.choice([0, 1, 2], p=priors)

                # Birthday phase, uniform within the year. It is 0 to 364 because we want to age up on the 365th day, not the 366th.
                # Active flag is True for all initial agents.
                self.days_since_last_bday[idx] = self._rng.uniform(0, 364)
                self.active[idx] = True

                # Sexual debut: certain after GUARANTEED_DEBUT_AGE, else use cumulative debut probability up to current age. Guaranteed debut is at 21.
                age = int(self.age_arr[idx])
                if age >= GUARANTEED_DEBUT_AGE:
                    self.sexually_active_arr[idx] = True
                else:
                    cumulative_prob = sum(
                        SEXUAL_DEBUT_PROBABILITIES.get(a, 0.0) for a in range(MIN_AGE, age + 1)
                    )
                    self.sexually_active_arr[idx] = self._rng.random() < cumulative_prob

                # ID mapping: This is done after the demographics are set so that the ID mapping reflects the actual agent in the slot.
                # The internal slot index is `idx`, and the stable agent ID is `aid`.
                # The first `num_agents` agents get IDs 1..num_agents, and new agents get incrementing IDs starting from num_agents + 1.
                aid = idx + 1
                self.idx2id[idx] = aid
                self.id2idx[aid] = idx
                if self.sexually_active_arr[idx]:
                    self.single_agents.add(aid)

                idx += 1

        # NB heterogeneity multipliers for the initial cohort
        self.nb_mult_form[: self.cfg.num_agents] = self._sample_nb(self.cfg.num_agents)
        self.nb_mult_break[: self.cfg.num_agents] = self._sample_nb(self.cfg.num_agents)

        # High-activity flag. This is currently disabled at the default, but support the config path so it can be implemented.
        if self.cfg.high_activity_proportion > 0:
            n_high = int(round(self.cfg.num_agents * self.cfg.high_activity_proportion))
            chosen = self._rng.choice(self.cfg.num_agents, size=n_high, replace=False)
            self.high_active_arr[chosen] = True

    # We use this method to separate the concurrency logic from the agent initialisation
    def _initialise_concurrency(self) -> None:
        """Flag the initial concurrency-allowed cohort.

        The number selected is `round(num_agents * concurrency_prop)`.
        The selection method depends on `cfg.concurrency_model` and
        implemented using the function in `concurrency.py`.
        """
        if self.cfg.concurrency_prop <= 0:
            return

        # Number of concurrent agents to select based on the config proportion.
        n_concurrent = int(round(self.cfg.num_agents * self.cfg.concurrency_prop))

        # Pre-compute age-group labels for the initial cohort
        age_codes = fast_digitise_age_group(self.age_arr[: self.cfg.num_agents].astype(np.int16))
        age_group_labels = _AGE_GROUP_LABELS_FOR_NUMBA[age_codes]

        # Assign attributes and select concurrent agents using the function, which applies the concurrency model logic and demographic distribution maintenance.
        # The selected indices are relative to the initial cohort.
        selected_idxs = select_concurrent_indices(
            candidate_indices=np.arange(self.cfg.num_agents, dtype=np.int32),
            n_target=n_concurrent,
            sex_arr=self.sex_arr,
            ori_arr=self.ori_arr,
            age_group_labels=age_group_labels,
            nb_mult_form=self.nb_mult_form,
            cfg=self.cfg,
            rng=self._rng,
        )

        # Flag the selected agents as concurrent and set their concurrency caps.
        # Also snapshot the per-combo targets based on the initial cohort's demographics to maintain the distribution during replenishment.
        for idx in selected_idxs:
            aid = int(self.idx2id[idx])
            self.concurrent_agents_ids.add(aid)
            self.active_concurrent_ids.add(aid)
            self.concurrent_agent_max_partners[aid] = self._draw_concurrency_cap()

            # Snapshot the per-combo target counts for stratified replenishment (Models 2 & 3).
            ag = age_group_labels[idx]
            key = (ag, int(self.sex_arr[idx]), int(self.ori_arr[idx]))
            self.concurrent_combo_targets[key] = self.concurrent_combo_targets.get(key, 0) + 1

    # Log the initial cohort after concurrency is set up, so the log reflects the correct concurrency status at timestep 1.
    def _log_initial_cohort(self) -> None:
        """Write agent log entries for the initial cohort.

        Called after `_initialise_concurrency()` so concurrency status is accurate at log time.
        """
        for idx in range(self.cfg.num_agents):
            self._log_agent_entry(idx, entry_timestep=1)

    # Draw NB heterogeneity multipliers, normalised by adding 1 so the minimum value is 1 and the mean is 2.
    # NB distribution has mean nb_r * (1 - nb_p) / nb_p.
    def _sample_nb(self, size: int) -> np.ndarray:
        """Draw NB heterogeneity multipliers, normalised so the mean is 2.

        The raw NB has mean nb_r * (1 - nb_p) / nb_p. Dividing by that
        gives a multiplier with mean 2 and ensures that the sampled values are
        always 1 and above.
        """
        raw = self._rng.negative_binomial(self.cfg.nb_r, self.cfg.nb_p, size=size)
        mean_nb = self.cfg.nb_r * (1 - self.cfg.nb_p) / self.cfg.nb_p
        return 1 + raw / mean_nb

    # Draw an agent's per-step concurrency cap, which is the maximum number of concurrent partnerships they can hold.
    # The cap is drawn from a Poisson distribution with mean `lambda_concurrency`, but it is floored at `concurrency_min_partner_cap`
    # to ensure a minimum level of concurrency for eligible agents.
    def _draw_concurrency_cap(self) -> int:
        """Draw an agent's per-step concurrency cap.

        max(concurrency_min_partner_cap, Poisson(lambda_concurrency)).
        """
        poisson_draw = int(self._rng.poisson(self.cfg.lambda_concurrency))
        return max(self.cfg.concurrency_min_partner_cap, poisson_draw)

    # Probability lookups created once per agent per timestep, so we cache the base values for each (sex, ori, age_group) tuple.
    # The final effective probability is computed by applying the NB multiplier and high-activity boost (if activated), then clipping to [prob_floor, prob_ceiling].
    def _formation_prob(self, sex_code: int, ori_code: int, age_group: str, idx: int) -> float:
        """Effective per-step formation probability for agent at `idx`.

        Applies the agent's NB multiplier and high-activity boost on top
        of the base table value, then clips the probability to [prob_floor, prob_ceiling].
        This returns the final per-step probability that an agent with the given
        demographics and heterogeneity will form a partnership, before applying the concurrency cap filter.

        """
        key = (sex_code, ori_code, age_group)
        base = self._formation_base_cache.get(key)
        if base is None:
            sex_str = SEX_CODE_TO_STR[sex_code]
            ori_str = ORI_CODE_TO_STR[ori_code]
            base = self._formation_probs[sex_str][ori_str].get(age_group, 0.0)
            # Cache the base value for future lookups to avoid repeated dictionary access and computation for the same demographic tuple.
            self._formation_base_cache[key] = base

        prob = base * self.nb_mult_form[idx]
        if self.high_active_arr[idx]:
            prob *= self.cfg.high_activity_multiplier

        return max(self.cfg.prob_floor, min(float(prob), self.cfg.prob_ceiling))

    # Precompute the base breakage probabilities for all active agents in a vectorised manner.
    # This avoids repeated lookups and allows for efficient application of NB multipliers and high-activity boosts across the entire active cohort.
    def _precompute_all_breakage_probs(
        self,
        active_idx: np.ndarray,
        age_group_codes_arr: np.ndarray,
    ) -> np.ndarray:
        """Vectorised base breakage probability lookup for all active agents.

        Returns an array of shape which is the capacity, with the per-agent base
        breakage probability filled in for active slots and 0 elsewhere.

        Pre-build a flat lookup table once (3 ori × 2 sex × 6
        age groups), then use NumPy fancy indexing to pull the right base
        for every agent in one operation. Apply the NB multiplier and
        high-activity boost (if enabled) vectorised, then clip.

        """
        breakage_probs = np.zeros(self.capacity, dtype=np.float64)
        if active_idx.size == 0:
            return breakage_probs

        # Compose a per-agent index into a flat (sex, ori, age_group) table
        sex_codes = self.sex_arr[active_idx]
        ori_codes = self.ori_arr[active_idx]
        age_codes = age_group_codes_arr[active_idx]

        # If the base table has not been built yet, build it once and cache it.
        # This avoids repeated dictionary lookups for each agent and allows for efficient vectorised access.
        if not hasattr(self, "_breakage_base_table"):
            n_age = len(_AGE_GROUP_LABELS_FOR_NUMBA)  # 8 = 6 groups + "75" + "Unknown"
            table = np.zeros((n_age, 2, 3), dtype=np.float64)  # age_group x sex x ori
            for age_code in range(n_age):
                age_label = _AGE_GROUP_LABELS_FOR_NUMBA[age_code]
                for sc in (0, 1):
                    for oc in (0, 1, 2):
                        sex_str = SEX_CODE_TO_STR[sc]
                        ori_str = ORI_CODE_TO_STR[oc]
                        table[age_code, sc, oc] = self._breakage_probs[sex_str][ori_str].get(
                            str(age_label), 0.0
                        )
            self._breakage_base_table = table
        # Fancy-index into the table: one base per active agent to efficiently compute the effective breakage probabilities in a vectorised manner.
        base_per_agent = self._breakage_base_table[age_codes, sex_codes, ori_codes]

        # Apply per-agent NB multiplier to the base breakage probabilities. This accounts for individual heterogeneity in breakage risk.
        eff = base_per_agent * self.nb_mult_break[active_idx]

        # Apply high-activity boost where applicable
        high_active_mask = self.high_active_arr[active_idx]
        if high_active_mask.any():
            eff = np.where(
                high_active_mask,
                eff * self.cfg.high_activity_multiplier,
                eff,
            )

        # Clip to [prob_floor, prob_ceiling]
        eff = np.clip(eff, self.cfg.prob_floor, self.cfg.prob_ceiling)

        # Write back into the capacity-sized output array
        breakage_probs[active_idx] = eff
        return breakage_probs

    # Agent log management: This is a list of dicts, one per agent, with demographic info and entry/exit metadata.
    # This logging is separate from the partnership records and allows for tracking of agents who have been removed from the simulation,
    # so that external partnerships can still be recorded with their demographics.

    def _log_agent_entry(self, idx: int, entry_timestep: int) -> None:
        """Record the entry of an agent in the agent log.

        Called once at agent creation. Stores demographic snapshot plus entry metadata.
        Exit fields are filled in later (or remain None if the agent survives to end-of-simulation).
        """
        aid = int(self.idx2id[idx])
        is_concurrent = aid in self.active_concurrent_ids
        self._agent_log.append(
            {
                "Agent": aid,
                "Sex": SEX_CODE_TO_STR[int(self.sex_arr[idx])],
                "Orientation": ORI_CODE_TO_STR[int(self.ori_arr[idx])],
                "EntryAge": int(self.age_arr[idx]),
                "EntryTimestep": entry_timestep,
                "ExitTimestep": None,
                "ExitAge": None,
                "NBMultiplierForm": float(self.nb_mult_form[idx]),
                "NBMultiplierBreak": float(self.nb_mult_break[idx]),
                "HighActive": bool(self.high_active_arr[idx]),
                "ConcurrencyAllowed": is_concurrent,
                "ConcurrencyCap": (
                    self.concurrent_agent_max_partners.get(aid) if is_concurrent else None
                ),
            }
        )
        self._agent_log_idx[aid] = len(self._agent_log) - 1

    # Log the exit of an agent in the agent log. This is called when an agent is removed (e.g., aged out at MAX_AGE).
    def _log_agent_exit(self, aid: int, exit_timestep: int, exit_age: int) -> None:
        """Update the agent log when an agent is removed."""
        log_idx = self._agent_log_idx.get(aid)
        if log_idx is not None:
            self._agent_log[log_idx]["ExitTimestep"] = exit_timestep
            self._agent_log[log_idx]["ExitAge"] = exit_age

    # Agent removal and replenishment but keep the same slot index. This is done to maintain a fixed-capacity population while allowing for demographic turnover.
    def _remove_agent(self, idx: int, t: int) -> None:
        """Remove agent at slot `idx` at timestep `t`.

        - Records the agent's demographic snapshot to `removed_agents_info`
          so any external partnerships can still produce records.
        - Moves the agent's internal partnerships into their partners'
          external-partner dicts (so dissolution records still get
          generated when those partnerships eventually end).
        - Updates the agent log with exit info.
        - Marks the slot inactive.
        """
        old_aid = int(self.idx2id[idx])
        self.removed_agents_info[old_aid] = {
            "sex": SEX_CODE_TO_STR[int(self.sex_arr[idx])],
            "orientation": ORI_CODE_TO_STR[int(self.ori_arr[idx])],
            "age": int(self.age_arr[idx]),
            "removed_time": t,
        }

        # Hand off internal partnerships to external tracking on the surviving partner's side.
        for partner_idx, start_time in list(self.partnerships[idx].items()):
            if self.active[partner_idx]:
                existing = self.external_partner[partner_idx].get(old_aid)
                if existing is None or start_time < existing.get("start_time", np.inf):
                    self.external_partner[partner_idx][old_aid] = {"start_time": start_time}
                self.partnerships[partner_idx].pop(idx, None)
                self.partner_count_arr[partner_idx] = max(
                    0, int(self.partner_count_arr[partner_idx]) - 1
                )
                self.external_count_arr[partner_idx] += 1
                if (
                    self.partner_count_arr[partner_idx] == 0
                    and self.sexually_active_arr[partner_idx]
                ):
                    self.single_agents.add(int(self.idx2id[partner_idx]))

        # This agent is now removed: clear their partnerships, mark inactive, remove from single_agents and active_concurrent_ids, and log exit.
        self.partnerships[idx].clear()
        self.external_partner[idx].clear()
        self.active[idx] = False
        self.single_agents.discard(old_aid)
        self.active_concurrent_ids.discard(old_aid)
        self._log_agent_exit(old_aid, exit_timestep=t, exit_age=int(self.age_arr[idx]))
        if old_aid in self.id2idx:
            del self.id2idx[old_aid]

    # Create a new agent in a freed slot, with demographics drawn from the priors and NB multipliers.
    def _add_agent(self, idx: int, t: int) -> None:
        """Create a new agent at slot `idx` at timestep `t`.

        New agents start at REPLENISHMENT_AGE. NB multipliers and
        demographic priors are drawn fresh. Concurrency is decided
        separately by `_maybe_flag_replenishment_concurrent`.
        """
        # Assign a new stable agent ID and increment the next_agent_id counter. Update the idx2id and id2idx mappings to reflect the new agent.
        new_aid = self.next_agent_id
        self.next_agent_id += 1
        self.idx2id[idx] = new_aid
        self.id2idx[new_aid] = idx

        # Demographics and NB multipliers for the new agent. Age is set to REPLENISHMENT_AGE, birthday is random
        self.age_arr[idx] = REPLENISHMENT_AGE
        self.days_since_last_bday[idx] = self._rng.uniform(0, 364)
        self.sex_arr[idx] = 0 if self._rng.random() < PROPORTION_MALE else 1
        priors = ORIENTATION_PRIORS_MALE if self.sex_arr[idx] == 0 else ORIENTATION_PRIORS_FEMALE
        self.ori_arr[idx] = self._rng.choice([0, 1, 2], p=priors)

        # Draw NB heterogeneity multipliers for formation and breakage, reset high-activity flag, clear partnerships, reset counts, mark active.
        self.nb_mult_form[idx] = self._sample_nb(1)[0]
        self.nb_mult_break[idx] = self._sample_nb(1)[0]
        self.high_active_arr[idx] = False
        self.partnerships[idx] = {}
        self.external_partner[idx] = {}
        self.partner_count_arr[idx] = 0
        self.external_count_arr[idx] = 0
        self.active[idx] = True

        # Sexual debut: new agents are exactly REPLENISHMENT_AGE, so use the age-16 debut probability.
        debut_prob = SEXUAL_DEBUT_PROBABILITIES.get(REPLENISHMENT_AGE, 0.0)
        self.sexually_active_arr[idx] = self._rng.random() < debut_prob
        if self.sexually_active_arr[idx]:
            self.single_agents.add(new_aid)

    # This method decides whether a newly replenished agent should be flagged as concurrent based on the configured concurrency model
    # and the current demographic distribution of active concurrent agents. It updates the relevant sets and counts if the agent is flagged.
    def _maybe_flag_replenishment_concurrent(
        self, idx: int, current_combo_counts: dict[tuple, int]
    ) -> None:
        """Decide whether a newly replenished agent should be flagged concurrent.

        Models 1, 2, 3 all use the same per-combo deficit logic
        """
        if self.cfg.concurrency_prop <= 0:
            return

        new_aid = int(self.idx2id[idx])
        ag = "16-24"  # replenishments always enter at REPLENISHMENT_AGE
        sc = int(self.sex_arr[idx])
        oc = int(self.ori_arr[idx])
        combo_key = (ag, sc, oc)

        # Model 1: maintain overall proportion via Bernoulli draw
        if self.cfg.concurrency_model == 1:
            current_prop = len(self.active_concurrent_ids) / max(1, int(self.active.sum()))
            should_flag = self._rng.random() < current_prop

        # Models 2 & 3: per-combo deficit replenishment
        else:
            target = self.concurrent_combo_targets.get(combo_key, 0)
            current = current_combo_counts.get(combo_key, 0)
            should_flag = current < target

            # Model 3 extra filter: only flag if this agent's NB > threshold
            if (
                self.cfg.concurrency_model == 3
                and self.nb_mult_form[idx] <= self.cfg.concurrency_model_3_nb_threshold
            ):
                should_flag = False

        if should_flag:
            self.concurrent_agents_ids.add(new_aid)
            self.active_concurrent_ids.add(new_aid)
            self.concurrent_agent_max_partners[new_aid] = self._draw_concurrency_cap()
            current_combo_counts[combo_key] = current_combo_counts.get(combo_key, 0) + 1

    # Count the currently active concurrent agents per (age_group, sex, ori) combination.
    # This is used for Models 2 & 3 to maintain the demographic distribution of concurrent agents during replenishment.
    def _count_active_concurrent_per_combo(self) -> dict[tuple, int]:
        """Count currently-active concurrent agents per (age_group, sex, ori) combo."""
        counts: dict[tuple, int] = {}
        for aid in self.active_concurrent_ids:
            if aid not in self.id2idx:
                continue
            idx = self.id2idx[aid]
            ag_code = fast_digitise_age_group(np.array([self.age_arr[idx]], dtype=np.int16))[0]
            ag = _AGE_GROUP_LABELS_FOR_NUMBA[ag_code]
            key = (str(ag), int(self.sex_arr[idx]), int(self.ori_arr[idx]))
            counts[key] = counts.get(key, 0) + 1
        return counts

    # Simulate the full partnership dynamics over the configured number of timesteps, returning a DataFrame of all partnerships formed, dissolved, or censored.
    def simulate_partnerships(self) -> pd.DataFrame:
        """
        Run the full simulation and return the partnership DataFrame.

        The DataFrame has one row per partnership (dissolved, censored at
        end-of-simulation, or singleton for never-partnered agents) plus
        all the columns of `PartnershipRecord`. Column names are
        capitalised for downstream display: ``Agent``, ``PartnerAgent``,
        ``StartTime``, etc.
        """
        partnership_data: list[PartnershipRecord] = []
        T = self.cfg.total_timesteps

        # For each timestep, we perform the following phases:
        # ageing, removal of old agents, replenishment of new agents, and building helper arrays for partnership formation and breakage.
        for t in range(1, T + 1):
            if t % 100 == 0:
                logger.info(
                    "Step %d / %d: active=%d, partnerships=%d, single=%d",
                    t,
                    T,
                    int(self.active.sum()),
                    int(self.partner_count_arr.sum() // 2),
                    len(self.single_agents),
                )

            # Aging: increment days since last birthday for all active agents. If an agent has reached 365 days, they age up by 1 year and reset their birthday counter.
            active_idx = np.where(self.active)[0]
            self.days_since_last_bday[active_idx] += 1

            # Check for agents who have reached their birthday and age them up. If they are newly age-eligible for sexual debut, check that too.
            # If the debut probability is greater than 0 and a random draw is less than that probability, mark the agent as sexually active
            # and add them to the single_agents set if they have no partners.
            birthday_mask = self.days_since_last_bday[active_idx] >= 365
            if birthday_mask.any():
                aged_indices = active_idx[birthday_mask]
                self.age_arr[aged_indices] += 1
                self.days_since_last_bday[aged_indices] = 0
                # Sexual debut check for any newly age-eligible agent.
                for idx in aged_indices:
                    if not self.sexually_active_arr[idx]:
                        age = int(self.age_arr[idx])
                        debut_prob = (
                            1.0
                            if age >= GUARANTEED_DEBUT_AGE
                            else SEXUAL_DEBUT_PROBABILITIES.get(age, 0.0)
                        )
                        if debut_prob > 0 and self._rng.random() < debut_prob:
                            self.sexually_active_arr[idx] = True
                            aid = int(self.idx2id[idx])
                            if (
                                self.partner_count_arr[idx] == 0
                                and len(self.external_partner[idx]) == 0
                            ):
                                self.single_agents.add(aid)

            # Precompute age-group codes once per timestep to optimise the breakage probability lookups.
            # This avoids repeated calls to `fast_digitise_age_group` for each agent during the breakage phase.
            age_group_codes_arr = np.empty(self.capacity, dtype=np.int32)
            age_group_codes_arr[active_idx] = fast_digitise_age_group(
                self.age_arr[active_idx].astype(np.int16)
            )

            # Removal: identify agents who have exceeded MAX_AGE and are still active.
            # Remove them from the simulation, logging their exit and updating any partnerships they had.
            removal_mask = (self.age_arr > MAX_AGE) & self.active
            to_remove_idx = list(np.where(removal_mask)[0])
            n_removed = len(to_remove_idx)
            for old_idx in to_remove_idx:
                self._remove_agent(old_idx, t)

            # Replinishment: for each removed agent, create a new agent in the same slot with demographics drawn from the priors.
            # Compute per-combo concurrent counts ONCE before the loop, then update in-place as each replenishment is processed.
            if self.cfg.concurrency_model in (2, 3):
                current_combo_counts = self._count_active_concurrent_per_combo()
            else:
                current_combo_counts = {}

            # Freed slots are the indices of the removed agents. Agents are popped from this to fill new agents into the same slots.
            # If there are no freed slots left, use next_free_idx to assign a new slot.
            freed_slots = list(to_remove_idx)
            for _ in range(n_removed):
                idx = freed_slots.pop() if freed_slots else self.next_free_idx
                if not freed_slots and idx == self.next_free_idx:
                    self.next_free_idx += 1
                # Add a new agent in the freed slot, flag them for concurrency if applicable, and log their entry.
                self._add_agent(idx, t)
                self._maybe_flag_replenishment_concurrent(idx, current_combo_counts)
                self._log_agent_entry(idx, entry_timestep=t)

            # Helper arrays for partnership formation and breakage. These are used to efficiently compute probabilities and eligibility for forming or breaking partnerships.
            agent_idx = np.where(self.active)[0]
            total_counts_arr = self.partner_count_arr + self.external_count_arr
            formed_bool = np.zeros(self.capacity, dtype=bool)

            # Precompute the effective breakage probabilities for all active agents in a vectorised manner.
            # This avoids repeated lookups and allows for efficient application of NB multipliers and high-activity boosts across the entire active cohort.
            breakage_probs_arr = self._precompute_all_breakage_probs(agent_idx, age_group_codes_arr)

            # Compatibility masks per (sex_code, ori_code) of focal agent.
            # Determines which other agents are acceptable partners.
            compat_idx: dict[tuple[int, int], np.ndarray] = {}
            for sc in (0, 1):
                for oc in (0, 1, 2):
                    if oc == 0:  # opposite-sex
                        mask = (self.sex_arr == (1 - sc)) & (
                            (self.ori_arr == 0) | (self.ori_arr == 2)
                        )
                    elif oc == 1:  # same-sex
                        mask = (self.sex_arr == sc) & ((self.ori_arr == 1) | (self.ori_arr == 2))
                    else:  # bisexual
                        mask = (
                            (self.sex_arr == sc) & ((self.ori_arr == 1) | (self.ori_arr == 2))
                        ) | (
                            (self.sex_arr == (1 - sc)) & ((self.ori_arr == 0) | (self.ori_arr == 2))
                        )
                    compat_idx[(sc, oc)] = np.where(mask)[0].astype(np.int32)

            # Eligibility: non-concurrent agents need zero partners; concurrent
            # agents need to be below their personal cap.
            base_eligible_mask = self.active & (total_counts_arr == 0) & (~formed_bool)
            for aid in self.concurrent_agents_ids:
                if aid not in self.id2idx:
                    continue
                idx = self.id2idx[aid]
                if not self.active[idx]:
                    continue
                cap = self.concurrent_agent_max_partners.get(
                    aid, self.cfg.concurrency_min_partner_cap
                )
                base_eligible_mask[idx] = total_counts_arr[idx] < cap and not formed_bool[idx]
            base_candidate_idx = np.where(base_eligible_mask)[0].astype(np.int32)

            # Partnership formation: iterate through the initiator pool, draw a partner from the candidate pool, and commit the partnership.
            current_candidate_idx = base_candidate_idx.copy()

            # Initiator pool: singles + concurrent agents not yet at capacity.
            initiators = set(self.single_agents)
            for aid in self.concurrent_agents_ids:
                if aid not in self.id2idx:
                    continue
                idx = self.id2idx[aid]
                if not self.active[idx] or not self.sexually_active_arr[idx]:
                    continue
                cap = self.concurrent_agent_max_partners.get(
                    aid, self.cfg.concurrency_min_partner_cap
                )
                total = int(self.partner_count_arr[idx]) + len(self.external_partner[idx])
                if total < cap:
                    initiators.add(aid)

            for aid in list(initiators):
                if aid not in self.id2idx:
                    continue
                agent_idx_internal = self.id2idx[aid]
                if formed_bool[agent_idx_internal]:
                    continue

                # Re-check eligibility as the candidate pool may have changed since the initiator list was built.
                current_internal_count = int(self.partner_count_arr[agent_idx_internal])
                current_external_count = len(self.external_partner[agent_idx_internal])
                # Total partner count is the sum of internal and external partnerships for the agent.
                # This is used to determine if they can form a new partnership based on their concurrency cap.
                total_partner_count = current_internal_count + current_external_count

                # Determine if the agent can attempt to form a new partnership based on their concurrency status and current partner count.
                if aid in self.concurrent_agents_ids:
                    cap = self.concurrent_agent_max_partners.get(
                        aid, self.cfg.concurrency_min_partner_cap
                    )
                    can_attempt = total_partner_count < cap
                else:
                    can_attempt = total_partner_count == 0
                if not can_attempt:
                    continue

                # Build the formation probability for this agent based on their demographics and heterogeneity.
                # If a random draw exceeds this probability, skip to the next initiator.
                sc = int(self.sex_arr[agent_idx_internal])
                oc = int(self.ori_arr[agent_idx_internal])
                age_group = _AGE_GROUP_LABELS_FOR_NUMBA[age_group_codes_arr[agent_idx_internal]]
                formation_prob = self._formation_prob(
                    sc, oc, str(age_group), idx=agent_idx_internal
                )
                if self._rng.random() >= formation_prob:
                    continue

                # Remove self from the candidate pool for this draw.
                self_pos = np.searchsorted(current_candidate_idx, agent_idx_internal)
                if (
                    self_pos < len(current_candidate_idx)
                    and current_candidate_idx[self_pos] == agent_idx_internal
                ):
                    candidate_idx = np.concatenate(
                        [
                            current_candidate_idx[:self_pos],
                            current_candidate_idx[self_pos + 1 :],
                        ]
                    )
                else:
                    candidate_idx = current_candidate_idx
                if candidate_idx.size == 0:
                    continue

                # Restrict to demographically compatible agents still in the candidate pool.
                # This ensures that the selected partner is compatible with the initiator's sex and orientation.
                candidate_idx = np.intersect1d(
                    candidate_idx, compat_idx[(sc, oc)], assume_unique=True
                )
                if candidate_idx.size == 0:
                    continue

                # Weight by Gaussian age-difference kernel.
                # The kernel is centred at 0, so the probability of forming a partnership decreases with increasing age difference.
                age_diffs = self.age_arr[candidate_idx] - int(self.age_arr[agent_idx_internal])
                weights = fast_normal_pdf(
                    age_diffs.astype(np.float64),
                    loc=0.0,
                    scale=self.cfg.age_difference_scale,
                )
                wsum = weights.sum()
                probs = weights / wsum if (wsum > 0 and not np.isnan(wsum)) else None
                partner_internal = int(self._rng.choice(candidate_idx, p=probs))

                # Commit the partnership: record the start time, update the partnership dicts,
                # increment partner counts, mark both as formed, and remove from single_agents.
                self.partnerships[agent_idx_internal][partner_internal] = t
                self.partnerships[partner_internal][agent_idx_internal] = t
                self.partner_count_arr[agent_idx_internal] += 1
                self.partner_count_arr[partner_internal] += 1
                formed_bool[agent_idx_internal] = True
                formed_bool[partner_internal] = True
                self.single_agents.discard(int(self.idx2id[agent_idx_internal]))
                self.single_agents.discard(int(self.idx2id[partner_internal]))

                # Remove both from the single candidate pool to prevent them from being selected again in this timestep.
                # This ensures that once an agent has formed a partnership, they are no longer eligible to form another in the same timestep.
                for remove_idx in (agent_idx_internal, partner_internal):
                    pos = np.searchsorted(current_candidate_idx, remove_idx)
                    if (
                        pos < len(current_candidate_idx)
                        and current_candidate_idx[pos] == remove_idx
                    ):
                        current_candidate_idx = np.concatenate(
                            [
                                current_candidate_idx[:pos],
                                current_candidate_idx[pos + 1 :],
                            ]
                        )

            # Internal breakage: for each undirected partnership, compute the effective breakage probability and determine if the partnership dissolves.
            # Collect every undirected partnership pair (a < b) into arrays, run the vectorised hazard, then process partnership dissolutions.
            internal_pairs_a, internal_pairs_b, internal_durations, internal_probs = (
                self._collect_internal_partnerships(t, agent_idx, breakage_probs_arr)
            )
            # If there are any internal partnerships, compute the dissolution events using the vectorised hazard function.
            # For each partnership that dissolves, record the dissolution and update the agent states accordingly.
            if len(internal_pairs_a) > 0:
                uniforms = self._rng.random(len(internal_pairs_a))
                dissolves = compute_breakage_events(
                    durations=internal_durations,
                    base_breakage_probs=internal_probs,
                    alpha=self.cfg.dissolution_alpha,
                    gamma=self.cfg.dissolution_gamma,
                    uniforms=uniforms,
                )
                # Process the dissolved partnerships: for each partnership that dissolves, record the dissolution event and update the agent states to reflect the end of the partnership.
                for k in np.where(dissolves)[0]:
                    a, b = int(internal_pairs_a[k]), int(internal_pairs_b[k])
                    start_time = t - int(internal_durations[k])
                    self._record_and_dissolve_internal(a, b, start_time, t, partnership_data)

            # External breakage: for each external partnership, compute the effective breakage probability and determine if the partnership dissolves.
            # Collect every external partnership into arrays, run the vectorised hazard, then process partnership dissolutions.
            # External partnerships are those where the partner has already been removed from the simulation, but the initiator still tracks them in
            # their `external_partner` dict. This allows for proper dissolution records even after one partner has exited the simulation.
            ext_focal, ext_partner_aid, ext_durations, ext_probs = (
                self._collect_external_partnerships(t, agent_idx, breakage_probs_arr)
            )
            if len(ext_focal) > 0:
                uniforms = self._rng.random(len(ext_focal))
                dissolves = compute_breakage_events(
                    durations=ext_durations,
                    base_breakage_probs=ext_probs,
                    alpha=self.cfg.dissolution_alpha,
                    gamma=self.cfg.dissolution_gamma,
                    uniforms=uniforms,
                )
                for k in np.where(dissolves)[0]:
                    focal_idx = int(ext_focal[k])
                    partner_aid = int(ext_partner_aid[k])
                    start_time = t - int(ext_durations[k])
                    self._record_and_dissolve_external(
                        focal_idx, partner_aid, start_time, t, partnership_data
                    )

            pass

        # Finalise the partnership records at the end of the simulation: add censored records for any ongoing partnerships and singleton records for agents who never partnered.
        self._add_censored_records(partnership_data)
        self._add_singleton_records(partnership_data)
        return self._build_partnership_dataframe(partnership_data)

    # Breakage phase helpers: Vectorised collection of internal and external partnerships into aligned arrays for efficient processing of breakage events.

    # Internal partnerships are undirected and stored in the `partnerships` dict of each agent.
    # Each partnership is recorded exactly once (order a < b) to avoid double-counting.

    def _collect_internal_partnerships(
        self,
        t: int,
        active_idx: np.ndarray,
        breakage_probs_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gather every undirected internal partnership into parallel arrays.

        Iterates active agents and their partnership dicts once, recording
        each undirected pair exactly once (order a < b). For
        each pair, the initiator agent's base breakage probability is used —
        the algorithm draws from the focal agent's perspective.

        Returns
        -------
        (pair_a, pair_b, durations, base_probs) : 4-tuple of ndarrays
            Aligned arrays of the same length.
        """
        pairs_a: list[int] = []
        pairs_b: list[int] = []
        durations: list[int] = []
        base_probs: list[float] = []
        for agent_internal in active_idx:
            for partner_internal, start_time in self.partnerships[agent_internal].items():
                if agent_internal >= partner_internal:
                    continue
                pairs_a.append(int(agent_internal))
                pairs_b.append(int(partner_internal))
                # Compute the duration of the partnership as the difference between the current timestep and the recorded start time.
                # This is used in the breakage hazard calculation.
                durations.append(t - int(start_time))
                base_probs.append(float(breakage_probs_arr[agent_internal]))
        return (
            np.asarray(pairs_a, dtype=np.int32),
            np.asarray(pairs_b, dtype=np.int32),
            np.asarray(durations, dtype=np.int32),
            np.asarray(base_probs, dtype=np.float64),
        )

    # External partnerships are directed and stored in the `external_partner` dict of each agent.
    # Each external partnership is recorded exactly once, and the focal agent's base breakage probability is used for the hazard calculation.
    def _collect_external_partnerships(
        self,
        t: int,
        active_idx: np.ndarray,
        breakage_probs_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gather every external partnership into parallel arrays.

        External partnerships are those where the partner has already been
        removed from the simulation; the focal agent still tracks them in
        their `external_partner` dict.

        Returns
        -------
        (focal_idx, partner_aid, durations, base_probs) : 4-tuple of ndarrays
        """
        focal: list[int] = []
        partner_aids: list[int] = []
        durations: list[int] = []
        base_probs: list[float] = []
        for agent_internal in active_idx:
            for partner_aid, meta in self.external_partner[agent_internal].items():
                focal.append(int(agent_internal))
                partner_aids.append(int(partner_aid))
                # Compute the duration of the partnership as the difference between the current timestep and the recorded start time.
                # This is used in the breakage hazard calculation.
                durations.append(t - int(meta["start_time"]))
                base_probs.append(float(breakage_probs_arr[agent_internal]))
        return (
            np.asarray(focal, dtype=np.int32),
            np.asarray(partner_aids, dtype=np.int64),
            np.asarray(durations, dtype=np.int32),
            np.asarray(base_probs, dtype=np.float64),
        )

    # Record and dissolve partnerships: These methods handle the recording of partnership dissolution events and update the agent states accordingly.
    # They create `PartnershipRecord` entries for each dissolved partnership, whether internal or external, and manage the partner counts and single agent tracking.
    def _record_and_dissolve_internal(
        self,
        a: int,
        b: int,
        start_time: int,
        end_time: int,
        out: list[PartnershipRecord],
    ) -> None:
        """Record a dissolved internal partnership and update agent state."""
        partnership_type = self._partnership_type_internal(a, b)
        out.append(
            PartnershipRecord(
                agent_id=int(self.idx2id[a]),
                agent_sex=SEX_CODE_TO_STR[int(self.sex_arr[a])],
                agent_orientation=ORI_CODE_TO_STR[int(self.ori_arr[a])],
                agent_age=int(self.age_arr[a]),
                partner_id=int(self.idx2id[b]),
                partner_sex=SEX_CODE_TO_STR[int(self.sex_arr[b])],
                partner_orientation=ORI_CODE_TO_STR[int(self.ori_arr[b])],
                partner_age=int(self.age_arr[b]),
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                relationship_type=partnership_type,
                censored=False,
                external_partner=False,
            )
        )
        self.partnerships[a].pop(b, None)
        self.partnerships[b].pop(a, None)
        self.partner_count_arr[a] = max(0, int(self.partner_count_arr[a]) - 1)
        self.partner_count_arr[b] = max(0, int(self.partner_count_arr[b]) - 1)
        if self.partner_count_arr[a] == 0 and self.sexually_active_arr[a]:
            self.single_agents.add(int(self.idx2id[a]))
        if self.partner_count_arr[b] == 0 and self.sexually_active_arr[b]:
            self.single_agents.add(int(self.idx2id[b]))

    def _record_and_dissolve_external(
        self,
        focal_idx: int,
        partner_aid: int,
        start_time: int,
        end_time: int,
        out: list[PartnershipRecord],
    ) -> None:
        """Record a dissolved external partnership and update agent state."""
        removed_info = self.removed_agents_info.get(partner_aid, {})
        focal_sex = SEX_CODE_TO_STR[int(self.sex_arr[focal_idx])]
        partner_sex = removed_info.get("sex")
        partnership_type = self._partnership_type_strs(focal_sex, partner_sex)
        out.append(
            PartnershipRecord(
                agent_id=int(self.idx2id[focal_idx]),
                agent_sex=focal_sex,
                agent_orientation=ORI_CODE_TO_STR[int(self.ori_arr[focal_idx])],
                agent_age=int(self.age_arr[focal_idx]),
                partner_id=int(partner_aid),
                partner_sex=partner_sex,
                partner_orientation=removed_info.get("orientation"),
                partner_age=removed_info.get("age"),
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                relationship_type=partnership_type,
                censored=False,
                external_partner=True,
            )
        )
        self.external_partner[focal_idx].pop(partner_aid, None)
        self.external_count_arr[focal_idx] = max(0, int(self.external_count_arr[focal_idx]) - 1)
        if (
            len(self.partnerships[focal_idx]) == 0
            and len(self.external_partner[focal_idx]) == 0
            and self.sexually_active_arr[focal_idx]
        ):
            self.single_agents.add(int(self.idx2id[focal_idx]))

    # Partnership type helpers: Determine the type of partnership based on the sexes of the two agents involved.
    # This is used for recording the relationship type in the partnership records.
    def _partnership_type_internal(self, a: int, b: int) -> str:
        return self._partnership_type_strs(
            SEX_CODE_TO_STR[int(self.sex_arr[a])],
            SEX_CODE_TO_STR[int(self.sex_arr[b])],
        )

    @staticmethod
    def _partnership_type_strs(sex_a: str, sex_b: str | None) -> str:
        if sex_a != sex_b:
            return "M-F"
        if sex_a == "Male":
            return "M-M"
        return "F-F"

    # Finalisation: censored and single agent records. At the end of the simulation, add records for
    # any partnerships that are still active (censored) and for any agents who never partnered (single agents).

    def _add_censored_records(self, out: list[PartnershipRecord]) -> None:
        """Add records for partnerships still active at end-of-simulation."""
        T = self.cfg.total_timesteps
        # Internal censored partnerships
        for idx in range(self.capacity):
            if not self.active[idx]:
                continue
            for partner_idx, start_time in self.partnerships[idx].items():
                if idx >= partner_idx:
                    continue
                partnership_type = self._partnership_type_internal(idx, partner_idx)
                out.append(
                    PartnershipRecord(
                        agent_id=int(self.idx2id[idx]),
                        agent_sex=SEX_CODE_TO_STR[int(self.sex_arr[idx])],
                        agent_orientation=ORI_CODE_TO_STR[int(self.ori_arr[idx])],
                        agent_age=int(self.age_arr[idx]),
                        partner_id=int(self.idx2id[partner_idx]),
                        partner_sex=SEX_CODE_TO_STR[int(self.sex_arr[partner_idx])],
                        partner_orientation=ORI_CODE_TO_STR[int(self.ori_arr[partner_idx])],
                        partner_age=int(self.age_arr[partner_idx]),
                        start_time=start_time,
                        end_time=T,
                        duration=T - start_time,
                        relationship_type=partnership_type,
                        censored=True,
                        external_partner=False,
                    )
                )

        # External censored partnerships
        for idx in range(self.capacity):
            if not self.active[idx]:
                continue
            for removed_aid, meta in self.external_partner[idx].items():
                start_time = meta["start_time"]
                removed_info = self.removed_agents_info.get(removed_aid, {})
                focal_sex = SEX_CODE_TO_STR[int(self.sex_arr[idx])]
                partner_sex = removed_info.get("sex")
                partnership_type = self._partnership_type_strs(focal_sex, partner_sex)
                out.append(
                    PartnershipRecord(
                        agent_id=int(self.idx2id[idx]),
                        agent_sex=focal_sex,
                        agent_orientation=ORI_CODE_TO_STR[int(self.ori_arr[idx])],
                        agent_age=int(self.age_arr[idx]),
                        partner_id=int(removed_aid),
                        partner_sex=partner_sex,
                        partner_orientation=removed_info.get("orientation"),
                        partner_age=removed_info.get("age"),
                        start_time=start_time,
                        end_time=T,
                        duration=T - start_time,
                        relationship_type=partnership_type,
                        censored=True,
                        external_partner=True,
                    )
                )

    def _add_singleton_records(self, out: list[PartnershipRecord]) -> None:
        """Add a placeholder row for any agent who never partnered.

        Two cases:
        - Still-active agents with zero partnership records (still in the
          simulation but never paired up).
        - Removed agents with zero partnership records (aged out without
          ever pairing up).
        """
        agents_in_data = {r.agent_id for r in out}

        for idx in np.where(self.active)[0]:
            aid = int(self.idx2id[idx])
            if aid in agents_in_data:
                continue
            out.append(
                PartnershipRecord(
                    agent_id=aid,
                    agent_sex=SEX_CODE_TO_STR[int(self.sex_arr[idx])],
                    agent_orientation=ORI_CODE_TO_STR[int(self.ori_arr[idx])],
                    agent_age=int(self.age_arr[idx]),
                    partner_id=None,
                    partner_sex=None,
                    partner_orientation=None,
                    partner_age=None,
                    start_time=None,
                    end_time=None,
                    duration=None,
                    relationship_type="None",
                    censored=False,
                    external_partner=False,
                )
            )

        for removed_aid, info in self.removed_agents_info.items():
            if removed_aid in agents_in_data:
                continue
            out.append(
                PartnershipRecord(
                    agent_id=removed_aid,
                    agent_sex=info.get("sex"),
                    agent_orientation=info.get("orientation"),
                    agent_age=info.get("age"),
                    partner_id=None,
                    partner_sex=None,
                    partner_orientation=None,
                    partner_age=None,
                    start_time=None,
                    end_time=None,
                    duration=None,
                    relationship_type="None",
                    censored=False,
                    external_partner=False,
                )
            )

    # Build the final partnership DataFrame from the list of PartnershipRecords.
    # This method converts the list of records into a pandas DataFrame with capitalised column names for downstream analysis and display.
    def _build_partnership_dataframe(self, records: list[PartnershipRecord]) -> pd.DataFrame:
        """Convert PartnershipRecords to a capitalised-column DataFrame."""
        df = pd.DataFrame(
            [
                {
                    "Agent": r.agent_id,
                    "AgentSex": r.agent_sex,
                    "AgentOrientation": r.agent_orientation,
                    "AgentAge": r.agent_age,
                    "PartnerAgent": r.partner_id,
                    "PartnerSex": r.partner_sex,
                    "PartnerOrientation": r.partner_orientation,
                    "PartnerAge": r.partner_age,
                    "StartTime": r.start_time,
                    "EndTime": r.end_time,
                    "Duration": r.duration,
                    "RelationshipType": r.relationship_type,
                    "Censored": r.censored,
                    "ExternalPartner": r.external_partner,
                }
                for r in records
            ]
        )
        return df

    # Agent log: This method returns the agent log as a pandas DataFrame, providing a summary of all agents who were ever in the simulation,
    # including their entry and exit timestamps, as well as immutable demographic and heterogeneity attributes.
    #
    # Active agents at the end of the simulation will have NaN values for their exit timestamps and ages.

    def get_agent_log(self) -> pd.DataFrame:
        """Return the agent log as a DataFrame.

        One row per agent who was ever in the simulation, with entry/exit
        timestamps and immutable demographic + heterogeneity attributes.

        Active-at-end agents have ``ExitTimestep`` and ``ExitAge`` as NaN.
        """
        return pd.DataFrame(self._agent_log)
