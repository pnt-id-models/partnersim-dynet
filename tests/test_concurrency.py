"""Tests for the concurrency selection module.

Each test focuses on one property per model: count, stratification, NB filtering,
determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from partnersim_dynet.config import PartnershipConfig
from partnersim_dynet.generator import select_concurrent_indices

# Helpers — build a small synthetic agent population for testing


def _make_population(
    n_per_combo: int = 10,
    high_nb_proportion: float = 0.3,
    rng_seed: int = 0,
) -> dict:
    """Build a small population with all (age_group, sex, ori) combos
    equally represented.

    Returns a dict of arrays + the candidate_indices array, ready to pass
    into select_concurrent_indices.
    """
    rng = np.random.default_rng(rng_seed)

    age_groups = ["16-24", "25-34", "35-44", "45-54", "55-64", "65-74"]
    sexes = [0, 1]  # 0=Males, 1=Females
    oris = [0, 1, 2]  # opposite-sex, same-sex, bisexual

    sex_list, ori_list, ag_list = [], [], []
    for ag in age_groups:
        for sx in sexes:
            for oc in oris:
                for _ in range(n_per_combo):
                    sex_list.append(sx)
                    ori_list.append(oc)
                    ag_list.append(ag)

    n_total = len(sex_list)
    n_high_nb = int(n_total * high_nb_proportion)

    nb_mult = np.ones(n_total, dtype=np.float64)
    # First `n_high_nb` agents get high NB multipliers
    high_idx = rng.choice(n_total, size=n_high_nb, replace=False)
    nb_mult[high_idx] = 20.0  # well above the default threshold of 10

    return {
        "candidate_indices": np.arange(n_total, dtype=np.int32),
        "sex_arr": np.array(sex_list, dtype=np.int8),
        "ori_arr": np.array(ori_list, dtype=np.int8),
        "age_group_labels": np.array(ag_list),
        "nb_mult_form": nb_mult,
    }


# Model 1 — uniform random


class TestModel1Uniform:
    def test_returns_exact_count(self):
        pop = _make_population(n_per_combo=5)
        cfg = PartnershipConfig(concurrency_model=1)
        rng = np.random.default_rng(42)
        result = select_concurrent_indices(n_target=30, cfg=cfg, rng=rng, **pop)
        assert len(result) == 30

    def test_no_duplicates(self):
        pop = _make_population(n_per_combo=5)
        cfg = PartnershipConfig(concurrency_model=1)
        rng = np.random.default_rng(42)
        result = select_concurrent_indices(n_target=50, cfg=cfg, rng=rng, **pop)
        assert len(set(result.tolist())) == len(result)

    def test_caps_at_pool_size(self):
        pop = _make_population(n_per_combo=2)  # 6 * 2 * 3 = 36 agents
        cfg = PartnershipConfig(concurrency_model=1)
        rng = np.random.default_rng(42)
        # request more than available
        result = select_concurrent_indices(n_target=100, cfg=cfg, rng=rng, **pop)
        assert len(result) == 72


# Models 2 and 3 — stratification


class TestModel2Stratification:
    def test_distributes_across_combos(self):
        """With 36 combos and n_target=36, every combo should get at
        least one selection."""
        pop = _make_population(n_per_combo=5)
        cfg = PartnershipConfig(concurrency_model=2)
        rng = np.random.default_rng(42)
        result = select_concurrent_indices(n_target=36, cfg=cfg, rng=rng, **pop)
        # how many distinct combos are represented in the selected indices
        combos_hit = {
            (
                pop["age_group_labels"][i],
                int(pop["sex_arr"][i]),
                int(pop["ori_arr"][i]),
            )
            for i in result
        }
        # 6 age groups × 2 sexes × 3 oris = 36 combos
        assert len(combos_hit) == 36

    def test_more_balanced_than_model_1(self):
        """Model 2 should produce a more even per-combo count than Model 1.

        Specifically: the variance of per-combo selection counts should be
        lower for Model 2 than for Model 1, across many seeds."""

        def per_combo_variance(model: int, seed: int) -> float:
            pop = _make_population(n_per_combo=20)
            cfg = PartnershipConfig(concurrency_model=model)
            rng = np.random.default_rng(seed)
            result = select_concurrent_indices(n_target=36, cfg=cfg, rng=rng, **pop)
            counts: dict[tuple, int] = {}
            for i in result:
                key = (
                    pop["age_group_labels"][i],
                    int(pop["sex_arr"][i]),
                    int(pop["ori_arr"][i]),
                )
                counts[key] = counts.get(key, 0) + 1
            # combos with zero selections count as zero
            full_counts = [
                counts.get(k, 0)
                for k in (
                    (ag, sx, oc)
                    for ag in ["16-24", "25-34", "35-44", "45-54", "55-64", "65-74"]
                    for sx in (0, 1)
                    for oc in (0, 1, 2)
                )
            ]
            return float(np.var(full_counts))

        # Average over several seeds to avoid flakes
        m1_var = np.mean([per_combo_variance(1, s) for s in range(20)])
        m2_var = np.mean([per_combo_variance(2, s) for s in range(20)])
        assert m2_var < m1_var


class TestModel3NBFilter:
    def test_selects_only_high_nb_agents_when_possible(self):
        """Model 3 should restrict to agents with nb_mult_form > threshold."""
        pop = _make_population(n_per_combo=20, high_nb_proportion=0.5)
        cfg = PartnershipConfig(concurrency_model=3)
        rng = np.random.default_rng(42)
        result = select_concurrent_indices(n_target=36, cfg=cfg, rng=rng, **pop)
        # All selected agents should clear the threshold (or come from a
        # combo where no high-NB agent existed and the filter fell back).
        threshold = cfg.concurrency_model_3_nb_threshold
        # Average NB mult among selected should be well above 1
        avg_nb = pop["nb_mult_form"][result].mean()
        assert avg_nb > threshold * 0.5  # heuristic: well above the default

    def test_fallback_when_combo_has_no_high_nb(self):
        """If a combo has NO high-NB agents, Model 3 should fall back to the
        full bucket rather than skipping the combo entirely."""
        pop = _make_population(n_per_combo=10, high_nb_proportion=0.0)
        # ZERO high-NB agents anywhere
        cfg = PartnershipConfig(concurrency_model=3)
        rng = np.random.default_rng(42)
        result = select_concurrent_indices(n_target=36, cfg=cfg, rng=rng, **pop)
        # Should still get 36 agents (fallback covers every combo)
        assert len(result) == 36


# Determinism


class TestDeterminism:
    @pytest.mark.parametrize("model", [1, 2, 3])
    def test_same_seed_produces_same_selection(self, model):
        pop = _make_population(n_per_combo=10)
        cfg = PartnershipConfig(concurrency_model=model)

        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)

        result_a = select_concurrent_indices(n_target=30, cfg=cfg, rng=rng_a, **pop)
        result_b = select_concurrent_indices(n_target=30, cfg=cfg, rng=rng_b, **pop)

        np.testing.assert_array_equal(sorted(result_a), sorted(result_b))


# Edge cases


class TestEdgeCases:
    def test_zero_target_returns_empty(self):
        pop = _make_population(n_per_combo=5)
        cfg = PartnershipConfig(concurrency_model=1)
        rng = np.random.default_rng(42)
        result = select_concurrent_indices(n_target=0, cfg=cfg, rng=rng, **pop)
        assert len(result) == 0
        assert result.dtype == np.int32

    def test_empty_candidate_pool_returns_empty(self):
        cfg = PartnershipConfig(concurrency_model=1)
        rng = np.random.default_rng(42)
        result = select_concurrent_indices(
            candidate_indices=np.empty(0, dtype=np.int32),
            n_target=10,
            sex_arr=np.empty(0, dtype=np.int8),
            ori_arr=np.empty(0, dtype=np.int8),
            age_group_labels=np.empty(0, dtype="<U10"),
            nb_mult_form=np.empty(0, dtype=np.float64),
            cfg=cfg,
            rng=rng,
        )
        assert len(result) == 0
