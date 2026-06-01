"""Concurrency selection: deciding which agents may hold multiple partnerships.

Three selection models are supported, chosen via
``PartnershipConfig.concurrency_model``:

1. **Random uniform** — pick `n_target` agents at random from the candidate
   pool, no stratification.

2. **Stratified by demographic combo** — partition candidates into
   ``(age_group, sex, orientation)`` buckets and select proportionally
   from each, so the concurrent subpopulation mirrors the demographic
   structure of the full population.

3. **Stratified + NB-filtered** — like Model 2, but within each combo
   restrict to agents whose negative-binomial formation multiplier
   exceeds ``concurrency_model_3_nb_threshold``. Concurrent agents are
   then drawn only from the high-activity tail of the heterogeneity
   distribution.

Model 1 is a simple baseline and is the default. Model 2 captures the observed demographic
pattern of concurrency without assuming it's driven by NB heterogeneity.
Model 3 tests the hypothesis that concurrency is concentrated among high-NB
agents, which is a common assumption in the literature,
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from partnersim_dynet.config import PartnershipConfig


def select_concurrent_indices(
    candidate_indices: NDArray[np.int32],
    n_target: int,
    sex_arr: NDArray[np.int8],
    ori_arr: NDArray[np.int8],
    age_group_labels: NDArray[np.str_],
    nb_mult_form: NDArray[np.float64],
    cfg: PartnershipConfig,
    rng: np.random.Generator,
) -> NDArray[np.int32]:
    """Choose which agents in `candidate_indices` should be flagged concurrent.

    Parameters
    ----------
    candidate_indices : ndarray of int32
        Internal indices of agents eligible for selection.
    n_target : int
        How many concurrent agents to select. Capped at len(candidate_indices).
    sex_arr, ori_arr : ndarray
        Generator's per-agent sex and orientation arrays (indexed by internal
        index, not by candidate position).
    age_group_labels : ndarray of str
        Pre-computed age-group label for each candidate, aligned with
        ``candidate_indices``. Pre-computing once at the call site avoids
        repeated age-bucket lookups inside this function.
    nb_mult_form : ndarray of float
        Generator's per-agent negative-binomial formation multiplier
        (indexed by internal index). Used by Model 3.
    cfg : PartnershipConfig
        Configuration. Reads ``concurrency_model`` and
        ``concurrency_model_3_nb_threshold``.
    rng : numpy.random.Generator
        Random source. Passed in (rather than created here) so the caller
        controls reproducibility.

    Returns
    -------
    ndarray of int32
        Selected internal indices, length at most ``n_target``. The actual
        return length may be less if the candidate pool is smaller than
        ``n_target`` or, for Model 3, if the NB filter is too restrictive.
    """
    n_target = min(n_target, len(candidate_indices))
    if n_target == 0:
        return np.empty(0, dtype=np.int32)

    if cfg.concurrency_model == 1:
        return _model_1_uniform_random(candidate_indices, n_target, rng)

    elif cfg.concurrency_model in (2, 3):
        return _model_2_or_3_stratified(
            candidate_indices=candidate_indices,
            n_target=n_target,
            sex_arr=sex_arr,
            ori_arr=ori_arr,
            age_group_labels=age_group_labels,
            nb_mult_form=nb_mult_form,
            cfg=cfg,
            rng=rng,
        )

    else:
        # PartnershipConfig._validate already rejects bad values
        raise ValueError(f"unknown concurrency_model: {cfg.concurrency_model}")


# Model 1 — uniform random


def _model_1_uniform_random(
    candidate_indices: NDArray[np.int32],
    n_target: int,
    rng: np.random.Generator,
) -> NDArray[np.int32]:
    """Random draw without stratification."""
    chosen = rng.choice(candidate_indices, size=n_target, replace=False)
    return chosen.astype(np.int32)


# Models 2 and 3 — stratified across (age_group, sex, ori) combos


def _model_2_or_3_stratified(
    candidate_indices: NDArray[np.int32],
    n_target: int,
    sex_arr: NDArray[np.int8],
    ori_arr: NDArray[np.int8],
    age_group_labels: NDArray[np.str_],
    nb_mult_form: NDArray[np.float64],
    cfg: PartnershipConfig,
    rng: np.random.Generator,
) -> NDArray[np.int32]:
    """Stratified selection by (age_group, sex, orientation) combo.

    Model 2 uses every candidate within its combo. Model 3 additionally
    filters to agents with ``nb_mult_form > concurrency_model_3_nb_threshold``,
    falling back to the full bucket if the filter would empty it.

    Distributes ``n_target`` selections across combos as evenly as possible.
    Any unallocated demographic quota is replenished from a leftover pool of unselected
    candidates.
    """
    # Bucket candidates by combo
    combo_buckets = _build_combo_buckets(
        candidate_indices=candidate_indices,
        sex_arr=sex_arr,
        ori_arr=ori_arr,
        age_group_labels=age_group_labels,
    )

    n_combos = len(combo_buckets)
    per_combo = max(1, n_target // n_combos)
    # Distribute the remainder across the first few combos
    remainder = n_target - per_combo * n_combos

    selected: list[int] = []
    leftovers: list[int] = []

    for bucket in combo_buckets.values():
        pool = _apply_model_3_filter(bucket, nb_mult_form, cfg)

        quota = per_combo + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1

        if len(pool) <= quota:
            selected.extend(pool)
        else:
            chosen = rng.choice(pool, size=quota, replace=False).tolist()
            selected.extend(chosen)
            # Anything we didn't pick goes into the back-fill pool
            chosen_set = set(chosen)
            leftovers.extend(i for i in pool if i not in chosen_set)

    # Back-fill if we came up short (sparse combos)
    shortfall = n_target - len(selected)
    if shortfall > 0 and leftovers:
        extra = rng.choice(
            leftovers,
            size=min(shortfall, len(leftovers)),
            replace=False,
        )
        selected.extend(extra.tolist())

    return np.array(selected[:n_target], dtype=np.int32)


def _build_combo_buckets(
    candidate_indices: NDArray[np.int32],
    sex_arr: NDArray[np.int8],
    ori_arr: NDArray[np.int8],
    age_group_labels: NDArray[np.str_],
) -> dict[tuple[str, int, int], list[int]]:
    """Group candidate indices by their (age_group, sex_code, ori_code) tuple."""
    buckets: dict[tuple[str, int, int], list[int]] = {}
    for pos, idx in enumerate(candidate_indices):
        key = (
            str(age_group_labels[pos]),
            int(sex_arr[idx]),
            int(ori_arr[idx]),
        )
        buckets.setdefault(key, []).append(int(idx))
    return buckets


def _apply_model_3_filter(
    bucket: list[int],
    nb_mult_form: NDArray[np.float64],
    cfg: PartnershipConfig,
) -> list[int]:
    """For Model 3, restrict to high-NB agents within the bucket.

    Falls back to the unfiltered demographic bucket if the NB filter empties it.
    """
    if cfg.concurrency_model != 3:
        return bucket

    threshold = cfg.concurrency_model_3_nb_threshold
    filtered = [i for i in bucket if nb_mult_form[i] > threshold]
    return filtered if filtered else bucket
