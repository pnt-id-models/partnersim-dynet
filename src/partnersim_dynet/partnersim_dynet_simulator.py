"""Partnership network simulation: run one simulation, or run a batch of replicates.

- ``run_single(cfg, seed, output_dir)``: one simulation, one set of
  outputs.
- ``run_replicates(sim_cfg, base_output_dir)``: multi-replicate batch
  using seeds from ``SimulationConfig.partnership_seeds()``. Runs in
  parallel if ``sim_cfg.n_workers > 1``.

Outputs
---------------
Always:
- ``partnerships.{parquet,csv}``: the partnership DataFrame
- ``agent_log.{parquet,csv}``: the agent log

If ``run_metrics=True`` or ``run_plots=True``:
- ``metrics.{parquet,csv}``: per-timestep network metrics

If ``run_degree_distributions=True``:
- ``degree_by_demographic.{parquet,csv}``
- ``degree_at_snapshots.{parquet,csv}`` (snapshot_times argument)
- ``degree_in_window.{parquet,csv}`` (full simulation window)

If ``run_plots=True``:
- Plots subdirectory with all timeseries, heatmap, and ego network
  figures.

If ``run_diagnostics=True``:
- ``base_probabilities.csv``, ``effective_bounds.csv``
- Diagnostics subdirectory with agent-probability boxplots
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import pandas as pd

from partnersim_dynet.config import PartnershipConfig, SimulationConfig
from partnersim_dynet.generator import PartnershipGenerator

logger = logging.getLogger(__name__)

0
# Run summary returned by run_single


@dataclass
class RunResult:
    """Summary of a single simulation run.

    Returned by ``run_single`` so callers (including ``run_replicates``)
    know which files were written and where. Not pickled into multiprocess
    workers; constructed in the caller process.
    """

    seed: int
    output_dir: str
    n_agents: int
    n_partnerships: int
    files_written: list[str]


# I/O helper


def _save_dataframe(df: pd.DataFrame, output_dir: str, name: str, fmt: str) -> str:
    """Save ``df`` to ``{output_dir}/{name}.{fmt}``. Returns the path."""
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"unsupported format: {fmt!r}")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


# Single run


def run_single(
    cfg: PartnershipConfig,
    seed: int,
    output_dir: str,
    *,
    output_format: str = "parquet",
    verbose: bool = False,
    run_metrics: bool = False,
    run_degree_distributions: bool = False,
    run_plots: bool = False,
    run_diagnostics: bool = False,
    snapshot_times: list[int] | None = None,
) -> RunResult:
    """Run single simulation and write outputs to ``output_dir``.

    Parameters
    ----------
    cfg : PartnershipConfig
        The simulation parameters.
    seed : int
        Random seed for this run. Reproducible.
    output_dir : str
        Where to write outputs. Created if missing.
    output_format : "parquet" | "csv"
        Format for DataFrame outputs. Parquet is recommended.
    verbose : bool
        If True, log progress at INFO level.
    run_metrics, run_degree_distributions, run_plots, run_diagnostics : bool
        Analysis toggles. ``run_plots`` implies ``run_metrics``
        internally (plots need metrics).
    snapshot_times : list of int or None
        Used only when degree distributions or plots are requested. If
        None, defaults to 5 evenly-spaced points across the simulation.

    Returns
    -------
    RunResult
        Summary of what was written.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        logger.info("Starting run: seed=%d, output_dir=%s", seed, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    files: list[str] = []

    # ── (1) Simulation ────────────────────────────────────────────────
    gen = PartnershipGenerator(cfg, seed=seed)
    partnerships_df = gen.simulate_partnerships()
    agent_log = gen.get_agent_log()
    if verbose:
        logger.info(
            "Simulation done: %d partnership rows, %d agents ever",
            len(partnerships_df),
            len(agent_log),
        )

    files.append(_save_dataframe(partnerships_df, output_dir, "partnerships", output_format))
    files.append(_save_dataframe(agent_log, output_dir, "agent_log", output_format))

    # Default snapshot_times if needed downstream
    if snapshot_times is None and (run_degree_distributions or run_plots):
        T = cfg.total_timesteps
        snapshot_times = [max(1, int(round(T * k / 4))) for k in (0, 1, 2, 3, 4)]

    # ── (2) Metrics ──────────────────────────────────────────────────
    metrics_df: pd.DataFrame | None = None
    degree_demo_df: pd.DataFrame | None = None
    need_metrics = run_metrics or run_plots
    if need_metrics:
        from partnersim_dynet.network import (
            ActiveIntervals,
            compute_temporal_metrics,
            prepare_partnerships,
        )

        active = ActiveIntervals.from_agent_log(agent_log, total_timesteps=cfg.total_timesteps)
        arr = prepare_partnerships(partnerships_df, total_timesteps=cfg.total_timesteps)
        metrics_df = compute_temporal_metrics(arr, active, total_timesteps=cfg.total_timesteps)
        files.append(_save_dataframe(metrics_df, output_dir, "metrics", output_format))
        if verbose:
            logger.info("Metrics computed: %d timesteps", len(metrics_df))

    # ── (3) Degree distributions ─────────────────────────────────────
    if run_degree_distributions:
        from partnersim_dynet.network import (
            ActiveIntervals,
            degree_at_snapshots,
            degree_by_demographic_over_time,
            degree_in_window,
            prepare_partnerships,
        )

        # Reuse active and arr if already built above; otherwise build now
        if not need_metrics:
            active = ActiveIntervals.from_agent_log(agent_log, total_timesteps=cfg.total_timesteps)
            arr = prepare_partnerships(partnerships_df, total_timesteps=cfg.total_timesteps)

        degree_demo_df = degree_by_demographic_over_time(
            arr, active, agent_log, total_timesteps=cfg.total_timesteps
        )
        files.append(
            _save_dataframe(degree_demo_df, output_dir, "degree_by_demographic", output_format)
        )

        snap_df = degree_at_snapshots(snapshot_times, arr, active, agent_log)
        files.append(_save_dataframe(snap_df, output_dir, "degree_at_snapshots", output_format))

        win_df = degree_in_window(1, cfg.total_timesteps, arr, active, agent_log)
        files.append(_save_dataframe(win_df, output_dir, "degree_in_window", output_format))

        if verbose:
            logger.info("Degree distributions computed")

    # ── (4) Plots ────────────────────────────────────────────────────
    if run_plots:
        from partnersim_dynet.network import (
            ActiveIntervals,
            degree_by_demographic_over_time,
            prepare_partnerships,
        )
        from partnersim_dynet.network.plots import (
            build_node_attr,
            build_shared_ego_layouts,
            identify_top_concurrent_agents,
            plot_avg_degree,
            plot_avg_path_length,
            plot_degree_heatmap_evolution,
            plot_ego_network_active_snapshot,
            plot_ego_network_dynamic,
            plot_ego_network_static_aggregate,
            plot_max_degree,
            plot_transitivity,
        )

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Timeseries
        for plot_fn in (plot_avg_degree, plot_max_degree, plot_transitivity, plot_avg_path_length):
            files.extend(plot_fn(metrics_df, plots_dir))

        # Heatmap — needs degree_by_demographic
        if degree_demo_df is None:
            # Compute on the fly if degree distributions weren't enabled
            if not need_metrics:
                active = ActiveIntervals.from_agent_log(
                    agent_log, total_timesteps=cfg.total_timesteps
                )
                arr = prepare_partnerships(partnerships_df, total_timesteps=cfg.total_timesteps)
            degree_demo_df = degree_by_demographic_over_time(
                arr, active, agent_log, total_timesteps=cfg.total_timesteps
            )

        files.extend(plot_degree_heatmap_evolution(degree_demo_df, snapshot_times, plots_dir))

        # Ego networks
        node_attr = build_node_attr(agent_log)
        top_agents = identify_top_concurrent_agents(partnerships_df, top_n=3)
        if top_agents:
            layouts = build_shared_ego_layouts(
                arr,
                active,
                top_agents,
                t_start=1,
                t_end=cfg.total_timesteps,
            )
            files.extend(
                plot_ego_network_dynamic(
                    partnerships_df=partnerships_df,
                    partnerships=arr,
                    active=active,
                    output_dir=plots_dir,
                    top_n=len(top_agents),
                    timesteps=snapshot_times,
                    node_attr=node_attr,
                    shared_layouts=layouts,
                )
            )
            files.extend(
                plot_ego_network_active_snapshot(
                    partnerships_df=partnerships_df,
                    partnerships=arr,
                    active=active,
                    output_dir=plots_dir,
                    top_n=len(top_agents),
                    snapshot_t=snapshot_times[-1],
                    node_attr=node_attr,
                    shared_layouts=layouts,
                )
            )
            files.extend(
                plot_ego_network_static_aggregate(
                    partnerships_df=partnerships_df,
                    partnerships=arr,
                    active=active,
                    output_dir=plots_dir,
                    top_n=len(top_agents),
                    t_start=1,
                    t_end=cfg.total_timesteps,
                    node_attr=node_attr,
                    shared_layouts=layouts,
                )
            )

        if verbose:
            logger.info("Plots written to %s", plots_dir)

    # ── (5) Diagnostics ──────────────────────────────────────────────
    if run_diagnostics:
        from partnersim_dynet.diagnostics import (
            export_probability_bounds_csv,
            plot_agent_probability_distributions,
            save_probability_table,
        )

        diag_dir = os.path.join(output_dir, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        files.append(save_probability_table(cfg, os.path.join(diag_dir, "base_probabilities.csv")))
        files.append(
            export_probability_bounds_csv(
                cfg, agent_log, os.path.join(diag_dir, "effective_bounds.csv")
            )
        )
        files.extend(plot_agent_probability_distributions(cfg, agent_log, diag_dir))

        if verbose:
            logger.info("Diagnostics written to %s", diag_dir)

    return RunResult(
        seed=seed,
        output_dir=output_dir,
        n_agents=len(agent_log),
        n_partnerships=len(partnerships_df),
        files_written=files,
    )


# Multi-replicate batch


def _run_single_worker(kwargs: dict) -> RunResult:
    """Worker function for ProcessPoolExecutor.

    Takes a dict instead of unpacking arguments because that's the
    cleanest cross-process call pattern.
    """
    return run_single(**kwargs)


def run_replicates(
    sim_cfg: SimulationConfig,
    base_output_dir: str,
    snapshot_times: list[int] | None = None,
) -> list[RunResult]:
    """Run ``sim_cfg.n_partnership_replicates`` simulations, one per seed.

    Each replicate runs ``run_single`` with one of the seeds from
    ``sim_cfg.partnership_seeds()``. Outputs go in
    ``{base_output_dir}/partnership_seed_<N>/``.

    If ``sim_cfg.n_workers > 1``, replicates run in parallel via
    ``ProcessPoolExecutor``. Set ``n_workers=1`` for serial execution
    (avoids the multiprocessing overhead, useful for debugging).

    Parameters
    ----------
    sim_cfg : SimulationConfig
        Drives all per-replicate config and analysis flags.
    base_output_dir : str
        Parent directory for the per-replicate subdirectories. Created
        if missing.
    snapshot_times : list of int or None
        Passed through to ``run_single``.

    Returns
    -------
    list of RunResult
        One per replicate, in seed order.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    seeds = sim_cfg.partnership_seeds()

    if sim_cfg.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        logger.info(
            "run_replicates: %d replicates, %d workers, seeds=%s",
            len(seeds),
            sim_cfg.n_workers,
            seeds,
        )

    # Build per-replicate
    jobs = []
    for seed in seeds:
        replicate_dir = os.path.join(base_output_dir, f"partnership_seed_{seed}")
        jobs.append(
            dict(
                cfg=sim_cfg.partnership,
                seed=int(seed),
                output_dir=replicate_dir,
                output_format=sim_cfg.output_format,
                verbose=sim_cfg.verbose,
                run_metrics=sim_cfg.run_metrics,
                run_degree_distributions=sim_cfg.run_degree_distributions,
                run_plots=sim_cfg.run_plots,
                run_diagnostics=sim_cfg.run_diagnostics,
                snapshot_times=snapshot_times,
            )
        )

    # Serial path: avoids pool overhead
    if sim_cfg.n_workers == 1:
        return [_run_single_worker(job) for job in jobs]

    # Parallel path
    results: list[RunResult] = []
    with ProcessPoolExecutor(max_workers=sim_cfg.n_workers) as executor:
        future_to_seed = {executor.submit(_run_single_worker, job): job["seed"] for job in jobs}
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                results.append(result)
                if sim_cfg.verbose:
                    logger.info("Replicate seed=%d completed", seed)
            except Exception as exc:
                logger.error("Replicate seed=%d failed: %s", seed, exc)
                raise

    # Sort by seed to give a deterministic return order regardless of which worker finished first
    results.sort(key=lambda r: r.seed)
    return results
