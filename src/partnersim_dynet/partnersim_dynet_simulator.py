"""Partnership network simulation: run one simulation, or run a batch of replicates.

- ``run_single(cfg, seed, output_dir)``: one simulation, one set of
  outputs.
- ``run_replicates(sim_cfg, base_output_dir)``: multi-replicate batch
  using seeds from ``SimulationConfig.partnership_seeds()``. Runs in
  parallel if ``sim_cfg.n_workers > 1``. Plots are generated
  only for one selected replicate

Outputs (Note: some of these files are not currently used for any purpose)
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
- Plots subdirectory with all timeseries, heatmap, and ego network figures.

If ``run_structural_summary=True``:
- ``structural_summary.{parquet,csv}``: scalar network metrics for R analysis.
  Requires run_metrics=True (or run_plots=True) to supply metrics_df.

If ``run_diagnostics=True``:
- ``base_probabilities.csv``, ``effective_bounds.csv``
- Diagnostics subdirectory with agent-probability boxplots
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd

from partnersim_dynet.config import PartnershipConfig, SimulationConfig
from partnersim_dynet.generator import PartnershipGenerator
from partnersim_dynet.network.plots.ego_network_new import identify_agents_by_spec
from partnersim_dynet.network.plots.timeseries import (
    BURN_IN_STEPS,
    CENSORING_STEPS,
    plot_degree_summary,
)

logger = logging.getLogger(__name__)


# Run summary dataclass for returning results from a single simulation run.
@dataclass
class RunResult:
    """Summary of a single simulation run."""

    seed: int
    output_dir: str
    n_agents: int
    n_partnerships: int
    files_written: list[str]


# I/O helper function to save a DataFrame in the specified format (parquet or csv).


def _save_dataframe(df: pd.DataFrame, output_dir: str, name: str, fmt: str) -> str:
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

# Specify a default seed for reproducibility if a particular set of results need to be reproduced.
seed = 380863079


# For running a single simulation, we can use the run_single function.
# It takes a PartnershipConfig, a seed, and an output directory, along with optional parameters for output format,
# and which analyses to run.
def run_single(
    cfg: PartnershipConfig,
    seed: int,
    output_dir: str,
    *,  # force keyword-only arguments for clarity
    output_format: str = "parquet",
    verbose: bool = False,
    run_metrics: bool = False,
    run_degree_distributions: bool = False,
    run_plots: bool = False,
    run_diagnostics: bool = False,
    run_summary_table: bool = True,
    snapshot_times: list[int] | None = None,
) -> RunResult:
    """Run single simulation and write outputs to ``output_dir``.

    Parameters
    ----------
    run_structural_summary : bool
        If True, compute scalar structural metrics on the steady-state
        aggregate graph and write ``structural_summary.{fmt}``.
        Requires ``run_metrics=True`` or ``run_plots=True`` so that
        ``metrics_df``, ``arr``, and ``active`` are available.
        Automatically enabled when ``run_plots=True``.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        logger.info("Starting run: seed=%d, output_dir=%s", seed, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    files: list[str] = []

    # Simulation: generate partnerships and agent log
    gen = PartnershipGenerator(cfg, seed=seed)
    partnerships_df = gen.simulate_partnerships()
    agent_log = gen.get_agent_log()
    if verbose:
        logger.info(
            "Simulation done: %d partnership rows, %d agents ever",
            len(partnerships_df),
            len(agent_log),
        )

    # Write the raw outputs to disk in the specified format (parquet or csv).
    files.append(_save_dataframe(partnerships_df, output_dir, "partnerships", output_format))
    files.append(_save_dataframe(agent_log, output_dir, "agent_log", output_format))

    if snapshot_times is None and (run_degree_distributions or run_plots):
        T = cfg.total_timesteps
        snapshot_times = [max(1, int(round(T * k / 4))) for k in (0, 1, 2, 3, 4)]

    # Metrics and degree distributions require the partnerships and agent log to be processed into a usable format
    metrics_df: pd.DataFrame | None = None
    degree_demo_df: pd.DataFrame | None = None
    arr = None
    active = None

    # If any of the metrics, plots, or summary table are requested, compute the temporal metrics.
    need_metrics = run_metrics or run_plots or run_summary_table
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

    # Calculate degree distributions if requested. This requires the partnerships and agent log to be processed into a usable format.
    if run_degree_distributions:
        from partnersim_dynet.network import (
            ActiveIntervals,
            degree_at_snapshots,
            degree_by_demographic_over_time,
            degree_in_window,
            prepare_partnerships,
        )

        if arr is None:
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

    # Build the steady-state aggregate graph if plots are requested. This is used for plotting and summary statistics.
    g_agg = None
    if run_plots:
        import networkx as nx

        t_min = int(metrics_df["t"].min())
        t_max = int(metrics_df["t"].max())
        t_ss_start = t_min + BURN_IN_STEPS
        t_ss_end = t_max - CENSORING_STEPS
        overlap = (arr.start <= t_ss_end) & (arr.end > t_ss_start)

        node_universe: set[int] = set()
        for _t in range(t_ss_start, t_ss_end + 1):
            node_universe |= active.active_at(_t)

        g_agg = nx.Graph()
        g_agg.add_nodes_from(node_universe)
        for _ai, _bi in zip(
            arr.agent[overlap].tolist(), arr.partner[overlap].tolist(), strict=False
        ):
            if int(_ai) in node_universe and int(_bi) in node_universe and _ai != _bi:
                g_agg.add_edge(int(_ai), int(_bi))

    # Plot generation if requested. This requires the metrics and degree distributions to be computed.
    if run_plots:
        from partnersim_dynet.network import (
            ActiveIntervals,
            degree_by_demographic_over_time,
            prepare_partnerships,
        )
        from partnersim_dynet.network.plots import (
            build_node_attr,
            build_shared_ego_layouts,
            plot_ego_3hop_aggregate_per_agent,
            plot_ego_3hop_snapshots,
            plot_ego_network_static_aggregate,
        )

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Timeseries summary over per-timestep network metrics.
        files.extend(plot_degree_summary(metrics_df, plots_dir))

        # Ego networks — for selected agents, plot the ego network at a few timesteps, and also the 3-hop ego network over the entire simulation.
        ego_timesteps = [1000, 1500]
        reference_t = ego_timesteps[0]

        #
        node_attr = build_node_attr(agent_log, snapshot_t=reference_t)
        eligible_now = active.active_at(reference_t)

        # Identify agents by demographic specifications for ego network plotting. The specifications are defined as tuples of (gender, partnership type).
        # The function returns a list of agent IDs that match the specifications and are currently active.
        specs = [
            ("Female", "Bisexual"),
            ("Female", "Same-sex"),
            ("Male", "Opposite-sex"),
        ]
        spec_agents = identify_agents_by_spec(
            partnerships_df,
            node_attr,
            specs,
            eligible_agents=eligible_now,
        )
        spec_agents = [a for a in spec_agents if a is not None]

        if spec_agents:
            layouts = build_shared_ego_layouts(
                arr, active, spec_agents, t_start=1, t_end=cfg.total_timesteps
            )

            files.extend(
                plot_ego_network_static_aggregate(
                    partnerships_df=partnerships_df,
                    partnerships=arr,
                    active=active,
                    output_dir=plots_dir,
                    agents=spec_agents,
                    t_start=1,
                    t_end=cfg.total_timesteps,
                    node_attr=node_attr,
                    shared_layouts=layouts,
                )
            )
            files.extend(
                plot_ego_3hop_aggregate_per_agent(
                    partnerships_df=partnerships_df,
                    partnerships=arr,
                    active=active,
                    agent_log=agent_log,
                    output_dir=str(plots_dir),
                    agents=spec_agents,
                    total_timesteps=cfg.total_timesteps,
                    k_hops=3,
                    max_nodes=100,
                )
            )
            files.extend(
                plot_ego_3hop_snapshots(
                    partnerships_df=partnerships_df,
                    partnerships=arr,
                    active=active,
                    agent_log=agent_log,
                    output_dir=str(plots_dir),
                    agents=spec_agents,
                    timesteps=ego_timesteps,
                    k_hops=3,
                    max_nodes=100,
                )
            )

        elif verbose:
            logger.info("No agents matched the specs — skipping ego network plots")

        if verbose:
            logger.info("Plots written to %s", plots_dir)

    # Summary table generation if requested. This requires the metrics to be computed, and optionally the aggregate graph if plots are requested.
    if run_summary_table:
        from partnersim_dynet.network import steady_state_summary_table

        files.extend(
            steady_state_summary_table(
                metrics_df,
                output_dir,
                g_agg=g_agg,  # g_agg is None if run_plots=False
            )
        )
        if verbose:
            logger.info("Summary table written to %s", output_dir)
    # Diagnostics generation if requested. This requires the agent log to be available.
    # The diagnostics include exporting probability bounds, plotting agent probability distributions, and saving a probability table.
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


# Single worker function for parallel execution. This function is used by the ProcessPoolExecutor to run a single simulation in a separate process.
# It takes a dictionary of keyword arguments and passes them to the run_single function.
def _run_single_worker(kwargs: dict) -> RunResult:
    return run_single(**kwargs)


# Run multiple replicates of the simulation in parallel using ProcessPoolExecutor.
def run_replicates(
    sim_cfg: SimulationConfig,
    base_output_dir: str,
    snapshot_times: list[int] | None = None,
) -> list[RunResult]:
    """Run ``sim_cfg.n_partnership_replicates`` simulations, one per seed."""
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
    # Create a list of jobs, where each job is a dictionary of parameters for a single simulation run.
    # Each job includes the configuration, seed, output directory, and other options.
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
    # If only one worker is specified, run the jobs sequentially in the main process. Otherwise, use ProcessPoolExecutor to run the jobs in parallel.
    if sim_cfg.n_workers == 1:
        return [_run_single_worker(job) for job in jobs]

    # Run the jobs in parallel using ProcessPoolExecutor. The results are collected as they complete, and any exceptions are logged.
    # The results are then sorted by seed to maintain the original order.
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

    # Sort the results by seed to maintain the original order of seeds.
    # This is done by creating a mapping from seed to its index in the original list of seeds, and then sorting the results based on this mapping.
    seed_order = {int(seed): i for i, seed in enumerate(seeds)}
    results.sort(key=lambda r: seed_order[r.seed])
    return results
