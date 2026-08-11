"""
Profile the full-scale simulation to find bottlenecks.

Runs a 15000-agent, 1875-timestep simulation under cProfile and prints
the top time-consuming functions. Used to identify candidates for
optimisation.

This was useful to identify bottlenecks such as the per-timestep
formation/breakage probability calculations, which were then optimised
by pre-computing the base probabilities and applying the per-agent
multipliers in a vectorised manner.
Usage:
poetry run python examples/profile_full.py

Optional: pass --concurrency-prop to test under concurrency.
poetry run python examples/profile_full.py --concurrency-prop 0.15
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from datetime import date
from pathlib import Path

from partnersim_dynet.config import PartnershipConfig
from partnersim_dynet.generator import PartnershipGenerator


# These arguments are for profiling the full-scale simulation.
# Concurrency can be enabled by passing --concurrency-prop. The default is 0 (no concurrency).
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=15000)
    parser.add_argument("--total-timesteps", type=int, default=1875)
    parser.add_argument(
        "--concurrency-prop",
        type=float,
        default=0.0,
        help="Concurrency proportion (default 0 = no concurrency)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=40, help="Number of top functions to print")
    args = parser.parse_args()

    cfg = PartnershipConfig(
        num_agents=args.num_agents,
        total_timesteps=args.total_timesteps,
        concurrency_prop=args.concurrency_prop,
    )

    print(
        f"Profiling: num_agents={cfg.num_agents}, "
        f"total_timesteps={cfg.total_timesteps}, "
        f"concurrency_prop={cfg.concurrency_prop}"
    )
    print()

    # Profile the simulation run using cProfile. The output will show the top functions by cumulative time and self time.
    profiler = cProfile.Profile()
    gen = PartnershipGenerator(cfg, seed=args.seed)

    print("Starting simulation under profiler...")
    wall_start = time.time()
    profiler.enable()
    df = gen.simulate_partnerships()
    profiler.disable()
    wall_elapsed = time.time() - wall_start

    print(f"Simulation done in {wall_elapsed:.1f}s")
    print(f"Generated {len(df)} partnership rows")
    print()

    # Print top functions by cumulative time
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(args.top_n)
    print("=" * 70)
    print(f"TOP {args.top_n} FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 70)
    print(stream.getvalue())

    # Print top functions by self time (excluding callees)
    # We exclude callees to focus on functions that are inherently slow, rather than those that are slow due to calling other functions.
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    stats2.print_stats(args.top_n)
    print("=" * 70)
    print(f"TOP {args.top_n} FUNCTIONS BY SELF TIME (excluding callees)")
    print("=" * 70)
    print(stream2.getvalue())

    # Build custom directory for saving the profile output, based on the concurrency proportion and number of agents.
    # Short label describing this run, used in the output folder/file name.
    FEATURE_NAME = "profile"
    concurrency_pct = round(args.concurrency_prop * 100)
    date_str = date.today().strftime("%d%b%Y")  # e.g. 11Aug2026
    stem = f"{FEATURE_NAME}_{concurrency_pct}pcconcurrency_{args.num_agents}agents_{date_str}"

    base_dir = Path(__file__).parent / "output" / "profile_full"
    serial = 1
    while (base_dir / f"{stem}_serial{serial}.prof").exists():
        serial += 1
    run_name = f"{stem}_serial{serial}"

    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / f"{run_name}.prof"
    profiler.dump_stats(str(profile_path))
    print(f"Profile saved to: {profile_path}")
    print()
    print("To inspect interactively:")
    print(
        f"  poetry run python -c \"import pstats; pstats.Stats('{profile_path}').sort_stats('cumulative').print_stats(50)\""
    )

    # # Save profile to disk for follow-up analysis
    # output_dir = Path(__file__).parent / "output"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # label = f"conc{args.concurrency_prop}" if args.concurrency_prop > 0 else "noconc"
    # profile_path = output_dir / f"profile_n{args.num_agents}_t{args.total_timesteps}_{label}.prof"
    # profiler.dump_stats(str(profile_path))
    # print(f"Profile saved to: {profile_path}")
    # print()
    # print("To inspect interactively:")
    # print(
    #     f"  poetry run python -c \"import pstats; pstats.Stats('{profile_path}').sort_stats('cumulative').print_stats(50)\""
    # )


if __name__ == "__main__":
    main()
