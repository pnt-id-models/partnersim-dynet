"""Performance benchmark with regression detection.

The benchmark runs a fixed-size simulation and records wall time. It
compares against a stored baseline and warns (or fails) if the run is
dramatically slower.

Three goals:
1. Track absolute performance over time so we know how fast the
   simulation actually is.
2. Catch performance regressions from code changes before they merge.
3. Tolerate normal hardware variation (CI runners, laptops, desktops).

Usage:
- Run the simulation, record wall time.
- Compare against a baseline stored in BENCHMARK_BASELINE_SECONDS.
- WARN if the ratio > REGRESSION_WARN_THRESHOLD.
- FAIL only if the ratio > REGRESSION_FAIL_THRESHOLD (severe regression).

Update the baseline manually whenever a real optimisation lands —
otherwise the warning catches anything that creeps in.
"""

from __future__ import annotations

import time
import warnings

import pytest

from partnersim_dynet.config import PartnershipConfig
from partnersim_dynet.generator import PartnershipGenerator


# Configuration


# Baseline: time in seconds for the benchmark simulation on the original development machine. 
# Update this manually after intentional performance changes. 
BENCHMARK_BASELINE_SECONDS: float = 15.0

# Ratios over the baseline that trigger different responses.
REGRESSION_WARN_THRESHOLD: float = 1.5    # 50% slower → warn
REGRESSION_FAIL_THRESHOLD: float = 5.0    # 5x slower → fail

# How many runs to average over
N_REPEATS: int = 5


# The benchmark

@pytest.mark.slow
@pytest.mark.benchmark
def test_simulation_performance():
    """Run a fixed simulation N_REPEATS times, report timing, warn or
    fail if it's significantly slower than the baseline."""
    cfg = PartnershipConfig(
        num_agents=500,
        total_timesteps=500,
        concurrency_prop=0.10,
        concurrency_model=1,
    )

    times = []
    for trial in range(N_REPEATS):
        gen = PartnershipGenerator(cfg, seed=42 + trial)
        start = time.perf_counter()
        gen.simulate_partnerships()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = sorted(times)[len(times) // 2]
    ratio = median_time / BENCHMARK_BASELINE_SECONDS

    # Always print a one-line summary, even when the test passes
    print(
        f"\n[benchmark] median={median_time:.2f}s, "
        f"baseline={BENCHMARK_BASELINE_SECONDS:.2f}s, "
        f"ratio={ratio:.2f}x  "
        f"(all trials: {[f'{t:.2f}s' for t in times]})"
    )

    if ratio > REGRESSION_FAIL_THRESHOLD:
        pytest.fail(
            f"SEVERE regression: simulation took {median_time:.2f}s "
            f"({ratio:.1f}x slower than baseline {BENCHMARK_BASELINE_SECONDS:.2f}s). "
            f"Investigate before merging."
        )
    elif ratio > REGRESSION_WARN_THRESHOLD:
        warnings.warn(
            f"Performance regression: simulation took {median_time:.2f}s "
            f"({ratio:.1f}x slower than baseline {BENCHMARK_BASELINE_SECONDS:.2f}s). "
            f"If this is a CI/hardware variation, update BENCHMARK_BASELINE_SECONDS. "
            f"If this is from a code change, investigate before merging.",
            stacklevel=2,
        )


@pytest.mark.slow
@pytest.mark.benchmark
def test_simulation_memory_overhead():
    """Simulation should not consume runaway memory.

    Rough check: after a 500x500 simulation, the agent log + partnership
    DataFrame combined should be well under 100MB. This catches
    accidental quadratic-memory bugs (e.g. capturing per-timestep
    snapshots in a list).
    """
    cfg = PartnershipConfig(
        num_agents=500,
        total_timesteps=500,
    )
    gen = PartnershipGenerator(cfg, seed=42)
    partnerships = gen.simulate_partnerships()
    agent_log = gen.get_agent_log()

    # Approximate memory: pandas memory_usage returns bytes per column
    p_bytes = partnerships.memory_usage(deep=True).sum()
    l_bytes = agent_log.memory_usage(deep=True).sum()
    total_mb = (p_bytes + l_bytes) / 1024 / 1024

    print(
        f"\n[memory] partnerships={p_bytes / 1024 / 1024:.1f}MB, "
        f"agent_log={l_bytes / 1024 / 1024:.1f}MB, total={total_mb:.1f}MB"
    )

    # 100MB is a generous bound for 500 agents × 500 steps. If we exceed
    # this, something is wrong.
    assert total_mb < 100, (
        f"Memory usage {total_mb:.1f}MB exceeds 100MB ceiling. "
        f"Likely a quadratic-memory regression."
    )