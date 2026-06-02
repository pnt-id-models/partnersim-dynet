"""Input/output utilities for partnersim-dynet experiments.

Exports one helper for creating structured output directories.

"""

from __future__ import annotations

import os
import re
from datetime import date


def make_experiment_dir(
    base_dir: str,
    prefix: str,
    *,
    date_str: str | None = None,
    max_attempts: int = 100,
) -> str:
    """Create a uniquely-named experiment directory and return its path.

    The directory name has the form ``<prefix>_<date>_Run#<N>`` where:
    - ``date`` is today's date in ISO format (YYYY-MM-DD) unless
      ``date_str`` is provided
    - ``N`` is the smallest positive integer such that the directory
      doesn't already exist

    If two processes try to create directories simultaneously and
    both pick the same N, only one succeeds. The other will get a
    FileExistsError, bump to the next N, and try again. This handles
    the common case of multiple runs starting on the same day without
    needing to scan the directory with locks

    Parameters
    ----------
    base_dir : str
        Parent directory. Created if missing.
    prefix : str
        Experiment label, e.g. "female_hetero_baseline_without_concurrency".
        Forward slashes and whitespace are not allowed (use underscores
        or hyphens).
    date_str : str or None
        Override the date portion (useful for testing). If None, today's
        date in ISO format is used.
    max_attempts : int
        Maximum number of serial values to try before giving up. Should
        never matter in practice; if you have 100 collisions on the
        same prefix+date you have other problems.

    Returns
    -------
    str
        Absolute path of the directory.

    Raises
    ------
    ValueError
        If ``prefix`` is empty or contains forbidden characters.
    OSError
        If the directory can't be created after ``max_attempts`` tries.

    Examples
    --------
    >>> path = make_experiment_dir(
    ...     base_dir="results",
    ...     prefix="female_hetero_baseline",
    ... )
    >>> path
    'results/female_hetero_baseline_2026-06-15_Run#1'

    >>> # Second call on the same day creates Run#2
    >>> path2 = make_experiment_dir(
    ...     base_dir="results",
    ...     prefix="female_hetero_baseline",
    ... )
    >>> path2
    'results/female_hetero_baseline_2026-06-15_Run#2'
    """
    if not prefix:
        raise ValueError("prefix must be non-empty")
    if re.search(r"[\s/\\]", prefix):
        raise ValueError(f"prefix must not contain whitespace or slashes; got {prefix!r}")

    if date_str is None:
        date_str = date.today().isoformat()

    os.makedirs(base_dir, exist_ok=True)

    # Look at existing entries to determine the starting serial.

    pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(date_str)}_Run#(\d+)$")
    existing_serials: set[int] = set()
    if os.path.isdir(base_dir):
        for entry in os.listdir(base_dir):
            match = pattern.match(entry)
            if match:
                existing_serials.add(int(match.group(1)))

    next_serial = max(existing_serials, default=0) + 1

    # Try to create. If race condition strikes, bump and retry.
    for _attempt in range(max_attempts):
        candidate = os.path.join(base_dir, f"{prefix}_{date_str}_Run#{next_serial}")
        try:
            os.makedirs(candidate, exist_ok=False)
            return os.path.abspath(candidate)
        except FileExistsError:
            next_serial += 1

    raise OSError(
        f"Could not create experiment directory after {max_attempts} "
        f"attempts (last tried: {candidate}). This suggests massive "
        f"directory contention or a bug in the serial-finding logic."
    )
