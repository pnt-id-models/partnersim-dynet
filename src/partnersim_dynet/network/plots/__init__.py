"""Publication-quality plots for partnership network analysis.

All plot functions:

- Take pre-computed metric DataFrames (no internal graph building)
- Write files to disk via ``save_figure`` (configurable formats)
- Wrap their drawing in ``publication_style`` (no global rcParams pollution)
"""

from partnersim_dynet.network.plots.style import (
    PALETTE,
    NetworkPalette,
    OutputFormats,
    publication_style,
    save_figure,
)
from partnersim_dynet.network.plots.timeseries import (
    SPEC_AVG_DEGREE,
    SPEC_AVG_PATH_LENGTH,
    SPEC_MAX_DEGREE,
    SPEC_TRANSITIVITY,
    TimeseriesSpec,
    plot_avg_degree,
    plot_avg_path_length,
    plot_max_degree,
    plot_timeseries,
    plot_transitivity,
)

__all__ = [
    # style
    "PALETTE",
    "NetworkPalette",
    "OutputFormats",
    "publication_style",
    "save_figure",
    # timeseries
    "SPEC_AVG_DEGREE",
    "SPEC_AVG_PATH_LENGTH",
    "SPEC_MAX_DEGREE",
    "SPEC_TRANSITIVITY",
    "TimeseriesSpec",
    "plot_avg_degree",
    "plot_avg_path_length",
    "plot_max_degree",
    "plot_timeseries",
    "plot_transitivity",
]