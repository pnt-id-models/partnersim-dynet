"""Shared style infrastructure for network plots.

Three things live here:

1. The color palette — one source of truth, imported by every plot module.
2. A scoped rc_context that applies publication-quality matplotlib
   settings without mutating global state. Use it as a context manager.
3. A `save_figure` helper that writes PNG / PDF / SVG outputs with
   independent toggles.

Design rationale
----------------
The old visualiser called `plt.rcParams.update(...)` at module import.
That mutates global matplotlib state for the entire Python process —
any user importing the package would silently inherit our font and
spine choices in their own unrelated plots. We use `plt.rc_context()`
instead, which restores the previous settings on exit.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib.pyplot as plt

# Color palette


@dataclass(frozen=True)
class NetworkPalette:
    """Color palette for network plots.

    Frozen dataclass: instances are immutable, so a module-level
    `PALETTE` constant can't be silently mutated by a caller.
    """

    # Orientation colors — used in ego networks, degree heatmaps, etc.
    orientation_opposite: str = "#191592"  # deep blue
    orientation_same: str = "#145825"  # deep green
    orientation_bisexual: str = "#5B1C65"  # deep purple

    # Per-metric colors for timeseries plots
    avg_degree: str = "#0E3926"  # forest green
    max_degree: str = "#E63946"  # red
    transitivity: str = "#8338EC"  # purple
    avg_path_length: str = "#54C6EB"  # light blue

    # Shape codes for sex (used in ego networks)
    male_shape: str = "o"  # circle
    female_shape: str = "s"  # square

    # General
    grid: str = "#E8E8E8"
    annotation: str = "#555555"

    def orientation_color(self, orientation: str) -> str:
        """Return the color for a given orientation string."""
        mapping = {
            "Opposite-sex": self.orientation_opposite,
            "Same-sex": self.orientation_same,
            "Bisexual": self.orientation_bisexual,
        }
        return mapping.get(orientation, "#999999")

    def sex_shape(self, sex: str) -> str:
        """Return the marker shape for a given sex string."""
        return self.male_shape if sex == "Male" else self.female_shape


# Module-level default palette. Tests and callers should treat this as
# read-only (the frozen=True enforces that).
PALETTE = NetworkPalette()


# Formatting for publication-quality plots


@contextmanager
def publication_style(font: str = "Georgia") -> Iterator[None]:
    """Temporarily apply publication-quality matplotlib settings.

    Usage
    -----
    >>> with publication_style():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3])
    ...     fig.savefig("plot.png")

    On exit, the original rcParams are restored, so this never leaks
    style choices into other code that imports the package.

    Parameters
    ----------
    font : str
        Font family to use. Serif by default. If you pass a sans-serif
        font (e.g. 'Helvetica', 'Arial', 'Calibri'), the font family is
        set to sans-serif accordingly.
    """
    is_sans = font in ("Helvetica", "Arial", "Calibri")
    settings = {
        "font.family": "sans-serif" if is_sans else "serif",
        "font.serif": [font, "DejaVu Serif"],
        "font.sans-serif": [font, "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,  # embed fonts in PDF/PS for journals
        "ps.fonttype": 42,
    }
    with plt.rc_context(settings):
        yield


# File saving


@dataclass(frozen=True)
class OutputFormats:
    """Toggle which file formats to save.

    >>> OutputFormats(png=True, pdf=True, svg=False)
    """

    png: bool = True
    pdf: bool = False
    svg: bool = False

    def any_enabled(self) -> bool:
        return self.png or self.pdf or self.svg

    @classmethod
    def all_enabled(cls) -> OutputFormats:
        """Convenience: all three formats on."""
        return cls(png=True, pdf=True, svg=True)


def save_figure(
    fig,
    output_path_base: str,
    formats: OutputFormats = OutputFormats(),
) -> list[str]:
    """Save a figure to PNG / PDF / SVG according to ``formats``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    output_path_base : str
        Path without extension. Each enabled format appends its
        own extension. For example, base "/tmp/avg_degree" with
        png=True, pdf=True writes "/tmp/avg_degree.png" and
        "/tmp/avg_degree.pdf".
    formats : OutputFormats
        Which formats to write. At least one must be enabled.

    Returns
    -------
    list of str
        Paths of files actually written.

    Raises
    ------
    ValueError
        If no formats are enabled.
    """
    if not formats.any_enabled():
        raise ValueError("OutputFormats has all formats disabled; nothing to save")

    # Ensure parent directory exists
    parent = os.path.dirname(output_path_base)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written: list[str] = []
    if formats.png:
        path = f"{output_path_base}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        written.append(path)
    if formats.pdf:
        path = f"{output_path_base}.pdf"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        written.append(path)
    if formats.svg:
        path = f"{output_path_base}.svg"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        written.append(path)

    return written
